# -*- coding: utf-8 -*-
"""
批量处理入口。

流程：
    1. 读取 config.yaml，拿到水印区域以及修复参数
    2. 遍历 input 下所有 .mp4
    3. 每个视频把水印附近裁出来，交给修复后端（LaMa 逐帧 / STTN 整段），
       再带羽化贴回原帧，最后用 imageio-ffmpeg 编码成 H.264 无声视频
    4. 输出到 output 目录，分辨率和原视频保持一致

用法：
    python main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm

from inpaint import InpaintEngine
from temporal import temporal_smooth
from utils import (DEFAULT_CONFIG_PATH, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR,
                   VideoReader, check_resolution, format_duration, list_videos,
                   load_config, scale_region, write_video_ffmpeg)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


Region = Tuple[int, int, int, int]


class ProcessingCancelled(Exception):
    """Web 界面上点了「取消」。命令行用不到这个异常，正常流程不会抛。"""


def print_env_info() -> None:
    """打印一下 PyTorch / GPU 情况，方便排查环境问题。"""
    try:
        import torch
        print("[信息] PyTorch %s，CUDA 可用：%s" % (torch.__version__, torch.cuda.is_available()))
        if torch.cuda.is_available():
            print("[信息] GPU：%s" % torch.cuda.get_device_name(0))
    except Exception as err:
        print("[警告] 读取 PyTorch 信息失败：%s" % err)


def process_one_video(video_path: Path, output_dir: Path, engine: InpaintEngine,
                      region: Region, ref_size: Tuple[int, int], output_cfg: Dict,
                      temporal_cfg: Dict, progress_cb=None, cancel_check=None,
                      stage_times: Optional[Dict[str, float]] = None) -> Path:
    """
    处理单个视频，返回输出文件路径。

    出错时会把写了一半的输出文件删掉，避免 output 目录里留下坏文件。

    region 是在 ref_size（框选时的参考分辨率）上量的绝对像素，
    本视频分辨率不一样的话，先按比例换算成当前视频的坐标再用。

    分四步走：
        1. 逐帧把水印附近裁出来，帧和对应的 patch 都存进列表；
        2. 把这一摞 patch 交给修复后端（LaMa 逐帧独立修，STTN 整段一起修）；
        3. 修复结果沿时间轴平滑一次（temporal_smooth），压掉帧间闪烁；
        4. 带羽化贴回各帧，顺序喂给 ffmpeg 编码。

    代价是要把整个视频的帧缓存在内存里（1080p 大约每 100 帧 0.6GB）。
    视频都很短的话没问题，遇到几万帧的长视频就得再想办法分块处理了。

    两个可选回调是给 Web 界面用的，命令行不传、行为和以前完全一样：
        progress_cb(phase, done, total)：phase 是 read / repair / write，
            每读完或编码完一帧调一次，用来刷前端进度条；
        cancel_check()：用户点了取消时抛出 ProcessingCancelled，
            写了一半的输出文件会被下面的 except 清掉。

    stage_times 传一个 dict 进来，会把各阶段的耗时（秒）填进去，
    Web 界面拿它在汇总表里显示「读帧 / 修复 / 平滑 / 编码」各花了多久。
    命令行不传，行为和以前一样。
    """
    output_path = output_dir / (video_path.stem + ".mp4")
    timings: Dict[str, float] = {} if stage_times is None else stage_times
    _stage = time.perf_counter()

    def mark(name: str) -> None:
        nonlocal _stage
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + (now - _stage)
        _stage = now

    reader = VideoReader(video_path)
    try:
        check_resolution(reader.width, reader.height)

        video_region = scale_region(region, ref_size[0], ref_size[1],
                                    reader.width, reader.height)
        if video_region != region:
            print("    [信息] 分辨率 %dx%d 与框选时（%dx%d）不同，水印区域已换算为 x=%d, y=%d, w=%d, h=%d"
                  % (reader.width, reader.height, ref_size[0], ref_size[1], *video_region))

        total = reader.frame_count if reader.frame_count > 0 else None

        # --- 第一步：逐帧裁出要修的那块图，帧和 patch 分别存起来 ---
        # 同一个视频里裁的位置是固定的，算一次就够，后面直接按框切片。
        frames = []
        patches = []
        patch_box = paste_box = None
        bar = tqdm(reader.frames(),
                   total=total,
                   desc=video_path.stem,
                   unit="帧",
                   ncols=88,
                   smoothing=0.1)
        for frame in bar:
            if patch_box is None:
                _, patch_box, paste_box = engine.crop_patch(frame, video_region)
                if patch_box != paste_box:
                    print("    [信息] %s：修复区域 x=%d, y=%d, w=%d, h=%d，"
                          "另外裁了一条 x=%d, y=%d, w=%d, h=%d 的带子做时间参考。"
                          % (engine.method,
                             paste_box[0], paste_box[1], paste_box[2] - paste_box[0],
                             paste_box[3] - paste_box[1],
                             patch_box[0], patch_box[1], patch_box[2] - patch_box[0],
                             patch_box[3] - patch_box[1]))
            x0, y0, x1, y1 = patch_box
            patches.append(frame[y0:y1, x0:x1])
            frames.append(frame)
            if progress_cb is not None:
                progress_cb("read", len(frames), total)
            if cancel_check is not None:
                cancel_check()

        if not frames:
            raise RuntimeError("没有从视频里读到任何帧")
        mark("读帧裁切")

        # --- 第二步：交给修复后端 ---
        if progress_cb is not None:
            progress_cb("repair", 0, len(frames))
        if cancel_check is not None:
            cancel_check()
        fixed = engine.repair_patches(patches, video_region, patch_box, paste_box)
        mark("模型修复")
        # patch 都是原帧的切片，repair 用的新数组已经出来了，这里可以早点放掉
        del patches

        # --- 第三步：修复结果沿时间轴平滑 ---
        if temporal_cfg.get("enable") and len(fixed) >= 2:
            window = int(temporal_cfg.get("window", 3))
            if window > 1:
                fixed = temporal_smooth(fixed, window=window, method="median")
                print("    [信息] 已对 %d 帧的修复结果做时间轴中值平滑（window=%d）"
                      % (len(frames), window))
        mark("时间轴平滑")

        # --- 第四步：贴回 + 编码 ---
        # 贴回是就地修改，frames 里的数组直接被改掉，省一份内存
        def _patched_frames():
            for i in range(len(frames)):
                if cancel_check is not None:
                    cancel_check()
                yield engine.paste_patch(frames[i], fixed[i], paste_box)
                if progress_cb is not None:
                    progress_cb("write", i + 1, len(frames))

        write_bar = tqdm(_patched_frames(),
                         total=len(frames),
                         desc="贴回编码",
                         unit="帧",
                         ncols=88,
                         smoothing=0.1)

        write_video_ffmpeg(
            write_bar,
            output_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            crf=output_cfg.get("crf", 18),
            preset=output_cfg.get("preset", "medium"),
        )
        mark("贴回编码")
        return output_path

    except BaseException:
        # 处理失败时清掉半成品
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        raise
    finally:
        reader.release()


def main() -> int:
    print("=" * 60)
    print("视频固定位置水印去除")
    print("=" * 60)

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    # 水印区域
    region_cfg = config.get("region", {})
    region: Region = (
        int(region_cfg.get("x", 0)),
        int(region_cfg.get("y", 0)),
        int(region_cfg.get("w", 0)),
        int(region_cfg.get("h", 0)),
    )
    if region[2] <= 0 or region[3] <= 0:
        print("[错误] config.yaml 里的水印区域还没设置好（w/h 必须大于 0）。")
        print("       请先运行：python select_region.py")
        return 1
    print("[信息] 水印区域：x=%d, y=%d, w=%d, h=%d" % region)

    # 框选时用的参考分辨率。各视频分辨率不一致时，靠它按比例换算水印区域。
    ref_size: Tuple[int, int] = (
        int(region_cfg.get("ref_width", 0)),
        int(region_cfg.get("ref_height", 0)),
    )
    if ref_size[0] > 0 and ref_size[1] > 0:
        print("[信息] 框选时参考分辨率：%dx%d" % ref_size)
    else:
        print("[警告] config.yaml 里没有记录框选时的参考分辨率（ref_width / ref_height）。")
        print("       如果 input 里各个视频的分辨率不一样，水印区域会直接套用、可能对不准。")
        print("       建议重新运行 python select_region.py 重新框选一次。")

    inpaint_cfg = config.get("inpaint", {})
    output_cfg = config.get("output", {})

    # 修复方式：sttn（整段一起修，背景变化时更自然）/ lama（逐帧，快）/ opencv（降级）
    # 顶层 inpaint_method 优先，没写就回落到旧的 inpaint.backend，保证老配置还能用
    method = str(config.get("inpaint_method")
                 or inpaint_cfg.get("backend", "lama")).strip().lower()
    sttn_cfg = config.get("sttn", {}) or {}

    # 时间轴平滑 + 羽化相关设置
    temporal_cfg = {
        "enable": bool(config.get("temporal_enable", True)),
        "window": int(config.get("temporal_window", 3)),
    }
    feather_px = max(0, int(config.get("feather_px", 8)))

    print_env_info()
    print("[信息] 修复方式：%s" % method)
    if temporal_cfg["enable"] and temporal_cfg["window"] > 1:
        print("[信息] 时间轴平滑：开启，窗口 %d（消除修复区域的帧间闪烁）"
              % temporal_cfg["window"])
    else:
        print("[信息] 时间轴平滑：关闭")
    print("[信息] 贴回羽化：%d 像素" % feather_px)
    batch_cfg = int(inpaint_cfg.get("lama_batch_size", 0))

    # 模型只加载一次，整个批量过程复用
    try:
        engine = InpaintEngine(
            method=method,
            device=inpaint_cfg.get("device", "auto"),
            padding=int(inpaint_cfg.get("padding", 20)),
            mask_dilate=int(inpaint_cfg.get("mask_dilate", 4)),
            feather_px=feather_px,
            sttn_cfg=sttn_cfg,
            lama_batch_size=batch_cfg,
        )
    except ImportError as err:
        print("[错误] 修复后端加载失败：%s" % err)
        print("       如果是 simple-lama-inpainting 装不上，可以把 config.yaml 里的")
        print("       inpaint_method 改成 opencv，用降级方案先跑通。")
        return 1

    # 输入输出目录
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = list_videos(DEFAULT_INPUT_DIR)
    if not videos:
        print("[错误] %s 下没有 .mp4 文件。" % DEFAULT_INPUT_DIR)
        print("       把要处理的视频放进去再运行。")
        return 1

    print("[信息] 共发现 %d 个视频，输出目录：%s" % (len(videos), DEFAULT_OUTPUT_DIR))
    print()

    success = 0
    total_start = time.perf_counter()

    for index, video_path in enumerate(videos, start=1):
        print("-" * 60)
        print("[%d/%d] %s" % (index, len(videos), video_path.name))

        start = time.perf_counter()
        try:
            output_path = process_one_video(video_path, DEFAULT_OUTPUT_DIR, engine,
                                            region, ref_size, output_cfg, temporal_cfg)
        except Exception as err:
            print("[失败] %s 处理出错：%s" % (video_path.name, err))
            if "out of memory" in str(err).lower():
                print("       显存不足。可以把 config.yaml 里的 inpaint.padding 调小（例如 8），")
                print("       或者把 inpaint.device 改成 cpu 再试一次。")
            continue

        elapsed = time.perf_counter() - start
        size_mb = output_path.stat().st_size / 1024.0 / 1024.0
        print("[完成] %s -> %s，用时 %s，大小 %.1f MB"
              % (video_path.name, output_path.name, format_duration(elapsed), size_mb))
        success += 1

    total_elapsed = time.perf_counter() - total_start

    print()
    print("=" * 60)
    print("处理结束：成功 %d / %d 个，总用时 %s"
          % (success, len(videos), format_duration(total_elapsed)))
    print("输出目录：%s" % DEFAULT_OUTPUT_DIR)
    print("=" * 60)

    return 0 if success == len(videos) else 1


if __name__ == "__main__":
    sys.exit(main())
