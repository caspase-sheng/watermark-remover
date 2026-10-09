# -*- coding: utf-8 -*-
"""
Web 界面入口（Gradio）。

这个文件只干"界面 + 调度"两件事，水印修复本身没有重写：

    裁 ROI / 调后端 / 贴回   ——  inpaint.InpaintEngine
    单个视频的完整处理流程   ——  main.process_one_video
    视频读写 / 区域换算 / 编码 ——  utils
    时间轴平滑               ——  temporal

参数面板上的几个开关会覆盖 config.yaml 里同名的项；其余（device、
padding、sttn 的分段参数、编码参数）还是从 config.yaml 读。

模型只会加载一次。只要参数没变，后面每个视频都复用同一个引擎实例，
显存里的权重一直留着，不用反复加载。

用法：
    python gradio_app.py
然后浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

import main as _main
from inpaint import InpaintEngine
from utils import DEFAULT_CONFIG_PATH, VideoReader, format_duration, load_config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Gradio 5.x 会对 Blocks(theme=..., css=...) 发两条 DeprecationWarning（说 6.0 要把
# theme / css 挪到 launch()），但 5.x 的 launch() 还没有这两个参数，只能先在这里压掉，
# 不然控制台启动就是一片红。以后升级到 Gradio 6.0 时，把 build_ui() 里 Blocks 的
# theme / css 移到 launch() 上，再把这两行删掉。
warnings.filterwarnings("ignore",
                        message=r"The 'theme' parameter in the Blocks constructor")
warnings.filterwarnings("ignore",
                        message=r"The 'css' parameter in the Blocks constructor")

# tqdm 默认把进度条写到 stderr，Trae / PowerShell 会把它当成错误流标红。
# Web 里进度已经由 gr.Progress 显示，这里让 main.py 里的 tqdm 改走 stdout，
# 控制台就不会再出现一堆红色的"错误"了。
_real_tqdm = _main.tqdm


def _tqdm_stdout(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    return _real_tqdm(*args, **kwargs)


_main.tqdm = _tqdm_stdout


# 发布到 GitHub 前把这里改成自己的仓库地址
GITHUB_URL = "https://github.com/your-name/watermark-remover"

# 送进 gr.ImageEditor 之前先把第一帧缩小到这个边长以内。
# 2K/4K 的原图直接丢进浏览器又大又卡，缩过之后框选轻松很多；
# 用户涂的位置最后会按比例换算回原分辨率，不影响精度。
MAX_EDITOR_SIDE = 1280

# 引擎缓存。key 变了（换了修复方式 / 改了参数）才重新加载模型。
_ENGINE: Optional[InpaintEngine] = None
_ENGINE_KEY = None

# 每次运行的结果都写在系统临时目录下，下次运行或退出时清掉。
# 放在固定目录里是为了加到 launch 的 allowed_paths，Gradio 才肯把
# 这里的文件发给浏览器下载。
SESSION_ROOT = Path(tempfile.gettempdir()) / "wmr_web"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
_SESSION: Dict[str, Optional[str]] = {"dir": None}

Region = Tuple[int, int, int, int]


# --------------------------------------------------------------------------
# 读取 / 换算
# --------------------------------------------------------------------------

def _iter_paths(files) -> List[str]:
    """把 gr.File 给出的值统一成路径字符串列表。"""
    paths: List[str] = []
    for item in files or []:
        path = getattr(item, "name", None) or getattr(item, "path", None) or item
        if path:
            paths.append(str(path))
    return paths


def grab_first_frame(video_path: str):
    """取视频第一帧（BGR），顺便拿到原分辨率。读不到就返回 (None, 0, 0)。"""
    reader = VideoReader(video_path)
    try:
        frames = reader.frames()
        try:
            frame = next(frames)
        except StopIteration:
            frame = None
        finally:
            frames.close()
        return frame, reader.width, reader.height
    finally:
        reader.release()


def fit_for_editor(frame_bgr: np.ndarray) -> np.ndarray:
    """把第一帧缩到 MAX_EDITOR_SIDE 以内，再转成 RGB 给 ImageEditor。"""
    h, w = frame_bgr.shape[:2]
    longest = float(max(w, h))
    if longest > MAX_EDITOR_SIDE:
        scale = MAX_EDITOR_SIDE / longest
        frame_bgr = cv2.resize(frame_bgr,
                               (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def extract_region(editor_value, frame_w: int, frame_h: int):
    """
    从 gr.ImageEditor 的结果里取出用户涂画/框选的包围盒。

    ImageEditor 把用户的笔迹存在 layers 里（RGBA），没画的地方 alpha=0。
    所以把所有 layer 的 alpha 并起来，找非零像素的外接矩形，就是水印区域。

    画布尺寸可能和原帧不一样（我们显示前缩过），这里按比例换算回原帧坐标。

    返回 (x, y, w, h) 或者 (None, 错误说明)。
    """
    if not isinstance(editor_value, dict):
        return None, "还没有框选"

    layers = editor_value.get("layers") or []
    painted = None
    for layer in layers:
        if layer is None:
            continue
        arr = np.asarray(layer)
        if arr.ndim != 3:
            continue
        if arr.shape[2] == 4:
            mark = arr[:, :, 3] > 0
        else:
            # 万一给的是不带 alpha 的图层，就把非黑像素当笔迹
            mark = arr[:, :, :3].max(axis=2) > 0
        painted = mark if painted is None else (painted | mark)

    if painted is None or not painted.any():
        return None, "还没有在画面上涂住水印"

    ys, xs = np.where(painted)
    canvas_h, canvas_w = painted.shape[:2]
    if canvas_w <= 0 or canvas_h <= 0:
        return None, "框选画布尺寸异常"

    sx = float(frame_w) / float(canvas_w)
    sy = float(frame_h) / float(canvas_h)

    x = int(round(xs.min() * sx))
    y = int(round(ys.min() * sy))
    w = int(round((xs.max() - xs.min() + 1) * sx))
    h = int(round((ys.max() - ys.min() + 1) * sy))

    # 夹回画面内，防止越界
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h), None


# --------------------------------------------------------------------------
# 引擎缓存
# --------------------------------------------------------------------------

def _engine_key(method, padding, mask_dilate, feather_px, device, sttn_cfg):
    sttn_items = tuple(sorted((str(k), str(v)) for k, v in (sttn_cfg or {}).items()))
    return (str(method), int(padding), int(mask_dilate), int(feather_px),
            str(device), sttn_items)


def get_engine(method, padding, mask_dilate, feather_px, device, sttn_cfg) -> InpaintEngine:
    """按参数取引擎。参数没变就直接复用已经加载好的那个，模型不重复加载。"""
    global _ENGINE, _ENGINE_KEY
    key = _engine_key(method, padding, mask_dilate, feather_px, device, sttn_cfg)
    if _ENGINE is None or _ENGINE_KEY != key:
        _ENGINE = InpaintEngine(method=method, device=device, padding=padding,
                                mask_dilate=mask_dilate, feather_px=feather_px,
                                sttn_cfg=sttn_cfg)
        _ENGINE_KEY = key
    return _ENGINE


# --------------------------------------------------------------------------
# 临时目录
# --------------------------------------------------------------------------

def new_session_dir() -> Path:
    """新建一次运行的临时目录，顺手把上一次的清掉。"""
    old = _SESSION.get("dir")
    if old:
        shutil.rmtree(old, ignore_errors=True)
    session = Path(tempfile.mkdtemp(prefix="run_", dir=str(SESSION_ROOT)))
    _SESSION["dir"] = str(session)
    return session


@atexit.register
def _cleanup_session() -> None:
    """退出时清掉这次运行的中间文件。"""
    old = _SESSION.get("dir")
    if old:
        shutil.rmtree(old, ignore_errors=True)


# --------------------------------------------------------------------------
# 事件处理
# --------------------------------------------------------------------------

def on_upload(files):
    """上传后自动抽第一帧，放进 ImageEditor 供框选。"""
    paths = _iter_paths(files)
    if not paths:
        return None, [], {"width": 0, "height": 0}, "请先上传至少一个 .mp4 文件。"

    first = paths[0]
    try:
        frame, width, height = grab_first_frame(first)
    except Exception as err:
        return None, paths, {"width": 0, "height": 0}, "**读取第一个视频失败**：%s" % err

    if frame is None:
        return None, paths, {"width": 0, "height": 0}, "读不到第一个视频的画面，文件可能损坏。"

    preview = fit_for_editor(frame)
    ref = {"width": int(width), "height": int(height)}
    message = ("已上传 **%d** 个视频。\n\n"
               "第一个：`%s`（%d×%d）。请在右边的画面上涂住水印，"
               "涂完这里会显示换算好的坐标。"
               % (len(paths), Path(first).name, width, height))
    return preview, paths, ref, message


def on_editor_change(editor_value, ref):
    """用户在画面上涂画后，实时显示换算出来的水印区域。"""
    width = int((ref or {}).get("width", 0))
    height = int((ref or {}).get("height", 0))
    if width <= 0 or height <= 0:
        return "请先上传视频，再在画面上框选水印。"

    region, err = extract_region(editor_value, width, height)
    if region is None:
        return "水印区域：%s" % err

    x, y, w, h = region
    return ("水印区域：**x=%d, y=%d, w=%d, h=%d**（换算到原分辨率 %d×%d）"
            % (x, y, w, h, width, height))


def run_process(files, editor_value, ref, method, mask_dilate, feather_px,
                temporal_enable, temporal_window, progress=gr.Progress()):
    """批量处理：挨个视频走 main.process_one_video，最后打包成一个 zip。"""
    paths = _iter_paths(files)
    if not paths:
        return None, None, None, "**没有待处理的视频**，请先在左上角上传 mp4。"

    config = load_config(DEFAULT_CONFIG_PATH)
    inpaint_cfg = config.get("inpaint", {}) or {}
    output_cfg = config.get("output", {}) or {}
    sttn_cfg = config.get("sttn", {}) or {}

    ref_w = int((ref or {}).get("width", 0))
    ref_h = int((ref or {}).get("height", 0))

    # 优先用界面上框选出来的区域；没框选就退回 config.yaml 里的 region
    region = None
    ref_size: Tuple[int, int] = (ref_w, ref_h)
    origin = "来自界面框选"
    if ref_w > 0 and ref_h > 0:
        region, _ = extract_region(editor_value, ref_w, ref_h)

    if region is None:
        region_cfg = config.get("region", {}) or {}
        region = (int(region_cfg.get("x", 0)), int(region_cfg.get("y", 0)),
                  int(region_cfg.get("w", 0)), int(region_cfg.get("h", 0)))
        ref_size = (int(region_cfg.get("ref_width", 0)), int(region_cfg.get("ref_height", 0)))
        origin = "来自 config.yaml"
        if region[2] <= 0 or region[3] <= 0:
            return (None, None, None,
                    "**没拿到水印区域**：既没在画面上框选，config.yaml 里的 region 也是空的。\n\n"
                    "请在上传后的画面上涂住水印再来。")

    temporal_cfg = {"enable": bool(temporal_enable), "window": int(temporal_window)}
    feather = max(0, int(feather_px))
    padding = int(inpaint_cfg.get("padding", 20))
    device = inpaint_cfg.get("device", "auto")

    try:
        engine = get_engine(method, padding, int(mask_dilate), feather, device, sttn_cfg)
    except Exception as err:
        return None, None, None, "**模型加载失败**：%s" % err

    session = new_session_dir()
    outputs: List[Path] = []
    failures: List[Tuple[str, str]] = []
    started = time.perf_counter()
    total = len(paths)

    for index, path in enumerate(paths, start=1):
        name = Path(path).name
        progress((index - 1) / float(total), desc="[%d/%d] 正在处理 %s" % (index, total, name))
        try:
            out = _main.process_one_video(Path(path), session, engine, region, ref_size,
                                          output_cfg, temporal_cfg)
            outputs.append(Path(out))
        except Exception as err:  # noqa: BLE001
            failures.append((name, str(err)))
        progress(index / float(total), desc="[%d/%d] %s 完成" % (index, total, name))

    if not outputs:
        detail = "；".join("%s（%s）" % (n, e) for n, e in failures) or "未知错误"
        return None, None, None, "**全部处理失败**。%s" % detail

    # 打包下载
    zip_path = session / "watermark_removed.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in outputs:
            zf.write(item, arcname=item.name)

    elapsed = time.perf_counter() - started
    lines = [
        "处理完成：成功 **%d / %d** 个，用时 **%s**。"
        % (len(outputs), total, format_duration(elapsed)),
        "水印区域（%s）：x=%d, y=%d, w=%d, h=%d" % (origin, region[0], region[1], region[2], region[3]),
        "修复方式：`%s`。输出保持原分辨率，H.264 CRF 18，无音频。" % method,
    ]
    if failures:
        lines.append("失败：%s" % "；".join("%s（%s）" % (n, e) for n, e in failures))

    return str(outputs[0]), [str(p) for p in outputs], str(zip_path), "\n\n".join(lines)


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

HEADER_HTML = """
<div id="wmr-header">
  <h1>视频固定位置水印去除</h1>
  <p>框住水印 → 选修复方式 → 批量处理。输出保持原分辨率、H.264 CRF 18、无音频。</p>
  <p><a href="%s" target="_blank">GitHub 仓库</a></p>
</div>
""" % GITHUB_URL

DISCLAIMER_HTML = """
<div class="wmr-disclaimer">
  <b>免责声明</b>：本工具仅用于处理你自己拥有版权、或已获得授权的视频。
  请勿用于去除他人作品的水印或任何侵权用途。
</div>
"""

CUSTOM_CSS = """
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
#wmr-header h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 .3rem 0; }
#wmr-header p { margin: .15rem 0; color: var(--body-text-color-subdued); }
.wmr-disclaimer {
  background: #fff4e5;
  border-left: 6px solid #ff9800;
  border-radius: 10px;
  padding: 12px 16px;
  margin: 6px 0 12px 0;
  line-height: 1.7;
}
.dark .wmr-disclaimer { background: rgba(255, 152, 0, .14); }
"""


def build_ui() -> gr.Blocks:
    """搭界面。默认值都从 config.yaml 读，保证和命令行版本一致。"""
    config = load_config(DEFAULT_CONFIG_PATH)
    inpaint_cfg = config.get("inpaint", {}) or {}

    default_method = str(config.get("inpaint_method", "lama")).strip().lower()
    if default_method not in ("lama", "sttn"):
        default_method = "lama"
    default_mask_dilate = int(inpaint_cfg.get("mask_dilate", 4))
    default_feather = int(config.get("feather_px", 8))
    default_temporal = bool(config.get("temporal_enable", True))
    default_window = int(config.get("temporal_window", 3))

    theme = gr.themes.Soft(primary_hue="sky", secondary_hue="violet",
                           neutral_hue="slate", radius_size="lg", spacing_size="lg")

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="视频固定位置水印去除") as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)

        files_state = gr.State([])
        ref_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=3):
                files = gr.File(label="① 上传视频（可多选，.mp4）",
                                file_count="multiple", file_types=[".mp4"])
                editor = gr.ImageEditor(
                    label="② 在画面上涂住 / 框住水印",
                    type="numpy",
                    sources=("upload", "clipboard"),
                    show_download_button=False,
                )
                region_box = gr.Textbox(label="识别到的水印区域", interactive=False,
                                        value="请先上传视频，再在画面上框选水印。")

            with gr.Column(scale=2):
                with gr.Accordion("③ 修复参数", open=True):
                    method = gr.Dropdown(
                        choices=["lama", "sttn"], value=default_method,
                        label="修复方式 inpaint_method",
                        info="lama：逐帧修，快，动画/插画风够用；"
                             "sttn：整段一起修，实拍、背景在动时更自然，慢一些")
                    mask_dilate = gr.Slider(
                        0, 20, value=default_mask_dilate, step=1,
                        label="掩码膨胀 mask_dilate",
                        info="把水印边缘抗锯齿一起盖住的像素数，默认 4；"
                             "水印外圈有白边残留就调大")
                    feather_px = gr.Slider(
                        0, 32, value=default_feather, step=1,
                        label="边缘羽化 feather_px",
                        info="贴回时边缘过渡的像素数，默认 8；0 表示不羽化")
                    temporal_enable = gr.Checkbox(
                        value=default_temporal,
                        label="时间轴平滑 temporal_enable",
                        info="沿时间轴做中值滤波，压掉修复区域的帧间闪烁")
                    temporal_window = gr.Slider(
                        1, 9, value=default_window, step=2,
                        label="平滑窗口 temporal_window",
                        info="只取奇数帧，3 或 5 够用；1 等于不平滑")
                run_btn = gr.Button("④ 开始处理", variant="primary", size="lg")

        status = gr.Markdown("等待上传视频。")

        with gr.Row():
            result_video = gr.Video(label="结果预览（第一个视频）")
            with gr.Column():
                result_files = gr.File(label="全部结果（可单个下载）", file_count="multiple")
                result_zip = gr.File(label="打包下载 zip")

        files.change(on_upload,
                     inputs=[files],
                     outputs=[editor, files_state, ref_state, region_box])
        editor.change(on_editor_change,
                      inputs=[editor, ref_state],
                      outputs=[region_box])
        run_btn.click(run_process,
                      inputs=[files, editor, ref_state, method, mask_dilate,
                              feather_px, temporal_enable, temporal_window],
                      outputs=[result_video, result_files, result_zip, status])

    return demo


def main() -> int:
    demo = build_ui()
    # 串行处理：同一时间只跑一个任务，避免两个视频抢显存
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(SESSION_ROOT)],
        inbrowser=os.environ.get("WMR_NO_BROWSER") != "1",
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
