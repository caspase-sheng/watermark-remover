# -*- coding: utf-8 -*-
"""
交互式框选水印区域，生成/更新 config.yaml。

用法：
    python select_region.py                 # 自动取 input 下第一个视频
    python select_region.py 某个视频.mp4     # 指定视频

操作：
    按住鼠标左键拖拽  —— 框选水印
    回车             —— 确认并写入 config.yaml
    R                —— 重新框选
    ESC 或 C         —— 取消，不修改 config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import yaml

from utils import (DEFAULT_CONFIG_PATH, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR,
                   VideoReader, check_resolution, imwrite_unicode, list_videos,
                   load_config)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# 窗口最大显示尺寸。画面比这个大就等比缩小显示，坐标最后会换算回原图。
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

WINDOW_NAME = "select watermark region"


class RegionSelector:
    """在 OpenCV 窗口里用鼠标拖拽选一个矩形区域。"""

    def __init__(self, frame):
        self.frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]

        # 算显示缩放比例：画面放得下就用原尺寸，放不下就等比缩小
        scale = min(1.0,
                    MAX_DISPLAY_WIDTH / float(self.frame_w),
                    MAX_DISPLAY_HEIGHT / float(self.frame_h))
        self.scale = scale

        if scale < 1.0:
            disp_w = int(round(self.frame_w * scale))
            disp_h = int(round(self.frame_h * scale))
            self.display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            self.display = frame.copy()

        self.disp_h, self.disp_w = self.display.shape[:2]

        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.selecting = False

        # 最终结果，原图坐标下的 (x, y, w, h)
        self.region: Optional[Tuple[int, int, int, int]] = None

    # -- 鼠标回调 ----------------------------------------------------------

    def on_mouse(self, event, x, y, flags, param):
        x = max(0, min(int(x), self.disp_w - 1))
        y = max(0, min(int(y), self.disp_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end = (x, y)
            self._commit_selection()

    def _commit_selection(self) -> None:
        """把显示坐标换算回原图坐标，存进 self.region。"""
        if not self.start or not self.end:
            return

        dx0, dx1 = sorted((self.start[0], self.end[0]))
        dy0, dy1 = sorted((self.start[1], self.end[1]))

        if dx1 - dx0 < 3 or dy1 - dy0 < 3:
            self.region = None
            print("[提示] 框选区域太小了，请重新拖拽。")
            return

        # 显示坐标 -> 原图坐标
        ox0 = int(round(dx0 / self.scale))
        oy0 = int(round(dy0 / self.scale))
        ox1 = int(round(dx1 / self.scale))
        oy1 = int(round(dy1 / self.scale))

        ox0 = max(0, min(ox0, self.frame_w - 1))
        oy0 = max(0, min(oy0, self.frame_h - 1))
        ox1 = max(ox0 + 1, min(ox1, self.frame_w))
        oy1 = max(oy0 + 1, min(oy1, self.frame_h))

        self.region = (ox0, oy0, ox1 - ox0, oy1 - oy0)
        x, y, w, h = self.region
        print("[信息] 当前框选：x=%d, y=%d, w=%d, h=%d" % (x, y, w, h))

    # -- 主循环 ------------------------------------------------------------

    def run(self) -> Optional[Tuple[int, int, int, int]]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            canvas = self.display.copy()

            if self.start and self.end:
                cv2.rectangle(canvas, self.start, self.end, (0, 255, 0), 2)

            if self.region:
                x, y, w, h = self.region
                info = "x=%d y=%d w=%d h=%d  |  ENTER=confirm  R=reset" % (x, y, w, h)
            else:
                info = "Drag to select watermark  |  ENTER=confirm  R=reset  ESC=cancel"

            # cv2.putText 只能写 ASCII，中文提示走控制台
            cv2.putText(canvas, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):          # 回车
                if self.region:
                    break
                print("[提示] 还没有框选区域，请先用鼠标拖拽框选水印。")
            elif key in (ord("r"), ord("R")):
                self.start = None
                self.end = None
                self.region = None
                print("[提示] 已重置框选，请重新拖拽。")
            elif key in (27, ord("c"), ord("C")):   # ESC / C 取消
                self.region = None
                break

        cv2.destroyAllWindows()
        return self.region


def grab_first_frame(video_path: Path):
    """读取视频第一帧，同时打印分辨率、帧率等信息。返回 (frame, width, height)。"""
    reader = VideoReader(video_path)
    try:
        print("[信息] 视频文件：%s" % video_path)
        count_text = reader.frame_count if reader.frame_count > 0 else "未知"
        print("[信息] 分辨率：%d x %d，帧率：%.3f fps，总帧数：%s"
              % (reader.width, reader.height, reader.fps, count_text))
        check_resolution(reader.width, reader.height)

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


def _read_header(config_path: Path) -> str:
    """取出 config.yaml 顶部那段注释，重写文件时原样保留。"""
    if not config_path.exists():
        return ""
    lines = []
    for line in config_path.read_text(encoding="utf-8").splitlines(True):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            lines.append(line)
        else:
            break
    return "".join(lines)


def save_region(region: Tuple[int, int, int, int], config_path: Path,
                ref_width: int, ref_height: int) -> None:
    """
    把框选结果写进 config.yaml，其它配置项保持不变。

    同时记下框选时视频的分辨率（ref_width / ref_height）。水印会随画面
    一起缩放，input 里视频分辨率不一致时，main.py 靠这两个值把区域等比换算过去。

    文件顶部的说明性注释会被保留，不会被覆盖掉。
    """
    config = load_config(config_path)
    x, y, w, h = region
    config["region"] = {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "ref_width": int(ref_width), "ref_height": int(ref_height),
    }

    # 优先沿用文件里已有的顶部注释；没有才用下面这段兜底
    header = _read_header(config_path) or (
        "# 去水印配置（region 由 select_region.py 生成，参数说明见 README）\n"
        "# inpaint_method: lama（默认，逐帧，快）/ sttn（视频修复，背景变化时更自然，慢）\n"
        "# region.ref_width / ref_height：框选时视频的分辨率，用于跨分辨率等比换算\n"
    )
    body = yaml.safe_dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)

    config_path.write_text(header + body, encoding="utf-8")


def main() -> int:
    print("=" * 60)
    print("水印区域框选")
    print("=" * 60)

    # 支持命令行指定视频
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        if not video_path.is_file():
            print("[错误] 找不到视频文件：%s" % video_path)
            return 1
    else:
        videos = list_videos(DEFAULT_INPUT_DIR)
        if not videos:
            print("[错误] 在 %s 下没有找到 .mp4 文件。" % DEFAULT_INPUT_DIR)
            print("       请先把要处理的视频放进 input 文件夹。")
            return 1
        video_path = videos[0]
        if len(videos) > 1:
            print("[信息] input 下共有 %d 个视频，本次用第一个来框选：%s"
                  % (len(videos), video_path.name))

    frame, ref_width, ref_height = grab_first_frame(video_path)
    if frame is None:
        print("[错误] 读不到视频第一帧，请检查文件是否完整。")
        return 1

    print()
    print("请在弹出窗口里操作：")
    print("  1. 按住鼠标左键拖拽，框住右下角的水印文字")
    print("  2. 按回车确认（按 R 重新框选，按 ESC 取消）")
    print("提示：框比水印稍大一点点即可，掩码只盖住框内，多出来的一点背景修复器能自然补上。")
    print()

    selector = RegionSelector(frame)
    if selector.scale < 1.0:
        print("[信息] 画面较大，窗口按 %.0f%% 缩放显示，坐标已自动换算回原图。"
              % (selector.scale * 100))

    region = selector.run()
    if region is None:
        print("[提示] 已取消，config.yaml 没有被修改。")
        return 1

    save_region(region, DEFAULT_CONFIG_PATH, ref_width, ref_height)

    # 顺手存一张框选预览图，方便确认范围对不对
    x, y, w, h = region
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = DEFAULT_OUTPUT_DIR / "region_preview.png"
    if imwrite_unicode(preview_path, preview):
        print("[信息] 已保存框选预览图：%s" % preview_path)

    print()
    print("[完成] 水印区域 x=%d, y=%d, w=%d, h=%d 已写入 %s"
          % (x, y, w, h, DEFAULT_CONFIG_PATH))
    print("       参考分辨率 %dx%d 也已一并记录，其它分辨率的视频会自动等比换算。"
          % (ref_width, ref_height))
    print("       下一步运行：python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
