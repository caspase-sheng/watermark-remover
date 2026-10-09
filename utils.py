# -*- coding: utf-8 -*-
"""
通用工具模块。

包含四类功能：
1. 视频读取（VideoReader，兼容中文路径 / 中文文件名）
2. 分辨率检查（check_resolution）
3. 用 imageio-ffmpeg 自带的 FFmpeg 写 H.264 无声视频（write_video_ffmpeg）
4. 读取 config.yaml（load_config）

注意：本模块不依赖系统安装 FFmpeg，imageio-ffmpeg 的轮子里自带 ffmpeg 可执行文件。
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
import yaml

# Windows 控制台默认是 GBK，这里尽量切成 UTF-8，否则中文提示会乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 本项目只处理 mp4
VIDEO_EXTS = (".mp4",)

# 1080p 门槛
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# 路径都基于脚本所在目录，避免因为工作目录不同找不到 input/output
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# config.yaml 的默认值，缺字段时自动补全
DEFAULT_CONFIG: Dict = {
    # ref_width / ref_height 记录框选时用的视频分辨率，跨分辨率时按比例换算
    "region": {"x": 0, "y": 0, "w": 0, "h": 0, "ref_width": 0, "ref_height": 0},
    "inpaint": {
        "backend": "lama",   # 旧字段，保留做兼容；新配置用顶层 inpaint_method
        "device": "auto",    # auto / cuda / cpu
        "padding": 20,       # ROI 在水印框基础上向外扩的像素数
        "mask_dilate": 4,    # 掩码膨胀像素，用来盖住水印边缘的抗锯齿
    },
    # 修复方式：sttn（整段一起修，背景变化时更自然，慢）/ lama（逐帧，快）/ opencv（降级）
    "inpaint_method": "lama",
    # STTN 专用参数，inpaint_method 为 sttn 时才读
    "sttn": {
        "segment_size": 10,        # 一次送进网络的帧数，显存不够会自动对半调小
        "neighbor_stride": 5,      # 相邻窗口步长
        "reference_length": 10,    # 每个窗口额外参考的帧数
        "min_segment_size": 2,     # 自动降 segment_size 的下限
        "context_scale": 2.0,      # 参考带子相对水印框的放大倍数
        "model_path": "",          # 留空则用 models/sttn-det.safetensors
    },
    "output": {
        "crf": 18,
        "preset": "slow",
    },
    # 逐帧修复的结果在相邻帧之间会有轻微跳动，背景一变就看着像闪烁。
    # 打开时间轴平滑后，会把每帧的 ROI 沿时间轴做一次中值滤波压掉波动。
    "temporal_enable": True,
    "temporal_window": 3,   # 时间窗口（帧），必须 >= 3 才有意义，取奇数
    "feather_px": 8,        # ROI 贴回时边缘羽化的像素数，0 表示不羽化
}


# --------------------------------------------------------------------------
# 路径相关
# --------------------------------------------------------------------------

def is_ascii(text) -> bool:
    """判断字符串是否全是 ASCII 字符（中文路径会返回 False）。"""
    try:
        str(text).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def list_videos(folder) -> List[Path]:
    """列出目录下所有 mp4 文件，按文件名排序（.MP4 也能识别）。"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


# --------------------------------------------------------------------------
# 单张图片读写（中文路径安全）
# --------------------------------------------------------------------------

def imread_unicode(path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    读图片，支持中文路径。

    cv2.imread 在 Windows 上遇到中文路径会直接返回 None，
    所以改成先用 numpy 把文件读成字节流，再用 cv2.imdecode 解码。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image: np.ndarray, ext: str = ".png") -> bool:
    """
    写图片，支持中文路径。

    先用 cv2.imencode 编码到内存，再用 numpy 的 tofile 落盘，
    绕开 cv2.imwrite 在中文路径下静默失败的问题。
    """
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# 视频读取
# --------------------------------------------------------------------------

def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """统一成 3 通道 BGR，避免灰度/带 alpha 的视频把后面的逻辑搞崩。"""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


class VideoReader:
    """
    视频帧读取器，兼容中文路径。

    策略：先用原路径直接打开；如果失败且路径含中文，
    就把文件复制到一个纯 ASCII 的临时目录再打开，
    读完以后（release 时）把临时目录删掉。
    """

    def __init__(self, path):
        self.path = str(path)
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self._temp_path: Optional[str] = None
        self._open()

    # -- 内部方法 ----------------------------------------------------------

    @staticmethod
    def _try_open(path: str) -> Optional[cv2.VideoCapture]:
        """尝试打开视频，成功且能读到分辨率才返回 VideoCapture。"""
        try:
            cap = cv2.VideoCapture(path)
        except Exception:
            return None
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return cap
        cap.release()
        return None

    @staticmethod
    def _copy_to_temp(src: str) -> str:
        """把视频复制到 ASCII 临时目录，返回临时文件路径。"""
        tmp_dir = tempfile.mkdtemp(prefix="wmr_in_")
        suffix = Path(src).suffix or ".mp4"
        tmp_path = os.path.join(tmp_dir, "video_%s%s" % (uuid.uuid4().hex[:8], suffix))
        shutil.copy2(src, tmp_path)
        return tmp_path

    def _open(self) -> None:
        cap = self._try_open(self.path)

        if cap is None and not is_ascii(self.path):
            # 中文路径兜底：复制一份 ASCII 路径的副本再试
            try:
                self._temp_path = self._copy_to_temp(self.path)
                cap = self._try_open(self._temp_path)
            except Exception:
                self._temp_path = None

        if cap is None:
            self.release()
            raise RuntimeError(
                "无法打开视频：%s（路径不存在、编码不支持，或者 OpenCV 打不开该文件）" % self.path
            )

        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        # 有些视频读出来的 fps 是 0 或者 NaN，兜底成 25
        self.fps = float(fps) if (fps and 0 < fps < 1000) else 25.0

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = count if count > 0 else 0

    # -- 对外方法 ----------------------------------------------------------

    def frames(self) -> Iterator[np.ndarray]:
        """
        逐帧迭代。读到头或者读失败就结束，并在结束时释放资源。

        这里不用 CAP_PROP_FRAME_COUNT 来限制循环次数，因为容器里记录的帧数
        经常不准，按 count 截断会丢帧。
        """
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    break
                yield _normalize_frame(frame)
        finally:
            self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.frames()

    def release(self) -> None:
        """释放 VideoCapture，并清理临时副本。可以重复调用。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._temp_path:
            tmp_dir = os.path.dirname(self._temp_path)
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            self._temp_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def read_video(path):
    """
    打开视频并返回 (帧迭代器, fps, 宽, 高)。

    迭代器内部持有 VideoReader 的引用，读完会自动释放；
    中途 break 的话记得调用迭代器的 close()。
    """
    reader = VideoReader(path)
    return reader.frames(), reader.fps, reader.width, reader.height


# --------------------------------------------------------------------------
# 分辨率检查
# --------------------------------------------------------------------------

def check_resolution(width: int, height: int) -> bool:
    """
    检查视频分辨率。

    低于 1920x1080：打印警告（说明无法"提升"到 1080p，仍按原分辨率处理），返回 False。
    等于或高于 1920x1080：保持原分辨率输出，返回 True。
    """
    if width < TARGET_WIDTH or height < TARGET_HEIGHT:
        print("[警告] 原视频分辨率 %dx%d 低于 %dx%d。" % (width, height, TARGET_WIDTH, TARGET_HEIGHT))
        print("       本工具只做水印修复，不做超分辨率放大，无法把画面提升到 1080p；")
        print("       仍然按原分辨率 %dx%d 输出。" % (width, height))
        return False

    print("[信息] 原视频分辨率 %dx%d，将保持原分辨率输出。" % (width, height))
    return True


# --------------------------------------------------------------------------
# 水印区域换算
# --------------------------------------------------------------------------

def scale_region(region, src_width: int, src_height: int,
                 dst_width: int, dst_height: int):
    """
    把一个分辨率上框好的水印区域，等比换算到另一个分辨率。

    背景：框选是在某个视频（比如 720p）的画面坐标上做的，存的是绝对像素。
    但水印是跟着画面一起缩放的，直接把这组坐标用到其它分辨率的视频上就会偏。
    所以换成目标分辨率后要按 width/height 的比例重新算一遍。

    src_width/src_height 是框选时的参考分辨率，为 0 或者和目标一致时原样返回。
    换算结果会被夹在画面范围内，保证不会越界。
    """
    x, y, w, h = (int(v) for v in region)

    if src_width <= 0 or src_height <= 0:
        return (x, y, w, h)
    if src_width == dst_width and src_height == dst_height:
        return (x, y, w, h)
    if dst_width <= 0 or dst_height <= 0:
        return (x, y, w, h)

    rx = dst_width / float(src_width)
    ry = dst_height / float(src_height)

    nx = int(round(x * rx))
    ny = int(round(y * ry))
    nw = max(1, int(round(w * rx)))
    nh = max(1, int(round(h * ry)))

    # 夹回画面内
    nx = max(0, min(nx, dst_width - 1))
    ny = max(0, min(ny, dst_height - 1))
    nw = max(1, min(nw, dst_width - nx))
    nh = max(1, min(nh, dst_height - ny))

    return (nx, ny, nw, nh)


# --------------------------------------------------------------------------
# 配置读取
# --------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """把 override 合并进 base（只覆盖出现的字段，保留默认值）。"""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None) -> Dict:
    """
    读取 config.yaml，缺失的字段用默认值补全。

    文件不存在时直接返回默认配置（默认 region 是全 0，main.py 会判断为未设置）。
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = copy.deepcopy(DEFAULT_CONFIG)

    if not config_path.exists():
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("config.yaml 格式不对，顶层应该是一个字典（key: value）。")

    return _deep_merge(config, data)


# --------------------------------------------------------------------------
# 视频写出
# --------------------------------------------------------------------------

def _encode_with_ffmpeg(frames, target_path: str, width: int, height: int,
                        fps: float, crf: int, preset: str,
                        log_level: str = "error") -> str:
    """
    调用 imageio-ffmpeg 自带的 ffmpeg，把 BGR 原始帧流编码成 H.264 无声 mp4。

    之所以直接拼命令行而不用 imageio_ffmpeg.write_frames，是为了精确控制
    -crf / -preset / -pix_fmt / -an 这几个参数，避免不同版本默认值不一样。

    frames 里每个元素都必须是 shape=(height, width, 3)、dtype=uint8 的 BGR 数组。
    """
    import imageio_ffmpeg  # 延迟导入，没装也不影响本模块其它函数

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel", log_level,
        # 输入：stdin 上的原始 BGR 像素流
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-video_size", "%dx%d" % (width, height),
        "-framerate", "%.6f" % fps,
        "-i", "pipe:0",
        # 输出：H.264，无声
        "-an",
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-r", "%.6f" % fps,
        "-movflags", "+faststart",
        str(target_path),
    ]

    err_file = tempfile.TemporaryFile()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
        )

        for frame in frames:
            if frame is None:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                raise RuntimeError(
                    "帧尺寸 %dx%d 与输出尺寸 %dx%d 不一致"
                    % (frame.shape[1], frame.shape[0], width, height)
                )
            try:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                # ffmpeg 提前退出，跳出循环，下面统一读取返回码报错
                break

        proc.stdin.close()
        proc.stdin = None
        return_code = proc.wait()

        err_file.seek(0)
        err_text = err_file.read().decode("utf-8", errors="replace").strip()

        if return_code != 0:
            raise RuntimeError(
                "FFmpeg 编码失败（退出码 %d）：%s" % (return_code, err_text[:800] or "无输出信息")
            )
        return err_text
    finally:
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        err_file.close()


def write_video_ffmpeg(frames, output_path, fps, width, height,
                       crf: int = 18, preset: str = "slow") -> Path:
    """
    把帧序列写成 H.264 无声视频（libx264 / CRF 18 / preset slow / yuv420p / 无音频）。

    参数：
        frames      : 可迭代对象，逐帧产出 BGR 的 numpy 数组
        output_path : 输出文件路径
        fps         : 帧率
        width/height: 输出分辨率（严格等于原视频分辨率，不缩放不裁剪）
        crf/preset  : libx264 参数

    输出路径含中文时，先编码到 ASCII 临时文件再移动过去，
    避开个别环境下 ffmpeg 命令行参数编码的问题。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = float(fps) if (fps and fps > 0) else 25.0
    width, height = int(width), int(height)
    crf = int(crf)
    preset = str(preset)

    if width <= 0 or height <= 0:
        raise ValueError("输出分辨率非法：%dx%d" % (width, height))
    if width % 2 != 0 or height % 2 != 0:
        print("[警告] 输出分辨率 %dx%d 含奇数边，libx264 的 yuv420p 要求宽高为偶数，"
              "编码可能报错。请确认源视频分辨率。" % (width, height))

    if is_ascii(str(output_path)):
        _encode_with_ffmpeg(frames, str(output_path), width, height, fps, crf, preset)
        return output_path

    # 中文输出路径：先写临时文件再移动
    tmp_dir = tempfile.mkdtemp(prefix="wmr_out_")
    tmp_file = os.path.join(tmp_dir, "video.mp4")
    try:
        _encode_with_ffmpeg(frames, tmp_file, width, height, fps, crf, preset)
        if output_path.exists():
            output_path.unlink()
        shutil.move(tmp_file, str(output_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_path


def format_duration(seconds: float) -> str:
    """把秒数格式化成 xx 分 xx 秒，日志里好读一点。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "%.1f 秒" % seconds
    minutes, sec = divmod(seconds, 60)
    return "%d 分 %.1f 秒" % (int(minutes), sec)
