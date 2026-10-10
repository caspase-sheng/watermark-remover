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

处理过程在一个后台线程里跑，前端用一个生成器轮询共享状态、实时刷进度；
「取消」按钮只是置一个标志位，处理循环在下一帧检测到就停下来，
已经跑完的视频会保留并照样打包。

用法：
    python gradio_app.py
然后浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import atexit
import base64
import html
import json
import os
import shutil
import sys
import tempfile
import threading
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

# Gradio 5.x 会对 Blocks(theme=..., css=..., head=...) 发 DeprecationWarning
# （说 6.0 要把这些挪到 launch()），但 5.x 的 launch() 还没有这几个参数，
# 只能先在这里压掉，不然控制台启动就是一片红。
# 以后升级到 Gradio 6.0 时，把 build_ui() 里 Blocks 的 theme / css / head
# 移到 launch() 上，再把这三行删掉。
warnings.filterwarnings("ignore",
                        message=r"The 'theme' parameter in the Blocks constructor")
warnings.filterwarnings("ignore",
                        message=r"The 'css' parameter in the Blocks constructor")
warnings.filterwarnings("ignore",
                        message=r"The 'head' parameter in the Blocks constructor")

# tqdm 默认把进度条写到 stderr，Trae / PowerShell 会把它当成错误流标红。
# Web 里进度已经由状态面板显示，这里让 main.py 里的 tqdm 改走 stdout，
# 控制台就不会再出现一堆红色的"错误"了。
_real_tqdm = _main.tqdm


def _tqdm_stdout(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    return _real_tqdm(*args, **kwargs)


_main.tqdm = _tqdm_stdout


# 发布到 GitHub 前把这里改成自己的仓库地址
GITHUB_URL = "https://github.com/caspase-sheng/watermark-remover"

# 给浏览器画布用的第一帧，长边缩到这个值以内。
# 2K/4K 的原图直接丢进浏览器又大又卡，缩过之后框选轻松很多；
# 用户框的位置会按比例换算回原分辨率，不影响精度。
MAX_FRAME_SIDE = 1280

# 引擎缓存。key 变了（换了修复方式 / 改了参数）才重新加载模型。
_ENGINE: Optional[InpaintEngine] = None
_ENGINE_KEY = None

# 前端状态面板的刷新间隔（秒）。太密会把算力浪费在拼 HTML 和传数据上，
# 0.5 秒对着一张进度条来说完全够用。
_UI_REFRESH_SEC = 0.5

# 编码速度档位。preset 只影响"压得多狠"，同一个 CRF 下画质基本一样，
# 越快出来的文件越大。
PRESET_CHOICES = ["slow", "medium", "fast"]
PRESET_DEFAULT = "medium"

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


def frame_to_data_url(frame_bgr: np.ndarray, max_side: int = MAX_FRAME_SIDE) -> str:
    """把第一帧缩到 max_side 以内、编码成 JPEG data URL，直接嵌进页面。"""
    h, w = frame_bgr.shape[:2]
    longest = float(max(w, h))
    if longest > max_side:
        scale = max_side / longest
        frame_bgr = cv2.resize(frame_bgr,
                               (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("第一帧编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# 框选画布。
#
# 最早这里用的是 gr.ImageEditor，但它在 Gradio 5.50 上前端有 bug：
# 一加载背景图就抛 `TypeError: this.history.push is not a function`，
# 顺带把 undo/redo 那个命令栈也搞废了；而且它工具栏里根本没有矩形工具。
# 所以干脆自己用 HTML + JS 画一个：一张 <img> 铺第一帧，上面盖一个绝对定位的
# <div> 当矩形框，鼠标按住拖动改它的 left/top/width/height。
# 框选结果按缩放比换算回原分辨率，塞进隐藏的 #wmr-region-json 那个 Textbox，
# 再当成普通输入传给 run_process。
#
# 注意：gr.HTML 里的 <script> 不会执行（Gradio 用的是 {@html} 插值），
# 所以交互逻辑统一挂在 Blocks(head=CUSTOM_HEAD) 上，用轮询去发现画布元素。
CANVAS_HTML = """
<div class="wmr-picker">
  <div class="wmr-stage" id="wmr-stage">
    <img id="wmr-frame" class="wmr-frame" src="%(src)s" data-ow="%(ow)d"
         data-oh="%(oh)d" data-tok="%(tok)s" alt="第一帧" draggable="false">
    <div id="wmr-rect" class="wmr-rect"></div>
  </div>
  <div class="wmr-tools">
    <button type="button" class="wmr-btn" data-wmr="undo">撤销</button>
    <button type="button" class="wmr-btn" data-wmr="redo">重做</button>
    <button type="button" class="wmr-btn" data-wmr="clear">清除</button>
    <span id="wmr-readout" class="wmr-readout">在画面上按住鼠标拖动，框住水印</span>
  </div>
</div>
"""

CANVAS_PLACEHOLDER = """
<div class="wmr-picker">
  <div class="wmr-stage wmr-stage-empty" id="wmr-stage">
    <div>上传视频后这里会显示第一帧。<br>按住鼠标拖一个矩形框住水印就行。</div>
  </div>
  <div class="wmr-tools">
    <button type="button" class="wmr-btn" data-wmr="undo">撤销</button>
    <button type="button" class="wmr-btn" data-wmr="redo">重做</button>
    <button type="button" class="wmr-btn" data-wmr="clear">清除</button>
    <span id="wmr-readout" class="wmr-readout">等待上传视频</span>
  </div>
</div>
"""


def build_canvas_html(frame_bgr: np.ndarray, orig_w: int, orig_h: int) -> str:
    """把第一帧和它的原分辨率一起拼进画布 HTML。"""
    return CANVAS_HTML % {
        "src": frame_to_data_url(frame_bgr),
        "ow": int(orig_w),
        "oh": int(orig_h),
        "tok": "%d" % int(time.time() * 1000),
    }


def parse_region(raw):
    """解析前端回传的框选 JSON，返回 (region, 参考分辨率, 错误说明)。"""
    if not raw:
        return None, None, "画布上还没有框选"
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None, None, "框选数据解析失败，请重新拖一次"
    try:
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("w", 0))
        h = int(data.get("h", 0))
    except Exception:  # noqa: BLE001
        return None, None, "框选数据不完整"
    if w <= 0 or h <= 0:
        return None, None, "框选区域太小了，请重新拖一个大一点的矩形"
    ref = (int(data.get("ow", 0) or 0), int(data.get("oh", 0) or 0))
    return (x, y, w, h), ref, None


# --------------------------------------------------------------------------
# 引擎缓存
# --------------------------------------------------------------------------

def _engine_key(method, padding, mask_dilate, feather_px, device, sttn_cfg, lama_batch):
    sttn_items = tuple(sorted((str(k), str(v)) for k, v in (sttn_cfg or {}).items()))
    return (str(method), int(padding), int(mask_dilate), int(feather_px),
            str(device), sttn_items, int(lama_batch))


def get_engine(method, padding, mask_dilate, feather_px, device, sttn_cfg,
               lama_batch=0) -> InpaintEngine:
    """按参数取引擎。参数没变就直接复用已经加载好的那个，模型不重复加载。"""
    global _ENGINE, _ENGINE_KEY
    key = _engine_key(method, padding, mask_dilate, feather_px, device, sttn_cfg, lama_batch)
    if _ENGINE is None or _ENGINE_KEY != key:
        _ENGINE = InpaintEngine(method=method, device=device, padding=padding,
                                mask_dilate=mask_dilate, feather_px=feather_px,
                                sttn_cfg=sttn_cfg, lama_batch_size=lama_batch)
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
# 处理状态：后台线程写、生成器读
# --------------------------------------------------------------------------

# 同一时间只允许一个处理任务（queue 也是串行的，这里再加一道保险）
_RUN_LOCK = threading.Lock()
# 「取消」按钮置位的标志，处理循环在下一帧检测到就停
_CANCEL = threading.Event()

# 各阶段的说明文字，状态面板和进度条共用
_PHASE_LABELS = {
    "load_model": "正在加载模型（第一次较慢）",
    "read": "读取并裁切水印区域",
    "repair": "修复中（这一阶段没有逐帧进度）",
    "smooth": "时间轴平滑",
    "write": "贴回并编码",
}


def _new_state() -> Dict:
    return {
        "total": 0,            # 共几个视频
        "index": 0,            # 当前第几个（1 起）
        "file": "",            # 当前文件名
        "width": 0, "height": 0,
        "phase": "load_model",
        "done": 0,             # 当前阶段已处理帧数
        "done_total": 0,       # 当前视频总帧数（0 表示未知）
        "video_start": 0.0,
        "started": 0.0,
        "cancelled": False,
        "results": [],         # [(文件名, 耗时秒, 大小MB, 输出路径)]
        "failures": [],        # [(文件名, 错误)]
        "error": None,         # 致命错误（模型加载失败等）
        "final": None,         # 结束后要交给界面的结果
    }


def _phase_fraction(state: Dict) -> float:
    """把当前视频内部进度折算成 0~1。
    读帧算前一半，修复阶段拿不到逐帧进度、固定停在 0.625，
    贴回编码占最后四分之一。总比不动强。"""
    done = state["done"]
    dt = state["done_total"] or 0
    phase = state["phase"]
    if phase == "read":
        return 0.5 * (done / dt if dt else 0.0)
    if phase == "repair":
        return 0.625
    if phase == "smooth":
        return 0.72
    if phase == "write":
        return 0.75 + 0.25 * (done / dt if dt else 0.0)
    return 0.0


def _status_html(state: Dict) -> str:
    """根据共享状态拼状态面板的 HTML。"""
    total = max(1, state["total"])
    frac = ((state["index"] - 1) + _phase_fraction(state)) / total if state["index"] else 0.0
    frac = max(0.0, min(1.0, frac))
    now = time.perf_counter()
    total_used = now - state["started"] if state["started"] else 0.0
    video_used = now - state["video_start"] if state["video_start"] else 0.0

    phase_label = _PHASE_LABELS.get(state["phase"], state["phase"])
    frames_txt = ""
    if state["phase"] in ("read", "write"):
        if state["done_total"]:
            frames_txt = " <b>%d/%d</b> 帧" % (state["done"], state["done_total"])
        else:
            frames_txt = " <b>%d</b> 帧" % state["done"]

    rows = []
    if state["cancelled"]:
        rows.append('<div class="wmr-warn">收到取消请求，当前这一步完成后就停……</div>')
    if state["file"]:
        rows.append(
            "<div><b>[%d/%d]</b> %s <span class=\"wmr-muted\">· %d×%d · 共 %d 帧</span></div>"
            % (state["index"], state["total"], html.escape(state["file"]),
               state["width"], state["height"], state["done_total"]))
    rows.append(
        "<div>%s%s <span class=\"wmr-muted\">· 本视频已用 %.1f s · 总计 %.1f s</span></div>"
        % (phase_label, frames_txt, video_used, total_used))
    rows.append('<div class="wmr-bar"><i style="width:%.1f%%"></i></div>' % (frac * 100.0))
    return '<div class="wmr-status">%s</div>' % "".join(rows)


_STATUS_IDLE = ('<div class="wmr-status"><div class="wmr-muted">'
                '还没开始处理。框好水印后点上面的「开始处理」。</div></div>')


def _summary_html(state: Dict, origin: str, region: Region, method: str,
                  zip_name: Optional[str], preset: str = "") -> str:
    """跑完（或被取消）后的汇总。"""
    results = state["results"]
    failures = state["failures"]
    total_used = time.perf_counter() - state["started"] if state["started"] else 0.0

    parts = ['<div class="wmr-status">']
    if state["cancelled"]:
        parts.append('<div class="wmr-warn"><b>已取消。</b>'
                     '已完成的 %d 个视频保留在下面，可以直接下载。</div>' % len(results))
    parts.append("<div>处理结束：成功 <b>%d / %d</b> 个，总用时 <b>%s</b>。</div>"
                 % (len(results), state["total"], format_duration(total_used)))
    parts.append('<div class="wmr-muted">水印区域（%s）：x=%d, y=%d, w=%d, h=%d · '
                 '修复方式 %s · 编码 preset %s · 输出保持原分辨率、H.264 CRF 18、无音频</div>'
                 % (html.escape(origin), region[0], region[1], region[2], region[3],
                    html.escape(method), html.escape(preset or "-")))
    if results:
        parts.append('<table class="wmr-table"><tr><th>视频</th><th>耗时</th>'
                     '<th>大小</th><th>阶段（读/修/平/编）</th></tr>')
        for name, used, size_mb, _path, stages in results:
            detail = "—"
            if stages:
                detail = " / ".join("%.1fs" % v for v in stages.values())
            parts.append("<tr><td>%s</td><td>%s</td><td>%.1f MB</td><td>%s</td></tr>"
                         % (html.escape(name), format_duration(used), size_mb,
                            html.escape(detail)))
        parts.append("</table>")
    if zip_name:
        parts.append('<div class="wmr-muted">打包文件：%s（结果在临时目录，'
                     '请在关闭本程序前下载）</div>' % html.escape(zip_name))
    if failures:
        parts.append('<div class="wmr-warn">失败：%s</div>'
                     % html.escape("；".join("%s（%s）" % (n, e) for n, e in failures)))
    parts.append("</div>")
    return "".join(parts)


def _note_html(text: str) -> str:
    """一条提示性质的状态。"""
    return '<div class="wmr-status"><div>%s</div></div>' % text


# --------------------------------------------------------------------------
# 后台处理线程
# --------------------------------------------------------------------------

def _batch_worker(state: Dict, paths: List[str], session: Path,
                  engine_params, region: Region, ref_size: Tuple[int, int],
                  output_cfg: Dict, temporal_cfg: Dict) -> None:
    """在后台线程里挨个视频处理，进度写进 state，生成器那边轮询读取。"""
    method, padding, mask_dilate, feather, device, sttn_cfg, lama_batch = engine_params
    try:
        state["phase"] = "load_model"
        engine = get_engine(method, padding, mask_dilate, feather, device, sttn_cfg,
                            lama_batch)
    except Exception as err:  # noqa: BLE001
        state["error"] = "模型加载失败：%s" % err
        return

    def cancel_check():
        if _CANCEL.is_set():
            raise _main.ProcessingCancelled()

    outputs: List[Path] = []
    for index, path in enumerate(paths, start=1):
        if _CANCEL.is_set():
            state["cancelled"] = True
            break
        name = Path(path).name
        state["index"] = index
        state["file"] = name
        state["phase"] = "read"
        state["done"] = 0
        state["done_total"] = 0
        state["video_start"] = time.perf_counter()

        # 先摸一下分辨率和帧数，状态面板要用
        try:
            probe = VideoReader(path)
            state["width"], state["height"] = probe.width, probe.height
            state["done_total"] = max(0, probe.frame_count)
            probe.release()
        except Exception:  # noqa: BLE001
            state["width"] = state["height"] = 0

        def progress_cb(phase, done, total):
            state["phase"] = phase
            state["done"] = done
            if total:
                state["done_total"] = total

        start = time.perf_counter()
        stages: Dict[str, float] = {}
        try:
            out = _main.process_one_video(Path(path), session, engine, region, ref_size,
                                          output_cfg, temporal_cfg,
                                          progress_cb=progress_cb,
                                          cancel_check=cancel_check,
                                          stage_times=stages)
        except _main.ProcessingCancelled:
            state["cancelled"] = True
            break
        except Exception as err:  # noqa: BLE001
            state["failures"].append((name, str(err)))
            continue

        out = Path(out)
        used = time.perf_counter() - start
        size_mb = out.stat().st_size / 1024.0 / 1024.0 if out.exists() else 0.0
        state["results"].append((name, used, size_mb, str(out),
                                 dict(stages) if stages else None))
        outputs.append(out)

    if outputs:
        zip_path = session / "watermark_removed.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in outputs:
                zf.write(item, arcname=item.name)
        state["final"] = {
            "video": str(outputs[0]),
            "files": [str(p) for p in outputs],
            "zip": str(zip_path),
            "zip_name": zip_path.name,
        }
    else:
        state["final"] = {"video": None, "files": None, "zip": None, "zip_name": None}


# --------------------------------------------------------------------------
# 事件处理
# --------------------------------------------------------------------------

def on_upload(files, progress=gr.Progress()):
    """上传后抽第一帧，铺到自建画布上供框选。

    抽第一帧要 open 一次视频，2K/4K 的文件能感觉到一点延迟，
    所以这里用 gr.Progress 给个转圈的提示，别让人以为界面卡住了。
    """
    paths = _iter_paths(files)
    if not paths:
        return (CANVAS_PLACEHOLDER, {"width": 0, "height": 0},
                _note_html("请先上传至少一个 .mp4 文件。"))

    progress(0.0, desc="正在读取第一个视频的第一帧……")
    first = paths[0]
    try:
        frame, width, height = grab_first_frame(first)
    except Exception as err:  # noqa: BLE001
        return (CANVAS_PLACEHOLDER, {"width": 0, "height": 0},
                _note_html("<b>读取第一个视频失败</b>：%s" % html.escape(str(err))))

    if frame is None:
        return (CANVAS_PLACEHOLDER, {"width": 0, "height": 0},
                _note_html("读不到第一个视频的画面，文件可能损坏。"))

    ref = {"width": int(width), "height": int(height)}
    names = "、".join("<code>%s</code>" % html.escape(Path(p).name) for p in paths)
    message = _note_html(
        "已上传 <b>%d</b> 个视频：%s。<br>"
        "第一个视频的第一帧已经铺在左边画布上（%d×%d），"
        "按住鼠标拖一个矩形框住水印，然后点「开始处理」。"
        % (len(paths), names, width, height))
    return build_canvas_html(frame, width, height), ref, message


def on_cancel():
    """点「取消」只是置标志位，后台循环在下一帧检测到就停。"""
    _CANCEL.set()


def run_process(files, region_raw, ref, use_manual, mx, my, mw, mh,
                method, mask_dilate, feather_px, temporal_enable, temporal_window,
                preset, lama_batch, progress=gr.Progress()):
    """批量处理。生成器：后台线程干活，这里每 0.5 秒刷一次状态面板。"""
    skip3 = (gr.skip(), gr.skip(), gr.skip())

    if not _RUN_LOCK.acquire(blocking=False):
        yield skip3 + (_note_html("<b>已经有一个任务在跑了</b>，"
                                  "等它结束或先点「取消」。"),)
        return

    try:
        paths = _iter_paths(files)
        if not paths:
            yield skip3 + (_note_html("<b>没有待处理的视频</b>，请先上传 mp4。"),)
            return

        config = load_config(DEFAULT_CONFIG_PATH)
        inpaint_cfg = config.get("inpaint", {}) or {}
        # 界面上选的编码速度覆盖 config.yaml 里的 output.preset
        output_cfg = dict(config.get("output", {}) or {})
        output_cfg["preset"] = str(preset or PRESET_DEFAULT)
        sttn_cfg = config.get("sttn", {}) or {}

        ref_w = int((ref or {}).get("width", 0))
        ref_h = int((ref or {}).get("height", 0))

        # 水印区域的来源，按优先级：手动坐标 > 画布框选 > config.yaml
        region = None
        ref_size: Tuple[int, int] = (ref_w, ref_h)
        origin = ""

        if use_manual and int(mw) > 0 and int(mh) > 0:
            region = (max(0, int(mx)), max(0, int(my)), int(mw), int(mh))
            origin = "来自手动坐标"

        if region is None:
            region, box_ref, _err = parse_region(region_raw)
            if region is not None:
                origin = "来自画布框选"
                if box_ref and box_ref[0] > 0 and box_ref[1] > 0:
                    ref_size = (box_ref[0], box_ref[1])

        if region is None:
            region_cfg = config.get("region", {}) or {}
            region = (int(region_cfg.get("x", 0)), int(region_cfg.get("y", 0)),
                      int(region_cfg.get("w", 0)), int(region_cfg.get("h", 0)))
            ref_size = (int(region_cfg.get("ref_width", 0)),
                        int(region_cfg.get("ref_height", 0)))
            origin = "来自 config.yaml"
            if region[2] <= 0 or region[3] <= 0:
                yield skip3 + (_note_html(
                    "<b>没拿到水印区域</b>：既没勾选手动坐标、画布上也没框，"
                    "config.yaml 里的 region 还是空的。<br>"
                    "先在画面上拖一个矩形框住水印，或者展开「手动坐标」把 x/y/w/h 填好。"),)
                return

        temporal_cfg = {"enable": bool(temporal_enable), "window": int(temporal_window)}
        feather = max(0, int(feather_px))
        padding = int(inpaint_cfg.get("padding", 20))
        device = inpaint_cfg.get("device", "auto")

        _CANCEL.clear()
        session = new_session_dir()
        state = _new_state()
        state["total"] = len(paths)
        state["started"] = time.perf_counter()

        worker = threading.Thread(
            target=_batch_worker,
            args=(state, paths, session,
                  (method, padding, int(mask_dilate), feather, device, sttn_cfg,
                   int(lama_batch)),
                  region, ref_size, output_cfg, temporal_cfg),
            daemon=True,
        )
        worker.start()

        progress(0.0, desc="开始处理")
        last_html = None
        while worker.is_alive():
            frac = ((state["index"] - 1) + _phase_fraction(state)) / max(1, state["total"]) \
                if state["index"] else 0.0
            phase_label = _PHASE_LABELS.get(state["phase"], state["phase"])
            desc = "[%d/%d] %s · %s" % (state["index"] or 1, state["total"],
                                        state["file"] or "准备中", phase_label)
            progress(max(0.0, min(1.0, frac)), desc=desc)
            current = _status_html(state)
            if current != last_html:
                yield skip3 + (current,)
                last_html = current
            worker.join(timeout=_UI_REFRESH_SEC)

        progress(1.0, desc="处理结束")

        if state["error"]:
            yield skip3 + (_note_html("<b>%s</b>" % html.escape(state["error"])),)
            return

        final = state["final"] or {}
        if not final.get("files"):
            detail = "；".join("%s（%s）" % (n, e) for n, e in state["failures"]) or "未知错误"
            if state["cancelled"] and not state["results"]:
                yield skip3 + (_note_html("<b>已取消</b>，还没有任何视频处理完成。"),)
            else:
                yield skip3 + (_note_html("<b>全部处理失败</b>：%s" % html.escape(detail)),)
            return

        yield (final["video"], final["files"], final["zip"],
               _summary_html(state, origin, region, method, final.get("zip_name"),
                             output_cfg["preset"]))
    finally:
        _RUN_LOCK.release()


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

_GITHUB_ICON = ('<svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" '
                'aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 '
                '7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94'
                '-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82'
                '.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95'
                ' 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82'
                '.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82'
                '.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95'
                '.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38'
                'A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>')

HEADER_HTML = """
<div id="wmr-header">
  <h1>视频固定位置水印去除</h1>
  <p class="wmr-sub">框住水印 → 选修复方式 → 批量处理。输出保持原分辨率、H.264 CRF 18、无音频。</p>
  <a class="wmr-gh" href="%s" target="_blank" rel="noopener">%s GitHub 仓库</a>
  <div class="wmr-divider"></div>
</div>
""" % (GITHUB_URL, _GITHUB_ICON)

DISCLAIMER_HTML = """
<div class="wmr-disclaimer">
  <span class="wmr-badge">!</span>
  <span><b>免责声明</b>：本工具仅用于处理你自己拥有版权、或已获得授权的视频。
  请勿用于去除他人作品的水印或任何侵权用途。</span>
</div>
"""

CUSTOM_CSS = """
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
}

/* ---- 头部 ---- */
#wmr-header { padding: 6px 2px 0 2px; }
#wmr-header h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0;
  background: linear-gradient(120deg, #6366f1, #8b5cf6);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  display: inline-block;
}
#wmr-header .wmr-sub { margin: .35rem 0 .55rem; color: var(--body-text-color-subdued); }
#wmr-header .wmr-gh {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #6366f1;
  text-decoration: none;
  font-size: .92rem;
  font-weight: 600;
}
#wmr-header .wmr-gh:hover { color: #8b5cf6; }
.wmr-divider {
  height: 3px;
  border-radius: 2px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6 55%, transparent);
  margin: 12px 0 14px;
}

/* ---- 免责声明横幅 ---- */
.wmr-disclaimer {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  background: linear-gradient(120deg, #fff7ed, #ffedd5);
  border: 1px solid #fdba74;
  border-radius: 12px;
  padding: 12px 16px;
  margin: 0 0 16px 0;
  line-height: 1.7;
  color: #9a3412;
}
.wmr-disclaimer .wmr-badge {
  flex: none;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #f59e0b;
  color: #fff;
  font-weight: 800;
  text-align: center;
  line-height: 22px;
  font-size: .85rem;
  margin-top: 2px;
}
.dark .wmr-disclaimer {
  background: rgba(245, 158, 11, .12);
  border-color: rgba(245, 158, 11, .45);
  color: #fcd34d;
}

/* ---- 卡片 ---- */
.wmr-card {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, .06);
  margin-bottom: 16px;
}
.dark .wmr-card { box-shadow: 0 4px 16px rgba(0, 0, 0, .4); }
.wmr-card-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 10px 0;
  font-size: 1.05rem;
  font-weight: 700;
}
.wmr-step {
  flex: none;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: #fff;
  font-size: .85rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ---- 提示条 ---- */
.wmr-hint {
  background: #eef2ff;
  border-left: 5px solid #6366f1;
  border-radius: 10px;
  padding: 10px 14px;
  line-height: 1.7;
  font-size: .93rem;
}
.wmr-hint p { margin: 0; }
.dark .wmr-hint { background: rgba(99, 102, 241, .13); }

/* ---- 按钮 ---- */
#wmr-run button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 700;
  border-radius: 12px !important;
  box-shadow: 0 6px 16px rgba(99, 102, 241, .35);
  transition: transform .15s ease, box-shadow .15s ease;
}
#wmr-run button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 22px rgba(99, 102, 241, .45);
}
#wmr-cancel button {
  border: 1.5px solid #6366f1 !important;
  color: #6366f1 !important;
  background: transparent !important;
  border-radius: 12px !important;
  font-weight: 600;
  transition: background .15s ease;
}
#wmr-cancel button:hover { background: rgba(99, 102, 241, .08) !important; }
.dark #wmr-cancel button { color: #a5b4fc !important; border-color: #a5b4fc !important; }

/* ---- 状态面板 ---- */
.wmr-status {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 12px 14px;
  background: var(--background-fill-secondary);
  font-size: .92rem;
  line-height: 1.9;
}
.wmr-status .wmr-muted { color: var(--body-text-color-subdued); }
.wmr-status .wmr-warn { color: #d97706; font-weight: 600; }
.dark .wmr-status .wmr-warn { color: #fbbf24; }
.wmr-status .wmr-bar {
  height: 8px;
  border-radius: 4px;
  background: var(--border-color-primary);
  overflow: hidden;
  margin-top: 8px;
}
.wmr-status .wmr-bar i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  border-radius: 4px;
  transition: width .25s ease;
}
.wmr-status .wmr-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
.wmr-status .wmr-table th, .wmr-status .wmr-table td {
  text-align: left;
  padding: 4px 8px 4px 0;
  font-size: .9rem;
  border-bottom: 1px dashed var(--border-color-primary);
}
.wmr-status .wmr-table th { color: var(--body-text-color-subdued); font-weight: 600; }

/* ---- 框选画布 ---- */
.wmr-picker { margin: 4px 0 10px 0; }
.wmr-stage {
  position: relative;
  display: inline-block;
  max-width: 100%;
  line-height: 0;
  border-radius: 10px;
  overflow: hidden;
  background: #0f172a;
}
.wmr-stage-empty {
  display: block;
  padding: 56px 16px;
  text-align: center;
  line-height: 1.7;
  font-size: .95rem;
  color: var(--body-text-color-subdued);
}
.wmr-frame {
  display: block;
  max-width: 100%;
  height: auto;
  cursor: crosshair;
  user-select: none;
  -webkit-user-drag: none;
}
.wmr-rect {
  position: absolute;
  display: none;
  border: 2px solid #ff3b30;
  background: rgba(255, 59, 48, .18);
  pointer-events: none;
}
.wmr-tools {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}
.wmr-btn {
  padding: 4px 14px;
  border-radius: 8px;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
  color: var(--body-text-color);
  font-size: .875rem;
  line-height: 1.6;
  cursor: pointer;
  transition: border-color .15s ease, color .15s ease;
}
.wmr-btn:hover { border-color: #6366f1; color: #6366f1; }
.wmr-readout { font-size: .875rem; color: var(--body-text-color-subdued); }
#wmr-region-json { display: none !important; }
"""


# 画布上的交互逻辑。gr.HTML 里的 <script> 不会执行，所以统一挂到 Blocks(head=...)。
# 用一段轮询去发现（或重新发现）画布元素——上传新视频后整块 HTML 会被替换掉。
CUSTOM_HEAD = """
<script>
(function () {
  var imgEl = null, rectEl = null, readoutEl = null;
  var startPt = null, curRect = null, hist = [], hi = -1, lastTok = null;

  function boxEl() {
    return document.querySelector('#wmr-region-json textarea') ||
           document.querySelector('#wmr-region-json input');
  }

  function pushRegion() {
    var tb = boxEl();
    var payload = '';
    if (curRect && curRect.w >= 2 && curRect.h >= 2 && imgEl) {
      var ow = parseFloat(imgEl.getAttribute('data-ow')) || 0;
      var oh = parseFloat(imgEl.getAttribute('data-oh')) || 0;
      // 这里按「显示尺寸」换算，不能用 naturalWidth：图片会被 CSS 的 max-width
      // 缩小显示，鼠标坐标是在显示尺寸上量的，用 naturalWidth 算会偏。
      var box = imgEl.getBoundingClientRect();
      var nw = box.width || imgEl.naturalWidth || imgEl.width || 1;
      var nh = box.height || imgEl.naturalHeight || imgEl.height || 1;
      payload = JSON.stringify({
        x: Math.max(0, Math.round(curRect.x * ow / nw)),
        y: Math.max(0, Math.round(curRect.y * oh / nh)),
        w: Math.max(1, Math.round(curRect.w * ow / nw)),
        h: Math.max(1, Math.round(curRect.h * oh / nh)),
        ow: Math.round(ow),
        oh: Math.round(oh)
      });
    }
    if (tb) {
      tb.value = payload;
      tb.dispatchEvent(new Event('input', { bubbles: true }));
    }
    if (readoutEl) {
      if (!payload) {
        readoutEl.textContent = '还没有框选，在画面上拖一个矩形';
      } else {
        var d = JSON.parse(payload);
        readoutEl.textContent = '已框选 x=' + d.x + ', y=' + d.y + ', w=' + d.w +
                                ', h=' + d.h + '（原分辨率 ' + d.ow + '×' + d.oh + '）';
      }
    }
  }

  function drawRect() {
    if (!rectEl) { return; }
    if (!curRect || curRect.w < 2 || curRect.h < 2) {
      rectEl.style.display = 'none';
      return;
    }
    rectEl.style.display = 'block';
    rectEl.style.left = curRect.x + 'px';
    rectEl.style.top = curRect.y + 'px';
    rectEl.style.width = curRect.w + 'px';
    rectEl.style.height = curRect.h + 'px';
  }

  function apply(r) {
    curRect = r ? { x: r.x, y: r.y, w: r.w, h: r.h } : null;
    drawRect();
    pushRegion();
  }

  function commit(r) {
    hist = hist.slice(0, hi + 1);
    hist.push(r ? { x: r.x, y: r.y, w: r.w, h: r.h } : null);
    hi = hist.length - 1;
    apply(r);
  }

  function restore() {
    var r = hi >= 0 ? hist[hi] : null;
    apply(r ? { x: r.x, y: r.y, w: r.w, h: r.h } : null);
  }

  function localPoint(ev) {
    var b = imgEl.getBoundingClientRect();
    return { x: ev.clientX - b.left, y: ev.clientY - b.top };
  }

  document.addEventListener('mousemove', function (ev) {
    if (!startPt || !imgEl) { return; }
    var p = localPoint(ev);
    curRect = {
      x: Math.min(startPt.x, p.x), y: Math.min(startPt.y, p.y),
      w: Math.abs(p.x - startPt.x), h: Math.abs(p.y - startPt.y)
    };
    drawRect();
  });

  document.addEventListener('mouseup', function (ev) {
    if (!startPt || !imgEl) { return; }
    var p = localPoint(ev);
    var r = {
      x: Math.min(startPt.x, p.x), y: Math.min(startPt.y, p.y),
      w: Math.abs(p.x - startPt.x), h: Math.abs(p.y - startPt.y)
    };
    startPt = null;
    if (r.w < 4 || r.h < 4) {
      // 只是点了一下、没真拖：退回上一次已确认的框，别把已有的框误清掉
      apply(hi >= 0 ? hist[hi] : null);
      return;
    }
    commit(r);
  });

  document.addEventListener('click', function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('[data-wmr]') : null;
    if (!btn) { return; }
    var act = btn.getAttribute('data-wmr');
    if (act === 'clear') {
      commit(null);
    } else if (act === 'undo') {
      if (hi >= 0) { hi -= 1; restore(); }
    } else if (act === 'redo') {
      if (hi < hist.length - 1) { hi += 1; restore(); }
    }
  });

  setInterval(function () {
    var im = document.getElementById('wmr-frame');
    if (!im) { return; }
    var src = im.getAttribute('src') || '';
    var tok = im.getAttribute('data-tok') || '';
    if (!src || tok === lastTok) { return; }
    lastTok = tok;
    imgEl = im;
    rectEl = document.getElementById('wmr-rect');
    readoutEl = document.getElementById('wmr-readout');
    hist = [];
    hi = -1;
    startPt = null;
    curRect = null;
    im.addEventListener('mousedown', function (ev) {
      if (ev.button !== 0) { return; }
      ev.preventDefault();
      startPt = localPoint(ev);
      curRect = { x: startPt.x, y: startPt.y, w: 0, h: 0 };
      drawRect();
    });
    drawRect();
    pushRegion();
  }, 250);
})();
</script>
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
    default_lama_batch = max(0, int(inpaint_cfg.get("lama_batch_size", 0)))

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet",
                           neutral_hue="slate", radius_size="lg", spacing_size="lg")

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, head=CUSTOM_HEAD,
                   title="视频固定位置水印去除") as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)

        ref_state = gr.State({})

        with gr.Row():
            # ---- 左列：上传 + 框选 ----
            with gr.Column(scale=3):
                with gr.Group(elem_classes="wmr-card"):
                    gr.HTML('<div class="wmr-card-title"><span class="wmr-step">1</span>'
                            '上传视频</div>')
                    files = gr.File(label="拖进来或点一下选择（可多选，仅 .mp4）",
                                    file_count="multiple", file_types=[".mp4"])

                with gr.Group(elem_classes="wmr-card"):
                    gr.HTML('<div class="wmr-card-title"><span class="wmr-step">2</span>'
                            '框选水印</div>')
                    gr.Markdown(
                        "在下面的画面上**按住鼠标拖一个矩形**，把水印整个框住。"
                        "框的时候比水印**大一圈**，把外面的白边/描边也一起带进去，"
                        "不然文字去掉了框还在。拖歪了用画布下面的「撤销 / 重做 / 清除」调。",
                        elem_classes="wmr-hint")
                    canvas = gr.HTML(value=CANVAS_PLACEHOLDER)
                    region_json = gr.Textbox(value="", elem_id="wmr-region-json")
                    with gr.Accordion("不想拖框？直接手填坐标", open=False):
                        use_manual = gr.Checkbox(
                            value=False, label="改用手动坐标（忽略上面的框选）",
                            info="勾上后按第一个视频的原分辨率填 x/y/w/h；"
                                 "批量处理时会自动等比换算到其它分辨率的视频")
                        with gr.Row():
                            mx = gr.Number(label="x", value=0, precision=0)
                            my = gr.Number(label="y", value=0, precision=0)
                            mw = gr.Number(label="w", value=0, precision=0)
                            mh = gr.Number(label="h", value=0, precision=0)

            # ---- 右列：参数 + 处理 + 结果 ----
            with gr.Column(scale=2):
                with gr.Group(elem_classes="wmr-card"):
                    gr.HTML('<div class="wmr-card-title"><span class="wmr-step">3</span>'
                            '修复参数</div>')
                    with gr.Accordion("展开修改（默认值和 config.yaml 一致，一般不用动）",
                                      open=False):
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
                        lama_batch = gr.Dropdown(
                            choices=[("自动（按显存）", 0), ("1（逐帧，最省显存）", 1),
                                     ("2", 2), ("4", 4), ("8", 8), ("16", 16)],
                            value=default_lama_batch,
                            label="LaMa 批大小 lama_batch_size",
                            info="一次打包几张 ROI 送进模型，只对 lama 有效。"
                                 "逐帧调用每次都要等 GPU 同步，实测 243 帧要十几秒；"
                                 "打包成 8 张只要 2 秒多。填 1 回到逐帧老路子。"
                                 "显存不足会自动减半重试。")
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
                        preset = gr.Dropdown(
                            choices=PRESET_CHOICES, value=PRESET_DEFAULT,
                            label="编码速度 preset",
                            info="开了 LaMa 批量推理之后，编码成了最大的一块（约占四成）。"
                                 "slow 最省体积但最慢，medium 快一倍多，fast 更快、"
                                 "文件稍大。CRF 固定 18，画质基本不受影响。")

                with gr.Group(elem_classes="wmr-card"):
                    gr.HTML('<div class="wmr-card-title"><span class="wmr-step">4</span>'
                            '开始处理</div>')
                    with gr.Row():
                        run_btn = gr.Button("开始处理", variant="primary", size="lg",
                                            scale=3, elem_id="wmr-run")
                        cancel_btn = gr.Button("取消", size="lg", scale=1,
                                               elem_id="wmr-cancel")
                    status = gr.HTML(value=_STATUS_IDLE)

                with gr.Group(elem_classes="wmr-card"):
                    gr.HTML('<div class="wmr-card-title">结果</div>')
                    result_video = gr.Video(label="预览（第一个处理完的视频）")
                    result_files = gr.File(label="全部结果（可单个下载）",
                                           file_count="multiple")
                    result_zip = gr.File(label="打包下载 zip")
                    gr.Markdown(
                        "结果写在系统临时目录里，**关闭本程序前记得下载**。",
                        elem_classes="wmr-hint")

        files.change(on_upload,
                     inputs=[files],
                     outputs=[canvas, ref_state, status])
        run_btn.click(run_process,
                      inputs=[files, region_json, ref_state, use_manual, mx, my, mw, mh,
                              method, mask_dilate, feather_px, temporal_enable,
                              temporal_window, preset, lama_batch],
                      outputs=[result_video, result_files, result_zip, status])
        # 取消走单独的并发通道，不会被正在跑的处理任务堵在队列里
        cancel_btn.click(on_cancel, inputs=None, outputs=None,
                         concurrency_id="cancel", concurrency_limit=4)

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
