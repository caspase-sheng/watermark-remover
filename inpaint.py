# -*- coding: utf-8 -*-
"""
水印修复核心逻辑。

支持两种修复方式，用 config.yaml 里的 inpaint_method 切换：

    lama  —— 逐帧把水印周围的一小块图（ROI）丢给 LaMa，一张一张修。
             快，但 LaMa 只看这一张图，被挡住的背景是"编"出来的，
             背景一动就容易糊、容易闪。

    sttn  —— 把水印附近的帧当成一整段视频一起修（Spatial-Temporal Transformer）。
             它会去参考前后帧里没被挡住的同一块背景，所以背景变化时更自然。
             代价是要攒够一小段帧才能开始算，速度比 LaMa 慢。

共同的处理步骤：
    1. 按水印框外扩 padding 裁出一块图（sttn 会裁得更宽一点，见 crop_patch）；
    2. 在水印框的位置画白色掩码（再膨胀几个像素，盖住抗锯齿边）；
    3. 交给修复后端，得到补好的图；
    4. 带羽化贴回原帧。

因为送进模型的部分很小，显存占用和耗时都比较低。
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# 水印区域，统一用 (x, y, w, h) 表示，单位是像素，坐标基于原帧
Region = Tuple[int, int, int, int]


def resolve_device(device_pref: str = "auto"):
    """
    根据配置选择 torch 设备。

    auto：有 GPU 就用 GPU，否则用 CPU
    cuda：强制用 GPU，检测不到就退回 CPU 并提示
    cpu ：强制用 CPU
    """
    import torch

    pref = (device_pref or "auto").strip().lower()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if not torch.cuda.is_available():
            print("[警告] config.yaml 里 inpaint.device 写了 cuda，但当前 PyTorch 检测不到可用 GPU，"
                  "已改用 CPU。")
            return torch.device("cpu")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_out_of_memory(err: BaseException) -> bool:
    """判断异常是不是显存/内存不足。"""
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cuda error" in msg and "memory" in msg)


class InpaintEngine:
    """
    水印修复引擎。整个批量处理过程只创建一个实例，模型只加载一次。

    method 可选：
        "lama"   —— simple-lama-inpainting（逐帧，快，推荐默认）
        "sttn"   —— STTN 视频修复（成段一起修，背景变化时更自然，慢一些）
        "opencv" —— cv2.inpaint 的 TELEA 算法（降级方案，装不上模型时用）
    """

    def __init__(self, method: str = "lama", device: str = "auto",
                 padding: int = 20, mask_dilate: int = 4, feather_px: int = 0,
                 sttn_cfg: dict = None):
        self.method = (method or "lama").strip().lower()
        self.padding = max(0, int(padding))
        self.mask_dilate = max(0, int(mask_dilate))
        self.feather_px = max(0, int(feather_px))

        self.device = None
        self._lama = None
        self._sttn = None
        self._sttn_aspect = 1.0
        # STTN 的带子相对 ROI 放大多少倍，见 _band_box
        self._sttn_context = 2.0

        if self.method == "lama":
            self._init_lama(device)
        elif self.method == "sttn":
            self._init_sttn(device, sttn_cfg or {})
        elif self.method == "opencv":
            print("[信息] 使用 OpenCV 降级修复（cv2.inpaint / TELEA），效果会比 LaMa 弱一些。")
        else:
            raise ValueError("未知的修复方式：%s（可选 lama / sttn / opencv）" % method)

    # -- 初始化 ------------------------------------------------------------

    def _init_lama(self, device_pref: str) -> None:
        import torch
        from simple_lama_inpainting import SimpleLama

        self.device = resolve_device(device_pref)
        if self.device.type == "cuda":
            print("[信息] 正在加载 LaMa 模型到 GPU：%s" % torch.cuda.get_device_name(0))
        else:
            print("[信息] 正在加载 LaMa 模型到 CPU（速度会明显慢一些）")

        try:
            self._lama = SimpleLama(device=self.device)
        except TypeError:
            # 老版本的 SimpleLama 构造函数不接受 device 参数，手动把模型挪过去
            self._lama = SimpleLama()
            try:
                self._lama.device = self.device
                self._lama.model.to(self.device)
            except Exception:
                self.device = getattr(self._lama, "device", torch.device("cpu"))

        print("[信息] LaMa 模型加载完成。")

    def _fallback_to_cpu(self) -> None:
        """显存不够时把模型挪到 CPU 上继续跑。"""
        import torch

        self.device = torch.device("cpu")
        try:
            self._lama.device = self.device
            self._lama.model.to(self.device)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _init_sttn(self, device_pref: str, sttn_cfg: dict) -> None:
        """
        加载 STTN。权重首次运行会自动下载到 models/ 里（约 66MB）。
        """
        import torch

        from sttn import STTNInpainter

        self.device = resolve_device(device_pref)
        if self.device.type == "cuda":
            print("[信息] 正在加载 STTN 模型到 GPU：%s" % torch.cuda.get_device_name(0))
        else:
            print("[信息] 正在加载 STTN 模型到 CPU（会慢很多）")

        self._sttn = STTNInpainter(
            device=self.device,
            model_path=sttn_cfg.get("model_path") or None,
            segment_size=int(sttn_cfg.get("segment_size", 10)),
            neighbor_stride=int(sttn_cfg.get("neighbor_stride", 5)),
            reference_length=int(sttn_cfg.get("reference_length", 10)),
            min_segment_size=int(sttn_cfg.get("min_segment_size", 2)),
        )
        # 模型输入是 432x240，裁"带子"的时候要按这个宽高比来
        self._sttn_aspect = self._sttn.INPUT_W / float(self._sttn.INPUT_H)
        # 带子放大倍数：太小的话 mask 几乎占满整条带子，STTN 没东西可参考
        self._sttn_context = max(1.0, float(sttn_cfg.get("context_scale", 2.0)))
        print("[信息] STTN 模型加载完成（每次最多一起修 %d 帧，参考带子放大约 %.1f 倍）。"
              % (self._sttn.segment_size, self._sttn_context))

    # -- 对外接口 ----------------------------------------------------------

    def crop_patch(self, frame: np.ndarray, region: Region):
        """
        裁出要交给修复器的那块图（还没修），返回 (patch, patch_box, paste_box)。

        patch_box 是送进修复器的范围，paste_box 是最后贴回画面的范围。
        lama / opencv 是逐帧修的，两个框一样，都是水印框外扩 padding。
        sttn 要靠前后帧对齐背景，只看水印那一小块周围没东西可参考，
        所以另外裁一条更宽的"带子"送进去，修完再从带子里把水印那块抠出来。

        同一分辨率的视频、同一个 region，算出来的框每次都一样，
        所以调用方可以直接复用第一帧的 patch_box / paste_box。
        """
        paste_box = self._roi_box(frame.shape[:2], region)
        if self.method == "sttn":
            patch_box = self._band_box(paste_box, frame.shape[:2], self._sttn_aspect)
        else:
            patch_box = paste_box

        x0, y0, x1, y1 = patch_box
        return frame[y0:y1, x0:x1], patch_box, paste_box

    def repair_patches(self, patches: List[np.ndarray], region: Region,
                       patch_box: Tuple[int, int, int, int],
                       paste_box: Tuple[int, int, int, int]) -> List[np.ndarray]:
        """
        把 crop_patch 裁出来的一摞图修好，返回和 paste_box 同尺寸的结果列表。

        sttn 是整段一起算的（时间维度的信息全靠它）；
        其它后端逐帧独立修，互不影响。
        """
        if not patches:
            return []

        if self.method == "sttn":
            return self._sttn_repair(patches, region, patch_box, paste_box)

        x, y, w, h = (int(v) for v in region)
        out = []
        for patch in patches:
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                out.append(patch)
                continue
            mask = self._build_mask(patch.shape[0], patch.shape[1],
                                    (patch_box[0], patch_box[1], x, y, w, h))
            out.append(self._inpaint_roi(patch, mask))
        return out

    def paste_patch(self, frame: np.ndarray, patch: np.ndarray,
                    paste_box: Tuple[int, int, int, int],
                    feather_px: int = None) -> np.ndarray:
        """
        把修好的图贴回帧里（就地修改 frame 并返回）。

        paste_box   : crop_patch 返回的 (x0, y0, x1, y1)
        feather_px  : 边缘羽化像素，不传就用构造时设的 self.feather_px。
                      大于 0 时，四周留一条渐变带，从"修复结果"过渡到
                      "原画面"，免得贴回的位置出现一圈生硬的接缝。
                      这块图已经在水印框外扩了 padding，所以羽化带离水印还有
                      一段距离，不会把水印露出来。
        """
        x0, y0, x1, y1 = (int(v) for v in paste_box)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return frame
        if patch is None or patch.shape[0] != y1 - y0 or patch.shape[1] != x1 - x0:
            # 尺寸对不上就不贴，宁可不动也别贴歪
            return frame

        feather = self.feather_px if feather_px is None else max(0, int(feather_px))

        alpha = self._build_feather_alpha(patch.shape[0], patch.shape[1], feather)
        if alpha is None:
            frame[y0:y1, x0:x1] = patch
            return frame

        dst = frame[y0:y1, x0:x1]
        blended = patch.astype(np.float32) * alpha + dst.astype(np.float32) * (1.0 - alpha)
        dst[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame

    def process_frame(self, frame: np.ndarray, region: Region) -> np.ndarray:
        """
        修复单帧里的水印区域，返回修复后的新帧（不修改传入的 frame）。

        一步到位的写法，内部就是 crop_patch + repair_patches + paste_patch，
        方便单张图试效果。批量处理走 main.py 里那三步，效率更高。
        """
        patch, patch_box, paste_box = self.crop_patch(frame, region)
        fixed = self.repair_patches([patch], region, patch_box, paste_box)[0]
        out = frame.copy()
        self.paste_patch(out, fixed, paste_box)
        return out

    # -- 裁剪范围 ----------------------------------------------------------

    def _roi_box(self, frame_shape: Tuple[int, int], region: Region):
        """水印框外扩 padding，并裁到画面范围内。"""
        x, y, w, h = (int(v) for v in region)
        frame_h, frame_w = frame_shape[:2]
        return (max(0, x - self.padding),
                max(0, y - self.padding),
                min(frame_w, x + w + self.padding),
                min(frame_h, y + h + self.padding))

    def _band_box(self, roi_box: Tuple[int, int, int, int],
                  frame_shape: Tuple[int, int], aspect: float):
        """
        在 ROI 外面扩出一圈，凑成符合模型输入宽高比（432:240）的"带子"。

        为什么要往外扩：STTN 补洞靠的是"这一帧里别处没被挡住的像素"和"前后帧的同一块"。
        如果带子只比水印大一点点，mask 几乎占满整条带子，模型找不到可参考的内容，
        只能乱补，出来是一坨糊的。所以先按 context_scale 把 ROI 放大若干倍，
        让水印只占带子的一小部分。

        宽高比也不能乱给：模型输入固定 432x240，带子比例对不上就会被硬拉变形。
        所以扩完再调到 1.8，最后整条等比缩放进模型。

        带子尽量以水印为中心，但不能跑出画面；画面太小时按画面裁剪（这会略微变形，属少数情况）。
        """
        x0, y0, x1, y1 = (int(v) for v in roi_box)
        frame_h, frame_w = frame_shape[:2]
        rw, rh = x1 - x0, y1 - y0

        bw = int(math.ceil(rw * self._sttn_context))
        bh = int(math.ceil(rh * self._sttn_context))
        # 调到模型要的宽高比
        if bw / float(bh) < aspect:
            bw = int(math.ceil(bh * aspect))
        else:
            bh = int(math.ceil(bw / aspect))
        bw = min(bw, frame_w)
        bh = min(bh, frame_h)

        # 位置：以 ROI 中心对齐，再夹到"不越界且能包住 ROI"的区间里
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        bx0 = min(max(cx - bw // 2, max(0, x1 - bw)), min(x0, frame_w - bw))
        by0 = min(max(cy - bh // 2, max(0, y1 - bh)), min(y0, frame_h - bh))
        return (bx0, by0, bx0 + bw, by0 + bh)

    # -- 内部实现 ----------------------------------------------------------

    def _sttn_repair(self, patches: List[np.ndarray], region: Region,
                     patch_box: Tuple[int, int, int, int],
                     paste_box: Tuple[int, int, int, int]) -> List[np.ndarray]:
        """
        整段交给 STTN，再从修好的带子里把水印那块抠出来。

        只把水印那一块贴回画面，带子里其它位置就算 STTN 改动了也不管，
        免得把旁边的正常画面一起动了。
        """
        x, y, w, h = (int(v) for v in region)
        bx0, by0, bx1, by1 = (int(v) for v in patch_box)
        band_h, band_w = by1 - by0, bx1 - bx0

        mask = self._build_mask(band_h, band_w, (bx0, by0, x, y, w, h))
        fixed_bands = self._sttn.inpaint(patches, mask)

        ox0, oy0, ox1, oy1 = (int(v) for v in paste_box)
        rx, ry = ox0 - bx0, oy0 - by0
        rh, rw = oy1 - oy0, ox1 - ox0

        out = []
        for band in fixed_bands:
            out.append(band[ry:ry + rh, rx:rx + rw])
        return out

    def _build_feather_alpha(self, h: int, w: int, feather: int):
        """
        生成羽化用的 alpha 图（浮点，形状 (h, w, 1)）。

        ROI 中间是 1（完全用修复结果），四周 feather 像素内从 0 平滑升到 1。
        最外侧一行/一列 alpha 接近 0，也就是完全保留原画面，接缝就看不出来了。
        """
        if feather <= 0:
            return None

        # 别让羽化带超过一半边长，否则中间就没有"实心"区域了
        f = int(min(feather, max(1, min(h, w) // 2)))

        # 0.5/f 起步、逐级升高，f 个采样点
        ramp = (np.arange(f, dtype=np.float32) + 0.5) / float(f)

        alpha = np.ones((h, w), dtype=np.float32)
        alpha[:f, :] *= ramp[:, None]        # 上边
        alpha[h - f:, :] *= ramp[::-1, None]  # 下边
        alpha[:, :f] *= ramp[None, :]        # 左边
        alpha[:, w - f:] *= ramp[::-1][None, :]  # 右边

        return alpha[:, :, None]

    def _build_mask(self, roi_h: int, roi_w: int, geom: Tuple[int, int, int, int, int, int]) -> np.ndarray:
        """在 ROI 坐标系里生成二值掩码：白色(255)=需要修复的水印区域。"""
        x0, y0, x, y, w, h = geom

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # 水印框在 ROI 里的位置
        mx0 = max(0, x - x0)
        my0 = max(0, y - y0)
        mx1 = min(roi_w, x + w - x0)
        my1 = min(roi_h, y + h - y0)

        if mx1 <= mx0 or my1 <= my0:
            return mask

        mask[my0:my1, mx0:mx1] = 255

        if self.mask_dilate > 0:
            ksize = self.mask_dilate * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _inpaint_roi(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """对单块 ROI 做修复。"""
        if self.method == "opencv":
            return cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)
        return self._lama_roi(roi, mask)

    def _lama_roi(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import numpy as _np
        from PIL import Image

        image_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask, mode="L")

        try:
            result = self._lama(image_pil, mask_pil)
        except Exception as err:
            if _is_out_of_memory(err) and self.device is not None and self.device.type == "cuda":
                print("[警告] CUDA 显存不足，已把 LaMa 回退到 CPU 继续处理。")
                print("       想彻底解决可以把 config.yaml 里的 inpaint.padding 调小（比如 8）。")
                self._fallback_to_cpu()
                result = self._lama(image_pil, mask_pil)
            else:
                raise

        if isinstance(result, Image.Image):
            arr = _np.array(result.convert("RGB"))
        else:
            arr = _np.asarray(result)
            if arr.ndim == 2:
                arr = _np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != _np.uint8:
                arr = _np.clip(arr, 0, 255).astype(_np.uint8)

        # simple-lama-inpainting 内部会先把输入补齐到 8 的倍数（只在右侧和下方 pad），
        # 而且不会裁回去，所以返回的图可能比 ROI 大一点，这里自己裁掉。
        target_h, target_w = roi.shape[:2]
        if arr.shape[0] != target_h or arr.shape[1] != target_w:
            if arr.shape[0] >= target_h and arr.shape[1] >= target_w:
                arr = arr[:target_h, :target_w]
            else:
                arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
