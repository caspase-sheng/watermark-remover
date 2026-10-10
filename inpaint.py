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
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import PROJECT_ROOT

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
                 sttn_cfg: dict = None, lama_batch_size: int = 0):
        self.method = (method or "lama").strip().lower()
        self.padding = max(0, int(padding))
        self.mask_dilate = max(0, int(mask_dilate))
        self.feather_px = max(0, int(feather_px))

        # LaMa 一次喂几张 ROI。0 表示按显存自动定，1 表示逐帧（老路子）。
        # 逐帧调用每次都要等 GPU 同步、来回搬 PIL，GPU 大部分时间闲着，
        # 打包成一批送进去能快好几倍，见 _lama_repair_batch。
        self.lama_batch_size = max(0, int(lama_batch_size or 0))
        # 自动模式下算出来的批大小，跑完第一个视频就固定下来复用
        self._batch_eff = None

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

        # simple-lama-inpainting 找权重的时候先看环境变量 LAMA_MODEL，设了就直接用这个文件，
        # 没设才去 torch hub 缓存里找（缓存没有还会联网下 196MB）。
        # 项目 models/ 下本来就放了一份，所以这里优先指过去，省得依赖外面的缓存。
        local_weight = PROJECT_ROOT / "models" / "big-lama.pt"
        if local_weight.exists():
            os.environ.setdefault("LAMA_MODEL", str(local_weight))
            print("[信息] LaMa 权重路径：%s" % local_weight)
        else:
            print("[信息] 未找到 %s，回退到 torch hub 缓存。"
                  % (PROJECT_ROOT / "models" / "big-lama.pt"))

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
        geom = (patch_box[0], patch_box[1], x, y, w, h)

        # LaMa 逐帧是"一张一张独立修"的，帧和帧之间没有依赖，
        # 所以可以打包成一批一次算完，结果和逐帧一模一样（见 _lama_repair_batch）。
        # 只有尺寸全一致才敢打包；尺寸不齐或者显存不够会自动落回下面的逐帧循环。
        if self.method == "lama" and len({p.shape for p in patches}) == 1:
            batched = self._lama_repair_batch(patches, geom)
            if batched is not None:
                return batched

        out = []
        for patch in patches:
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                out.append(patch)
                continue
            mask = self._build_mask(patch.shape[0], patch.shape[1], geom)
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

    # -- 批量推理 ----------------------------------------------------------
    #
    # 逐帧调 LaMa 的时候，每次调用都要等一次 GPU 同步，还要在 numpy / PIL / tensor
    # 之间来回搬，真正在算的时间只占一小部分。243 帧实测要 6 秒多，几乎全是等出来的。
    # 把若干帧打包成一批送进去，同样的计算量两秒左右就能跑完，结果完全一样。

    def _resolve_batch_size(self, patch_h: int, patch_w: int) -> int:
        """这批图一次喂几张。配置里写死了就用配置，写 0 就按显存估一个。"""
        if self.lama_batch_size > 0:
            return self.lama_batch_size
        if self._batch_eff is None:
            self._batch_eff = self._auto_lama_batch_size(patch_h, patch_w)
        return self._batch_eff

    def _auto_lama_batch_size(self, patch_h: int, patch_w: int) -> int:
        """
        按显存粗估批大小。

        LaMa 的中间激活不好精确算，就按"单张输入张量字节数 × 12"折算每张的显存开销，
        再拿总显存的四分之一做预算。宁可估小一点，真不够还有 OOM 减半兜底。
        """
        if self.device is None or self.device.type != "cuda":
            # CPU 上批量省的是 Python 开销，收益有限，别开太大
            return 4

        import torch

        total = int(torch.cuda.get_device_properties(0).total_memory)
        gb = total / float(1024 ** 3)
        # 8G 卡实际只报 7.99G，阈值别卡在整数上，否则掉一档
        if gb >= 12:
            cap = 16
        elif gb >= 7.5:
            cap = 8
        elif gb >= 5.5:
            cap = 6
        elif gb >= 3.5:
            cap = 4
        else:
            cap = 2

        per_sample = 3 * max(1, patch_h) * max(1, patch_w) * 4 * 12
        budget = int(total * 0.25)
        return max(1, min(cap, max(1, budget // per_sample)))

    def _lama_repair_batch(self, patches: List[np.ndarray],
                           geom: Tuple[int, int, int, int, int, int]) -> Optional[List[np.ndarray]]:
        """
        把一批同样大小的 ROI 打包一次性送进 LaMa，返回修好的列表。

        做的和逐帧的 _lama_roi 是同一件事：BGR 转 RGB、归一化、只往右往下补齐到
        8 的倍数、过一次模型、裁回原尺寸、再转回 BGR。唯一区别是批大小从 1 变成 N，
        模型参数和掩码都没变，所以输出和逐帧跑是一样的（只有浮点上的极小差异）。

        显存不够就把批大小减半重试；减到 1 直接返回 None，让调用方走逐帧老路。
        """
        h, w = patches[0].shape[:2]
        if h < 2 or w < 2:
            return None

        bs = self._resolve_batch_size(h, w)
        if bs <= 1:
            return None

        # simple-lama-inpainting 只往右、往下 pad 到 8 的倍数，这里保持一致
        mod = 8
        oh = (h + mod - 1) // mod * mod
        ow = (w + mod - 1) // mod * mod

        # 同一个视频里水印位置固定，整批共用一张掩码，只算一次
        mask = self._build_mask(h, w, geom)
        mask_pad = np.pad((mask > 0).astype(np.float32),
                          ((0, oh - h), (0, ow - w)), mode="symmetric")

        out: List[np.ndarray] = []
        i = 0
        while i < len(patches):
            chunk = patches[i:i + bs]
            try:
                out.extend(self._lama_batch_call(chunk, mask_pad, h, w, oh, ow))
            except Exception as err:
                if (_is_out_of_memory(err) and bs > 1
                        and self.device is not None and self.device.type == "cuda"):
                    import torch
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    print("[警告] LaMa 批量推理显存不足，批大小降到 %d 后重试。" % bs)
                    if bs == 1:
                        return None
                    continue
                raise
            i += len(chunk)

        # 记住这次实际跑通的批大小，后面几个视频直接复用
        self._batch_eff = bs
        return out

    def _lama_batch_call(self, chunk: List[np.ndarray], mask_pad: np.ndarray,
                         h: int, w: int, oh: int, ow: int) -> List[np.ndarray]:
        """打包一小批，只调一次模型。"""
        import torch

        # BGR HWC -> RGB CHW，再除以 255，和 _lama_roi 里 PIL 那条路走出来的一致
        batch = np.stack([np.transpose(p[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
                          for p in chunk], axis=0)
        batch = np.pad(batch, ((0, 0), (0, 0), (0, oh - h), (0, ow - w)), mode="symmetric")

        images = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        masks = torch.from_numpy(np.ascontiguousarray(
            np.repeat(mask_pad[None, None], len(chunk), axis=0))).to(self.device)

        with torch.inference_mode():
            res = self._lama.model(images, masks)
            arr = res.permute(0, 2, 3, 1).detach().cpu().numpy()

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)[:, :h, :w]
        return [np.ascontiguousarray(a[:, :, ::-1]) for a in arr]
