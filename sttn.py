# -*- coding: utf-8 -*-
"""
STTN（Spatial-Temporal Transformer Network）视频修复。

和 LaMa 只搞定"一张图"不一样，STTN 是把一小段视频帧当成整体一起看：
被水印盖住的地方，它会去参考前后若干帧里"没被盖住"的同一块背景，
把画面补回来。所以背景一直在动的时候（镜头平移、物体经过），
STTN 补出来的内容比 LaMa 更接近真实画面，LaMa 只能凭空画一个大概。

这里用的是带 mask 的版本（社区里叫 sttn-det，对应官方 researchmm/STTN）：
我们把水印框做成 mask 一起喂进去，模型只填 mask 里面的洞，
不会被水印挡住的地方它一律不动。

用法上对外只暴露 STTNInpainter：
    inp = STTNInpainter(device=..., segment_size=10)
    fixed = inp.inpaint(band_frames, mask)   # 输入输出都是 BGR 的帧列表
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------
# 模型权重
# --------------------------------------------------------------------------

# sttn-det 是带 mask 的那一份权重，和下面 InpaintGenerator 的结构一一对应。
# 文件里存的是 PyTorch 的 state_dict（112 个张量），键名直接用 encoder./decoder./transformer. 开头。
DEFAULT_MODEL_NAME = "sttn-det.safetensors"

# 按顺序试这几个地址。国内直连 huggingface.co 基本不通，hf-mirror 是镜像。
MODEL_URLS = (
    "https://hf-mirror.com/Dev-Jahn/sttn-det/resolve/main/model.safetensors",
    "https://huggingface.co/Dev-Jahn/sttn-det/resolve/main/model.safetensors",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def default_model_path() -> Path:
    """权重默认放在项目里的 models/ 目录，方便一眼看到。"""
    env = os.environ.get("STTN_MODEL")
    if env:
        return Path(env)
    return DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME


def download_model(target: Path, urls=MODEL_URLS) -> Path:
    """
    首次运行时把权重下下来（约 66MB），下好以后就不再联网。

    下载先写 .part 临时文件，成功后再改名，
    这样中途断了不会留下一个"看起来存在但其实残缺"的文件。
    """
    target = Path(target)
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")

    last_err = None
    for url in urls:
        try:
            print("[信息] 正在下载 STTN 权重：%s" % url)
            _download(url, tmp)
            if tmp.stat().st_size <= 0:
                raise RuntimeError("下载到的文件是空的")
            if target.exists():
                target.unlink()
            tmp.replace(target)
            print("[信息] STTN 权重已保存到：%s（%.1f MB）"
                  % (target, target.stat().st_size / 1024.0 / 1024.0))
            return target
        except Exception as err:  # noqa: BLE001
            last_err = err
            print("[警告] 这个地址下不动（%s），换下一个试试。" % err)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    raise RuntimeError(
        "STTN 权重下载失败：%s\n"
        "       可以手动下载 %s\n"
        "       另存为 %s，或者用 STTN_MODEL 环境变量指定路径。"
        % (last_err, MODEL_URLS[0], target)
    )


def _download(url: str, dst: Path) -> None:
    """带进度提示的分块下载。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_print = 0.0
        with open(dst, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_print > 1.0:
                    last_print = now
                    if total > 0:
                        print("      进度 %.1f%%（%.1f / %.1f MB）"
                              % (done * 100.0 / total, done / 1048576.0, total / 1048576.0),
                              end="\r")
                    else:
                        print("      已下载 %.1f MB" % (done / 1048576.0), end="\r")
    print()


# safetensors 的 dtype 名字到 numpy 的映射。我们这份权重是 F32，其余只是防一手。
_SAFETENSORS_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def load_safetensors(path) -> Dict[str, "object"]:
    """
    读取 safetensors 文件，返回 {张量名: torch.Tensor}。

    没直接用 safetensors 这个库，因为环境里不一定装得上。
    文件格式很简单：开头 8 字节是头部 JSON 的长度，接着是头部 JSON，
    后面就是所有张量按顺序拼起来的裸数据，照着 offsets 切一下就行。
    """
    import torch

    path = Path(path)
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
        blob = f.read()

    tensors: Dict[str, "torch.Tensor"] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError("不支持的数据类型：%s（%s）" % (info["dtype"], name))
        start, end = info["data_offsets"]
        # frombuffer 出来的数组是只读的（指向文件缓冲区），copy 一份变成可写再转 tensor
        arr = np.frombuffer(blob[start:end], dtype=dtype).reshape(info["shape"]).copy()
        tensors[name] = torch.from_numpy(arr)
    return tensors


def load_model_state(path) -> Dict[str, "object"]:
    """
    统一读权重的入口，safetensors 和 VSR 那种 {'netG': ...} 的 .pth 都认。
    """
    import torch

    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        return load_safetensors(path)

    data = torch.load(str(path), map_location="cpu")
    if isinstance(data, dict) and "netG" in data:
        return data["netG"]
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


# --------------------------------------------------------------------------
# 网络结构（照官方 researchmm/STTN 的 mask 版本抄的，只留推理要用的部分）
# --------------------------------------------------------------------------

def _build_network():
    """把 torch 相关的东西都放在函数里，没装 torch 时 import 本模块也不会炸。"""
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class deconv(nn.Module):
        """先双线性放大两倍，再卷一下。解码器靠它把特征图一步步还原成图。"""

        def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
            super().__init__()
            self.conv = nn.Conv2d(input_channel, output_channel,
                                  kernel_size=kernel_size, stride=1, padding=padding)

        def forward(self, x):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            return self.conv(x)

    class MultiHeadedAttention(nn.Module):
        """
        时空注意力。把特征图按 patchsize 切块，块内所有位置、所有帧放在一起算注意力，
        这样第 t 帧被挡住的地方就能去第 t-1 / t+1 帧的同一块里找内容。
        """

        def __init__(self, patchsize, d_model):
            super().__init__()
            self.patchsize = patchsize
            self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.output_linear = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True))

        def forward(self, x, m, b, c):
            bt, _, h, w = x.size()
            t = bt // b
            d_k = c // len(self.patchsize)
            output = []
            _query = self.query_embedding(x)
            _key = self.key_embedding(x)
            _value = self.value_embedding(x)
            for (width, height), query, key, value in zip(
                    self.patchsize,
                    torch.chunk(_query, len(self.patchsize), dim=1),
                    torch.chunk(_key, len(self.patchsize), dim=1),
                    torch.chunk(_value, len(self.patchsize), dim=1)):
                out_w, out_h = w // width, h // height

                # mask 缩到 patch 粒度：一个 patch 里过半像素被挡，就当整块被挡
                mm = m.view(b, t, 1, out_h, height, out_w, width)
                mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b, t * out_h * out_w, height * width)
                mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t * out_h * out_w, 1)

                query = query.view(b, t, d_k, out_h, height, out_w, width)
                query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b, t * out_h * out_w, d_k * height * width)
                key = key.view(b, t, d_k, out_h, height, out_w, width)
                key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b, t * out_h * out_w, d_k * height * width)
                value = value.view(b, t, d_k, out_h, height, out_w, width)
                value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b, t * out_h * out_w, d_k * height * width)

                # 被挡住的 query 不去关注同样被挡住的 key，否则等于自己抄自己
                scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
                scores = scores.masked_fill(mm, -1e9)
                y = torch.matmul(F.softmax(scores, dim=-1), value)

                y = y.view(b, t, out_h, out_w, d_k, height, width)
                y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
                output.append(y)
            output = torch.cat(output, 1)
            return self.output_linear(output)

    class FeedForward(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True))

        def forward(self, x):
            return self.conv(x)

    class TransformerBlock(nn.Module):
        def __init__(self, patchsize, hidden=128):
            super().__init__()
            self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
            self.feed_forward = FeedForward(hidden)

        def forward(self, x):
            x, m, b, c = x["x"], x["m"], x["b"], x["c"]
            x = x + self.attention(x, m, b, c)
            x = x + self.feed_forward(x)
            return {"x": x, "m": m, "b": b, "c": c}

    class InpaintGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            channel = 256
            stack_num = 8
            # 4 个尺度，(宽, 高)。对应 432x240 的输入经编码器下采样 4 倍后的 60x108。
            patchsize = [(108, 60), (36, 20), (18, 10), (9, 5)]
            self.transformer = nn.Sequential(
                *[TransformerBlock(patchsize, hidden=channel) for _ in range(stack_num)])

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.decoder = nn.Sequential(
                deconv(channel, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                deconv(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            )

        def infer(self, feat, masks):
            """只跑 transformer。返回补全后的特征，交给 decoder 出图。"""
            t, c, h, w = masks.size()
            masks = F.interpolate(masks.view(t, c, h, w), scale_factor=1.0 / 4)
            c = feat.size(1)
            return self.transformer({"x": feat, "m": masks, "b": 1, "c": c})["x"]

    return InpaintGenerator


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cuda error" in msg and "memory" in msg)


# --------------------------------------------------------------------------
# 对外的修复器
# --------------------------------------------------------------------------

class STTNInpainter:
    """
    把一段同尺寸的 BGR 帧序列 + 一张 mask 交给 STTN，返回补好的帧序列。

    segment_size 是"一次塞进模型多少帧"。帧数越多，模型能看到的时间跨度越长，
    效果越好，但注意力是 O(帧数^2) 的，显存涨得非常快。
    显存不够时会自动砍半重试（比如 10 -> 5 -> 2），所以不用自己盯着 nvidia-smi。
    """

    # 模型是按这个尺寸训练的，输入必须缩放到这儿。
    # 432x240 -> 编码器下采样 4 倍得到 108x60，正好能被上面的 patchsize 整除。
    INPUT_W = 432
    INPUT_H = 240

    def __init__(self, device=None, model_path=None, segment_size: int = 10,
                 neighbor_stride: int = 5, reference_length: int = 10,
                 min_segment_size: int = 2):
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.segment_size = max(2, int(segment_size))
        self.min_segment_size = max(2, min(int(min_segment_size), self.segment_size))
        self.neighbor_stride = max(1, int(neighbor_stride))
        self.reference_length = max(1, int(reference_length))

        self.model_path = Path(model_path) if model_path else default_model_path()
        if not self.model_path.exists():
            self.model_path = download_model(self.model_path)

        net_cls = _build_network()
        self.model = net_cls()
        state = load_model_state(self.model_path)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    # -- 对外接口 ----------------------------------------------------------

    def inpaint(self, frames: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        """
        frames : 同一个尺寸的 BGR 帧（uint8），一般是水印附近裁出来的一条带子
        mask   : 同尺寸的二值图，255/True 表示要修复的水印区域

        返回修复后的帧，尺寸和输入一致。
        """
        if not frames:
            return []

        h, w = frames[0].shape[:2]
        small_frames = [self._to_model_size(f) for f in frames]
        small_mask = self._mask_to_model_size(mask)

        results: List[np.ndarray] = []
        seg = self.segment_size
        idx = 0
        while idx < len(small_frames):
            chunk = small_frames[idx:idx + seg]
            try:
                out = self._run_chunk(chunk, small_mask)
            except Exception as err:  # noqa: BLE001
                if _is_oom(err) and seg > self.min_segment_size:
                    new_seg = max(self.min_segment_size, seg // 2)
                    print("    [警告] STTN 显存不足，segment_size 从 %d 降到 %d 重试。"
                          % (seg, new_seg))
                    self._free_cuda()
                    seg = new_seg
                    continue  # 这一段重新按更小的窗口来
                raise
            results.extend(out)
            idx += len(chunk)

        # 缩回原来的尺寸
        return [cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR) for r in results]

    # -- 内部实现 ----------------------------------------------------------

    def _to_model_size(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)

    def _mask_to_model_size(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = (mask > 127).astype(np.uint8) * 255
        # mask 用最近邻缩放，不然边缘会糊出一堆半透明的假像素
        small = cv2.resize(binary, (self.INPUT_W, self.INPUT_H),
                           interpolation=cv2.INTER_NEAREST)
        return (small > 127).astype(np.uint8)

    def _run_chunk(self, frames: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        """
        跑一个窗口。整体思路和官方 test.py 一样：

        1. 先把带水印的帧编码成特征（mask 区域先抹掉）；
        2. 逐窗口做时空注意力，被挡住的块会去别的帧里找没被挡的块来补；
        3. 解码出图，mask 以外的地方贴回原帧。

        官方实现里还有"参考帧"（在整段里等间隔抽几帧一起送进去），
        帧数不够时参考帧就是空的，不影响跑。
        """
        import torch

        t = len(frames)
        feats = self._frames_to_tensor(frames).unsqueeze(0) * 2 - 1
        masks = self._mask_batch_tensor(mask, t)

        feats = feats.to(self.device)
        masks = masks.to(self.device)
        frames_u8 = [f for f in frames]

        comp: List[Optional[np.ndarray]] = [None] * t
        stride = max(1, min(self.neighbor_stride, max(1, t // 2)))

        with torch.no_grad():
            enc = self.model.encoder((feats * (1 - masks)).view(t, 3, self.INPUT_H, self.INPUT_W))
            _, c, fh, fw = enc.size()
            enc = enc.view(1, t, c, fh, fw)

            for f in range(0, t, stride):
                neighbor_ids = list(range(max(0, f - stride), min(t, f + stride + 1)))
                ref_ids = self._ref_index(neighbor_ids, t)
                picked = neighbor_ids + ref_ids

                pred_feat = self.model.infer(enc[0, picked, :, :, :], masks[0, picked, :, :, :])
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
                pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255

                for i, idx in enumerate(neighbor_ids):
                    # 模型是在 RGB 空间里算的（_frames_to_tensor 里做过 BGR->RGB），
                    # 这是要贴回 BGR 的帧，所以先转回来，不然红蓝通道会反过来
                    img = cv2.cvtColor(pred_img[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
                    m = mask[..., None]
                    merged = img * m + frames_u8[idx] * (1 - m)
                    if comp[idx] is None:
                        comp[idx] = merged.astype(np.float32)
                    else:
                        # 相邻窗口会重叠，重叠的帧两边各取一半，过渡自然点
                        comp[idx] = comp[idx] * 0.5 + merged.astype(np.float32) * 0.5

        return [np.clip(c if c is not None else frames_u8[i], 0, 255).astype(np.uint8)
                for i, c in enumerate(comp)]

    def _ref_index(self, neighbor_ids: List[int], length: int) -> List[int]:
        """在整段里按 reference_length 等间隔抽参考帧，已经在窗口里的就不重复取。"""
        return [i for i in range(0, length, self.reference_length) if i not in neighbor_ids]

    def _frames_to_tensor(self, frames: List[np.ndarray]):
        """帧列表 -> (T, 3, H, W) 的 float 张量，范围 0~1，RGB 通道序。"""
        import torch

        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], axis=0)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).contiguous()
        return t.float().div(255.0)

    def _mask_batch_tensor(self, mask: np.ndarray, t: int):
        """把单张 mask 复制成 (1, T, 1, H, W)。水印位置固定不动，每帧的 mask 都一样。"""
        import torch

        m = torch.from_numpy(np.ascontiguousarray(mask)).float()
        m = m.view(1, 1, self.INPUT_H, self.INPUT_W).repeat(1, t, 1, 1, 1)
        return m

    def _free_cuda(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
