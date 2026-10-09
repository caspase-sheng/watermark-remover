# -*- coding: utf-8 -*-
"""
修复区域的时间轴平滑。

逐帧独立跑 LaMa 的时候，同一块 ROI 在相邻帧上得到的结果会有细微差别。
背景一变化（光照变了、纹理在流动），这些差别就会在肉眼看来变成修复区域的
帧间波动/闪烁。这里把每帧的 ROI 沿时间轴做一次滤波，让相邻帧的结果更接近，
波动就被压下去了。

只动 ROI 那一小块，画面其余部分不受影响。
"""

from __future__ import annotations

import sys
from typing import Sequence, Union

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _as_stack(roi_frames: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
    """把 ROI 序列统一成 (T, H, W, 3) 的 4 维数组。"""
    if isinstance(roi_frames, np.ndarray) and roi_frames.ndim == 4:
        return roi_frames

    frames = list(roi_frames)
    if not frames:
        raise ValueError("temporal_smooth 收到空的 ROI 序列。")
    return np.stack([np.asarray(f) for f in frames], axis=0)


def temporal_smooth(roi_frames, window: int = 3, method: str = "median") -> np.ndarray:
    """
    沿时间轴对 ROI 帧序列做平滑。

    参数：
        roi_frames : ROI 序列。可以是 (T, H, W, 3) 的四维数组，
                     也可以是长度 T、每项 HxWx3 的列表。
                     要求每一帧的尺寸一致（水印框固定，本来就一致）。
        window     : 时间窗口大小，默认 3。小于 2 表示不平滑，原样返回。
                     传偶数会自动加一，取奇数窗口。
        method     : 'median' —— 中值滤波（scipy.ndimage.median_filter，默认）。
                     中值对偶然一帧的异常输出不敏感，适合压闪烁。
                     'mean'   —— 均值滤波（scipy.ndimage.uniform_filter），
                     结果更"糊"一点，一般用不上。

    返回：
        (T, H, W, 3) 的 uint8 数组，帧数和尺寸与输入一致。

    实现上就是给 scipy 一个 (window, 1, 1, 1) 的 size，
    只在时间轴（第 0 维）上取窗口，H/W/通道三个维度各自独立，互不影响。
    """
    arr = _as_stack(roi_frames)

    win = int(window)
    if win < 2 or arr.shape[0] < 2:
        # 只有一帧、或者关掉了平滑，直接把原序列转成数组返回
        return arr.astype(np.uint8, copy=False)

    if win % 2 == 0:
        win += 1

    # size 的四个维度分别对应 (时间, 高, 宽, 通道)，只在时间轴上有窗口
    size = (win, 1, 1, 1)

    name = (method or "median").strip().lower()
    if name == "median":
        from scipy.ndimage import median_filter
        # mode="nearest"：首尾帧缺的邻居用自己补，边界帧不会被拉偏
        out = median_filter(arr, size=size, mode="nearest")
    elif name == "mean":
        from scipy.ndimage import uniform_filter
        out = uniform_filter(arr.astype(np.float32), size=size, mode="nearest")
        out = np.clip(out, 0, 255)
    else:
        raise ValueError("未知的时间平滑方法：%s（可选 median / mean）" % method)

    return out.astype(np.uint8)
