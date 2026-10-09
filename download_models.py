# -*- coding: utf-8 -*-
"""
把 LaMa 和 STTN 的预训练权重下载到项目的 models/ 目录。

权重体积大，不适合放进 Git 仓库，所以这个脚本单独拎出来，
克隆完项目跑一次就行：

    python download_models.py            # 两个都下
    python download_models.py --lama     # 只下 LaMa
    python download_models.py --sttn     # 只下 STTN
    python download_models.py --lama-url <地址>   # 自己指定 big-lama.pt 的下载地址

下好之后，用环境变量把路径指给程序（也可以不设，程序会先去默认位置找）：

    set LAMA_MODEL=%cd%\models\big-lama.pt
    set STTN_MODEL=%cd%\models\sttn-det.safetensors

或者直接把 config.yaml 里的 sttn.model_path 指向 models 里的文件。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# LaMa 的权重。simple-lama-inpainting 默认也从这里下，只是它下到
# 用户目录的 torch 缓存里；我们统一放到项目的 models/ 下，方便一起管理。
DEFAULT_LAMA_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/"
    "v0.1.0/big-lama.pt"
)
LAMA_NAME = "big-lama.pt"


def _download(url: str, dst: Path) -> None:
    """带进度提示的分块下载，先写 .part，成功了再改名。"""
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 1.0:
                    last = now
                    if total > 0:
                        sys.stdout.write("      进度 %.1f%%（%.1f / %.1f MB）\r"
                                         % (done * 100.0 / total,
                                            done / 1048576.0, total / 1048576.0))
                    else:
                        sys.stdout.write("      已下载 %.1f MB\r" % (done / 1048576.0))
                    sys.stdout.flush()
    sys.stdout.write("\n")
    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("下载到的文件是空的")
    if dst.exists():
        dst.unlink()
    tmp.replace(dst)


def fetch_lama(url: str) -> Path:
    """下载 big-lama.pt（约 196MB）。"""
    target = MODELS_DIR / LAMA_NAME
    if target.exists() and target.stat().st_size > 0:
        print("[跳过] LaMa 权重已存在：%s（%.1f MB）"
              % (target, target.stat().st_size / 1048576.0))
        return target

    print("[信息] 开始下载 LaMa 权重：%s" % url)
    try:
        _download(url, target)
    except Exception as err:  # noqa: BLE001
        print("[错误] 下载失败：%s" % err)
        print("       国内直连 GitHub Releases 经常很慢或被重置，可以：")
        print("       1) 用 --lama-url 换一个镜像地址；")
        print("       2) 自己下好 big-lama.pt 放进 %s ；" % MODELS_DIR)
        print("       3) 暂时不用 LaMa，把 config.yaml 的 inpaint_method 改成 opencv。")
        raise
    print("[完成] LaMa 权重：%s（%.1f MB）"
          % (target, target.stat().st_size / 1048576.0))
    return target


def fetch_sttn() -> Path:
    """下载 STTN 权重（约 66MB），直接复用 sttn.py 里写好的下载逻辑。"""
    import sttn  # 放在函数里，避免没装 torch 时连 LaMa 也下不了

    target = sttn.default_model_path()
    if target.exists() and target.stat().st_size > 0:
        print("[跳过] STTN 权重已存在：%s（%.1f MB）"
              % (target, target.stat().st_size / 1048576.0))
        return target

    print("[信息] 开始下载 STTN 权重 ...")
    target = sttn.download_model(target)
    print("[完成] STTN 权重：%s（%.1f MB）"
          % (target, target.stat().st_size / 1048576.0))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 LaMa / STTN 预训练权重到 models/ 目录")
    parser.add_argument("--lama", action="store_true", help="只下 LaMa 权重")
    parser.add_argument("--sttn", action="store_true", help="只下 STTN 权重")
    parser.add_argument("--lama-url", default=os.environ.get("LAMA_MODEL_URL", DEFAULT_LAMA_URL),
                        help="自定义 big-lama.pt 的下载地址")
    args = parser.parse_args()

    # 两个都没指定就都下
    want_lama = args.lama or not (args.lama or args.sttn)
    want_sttn = args.sttn or not (args.lama or args.sttn)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("下载模型权重到：%s" % MODELS_DIR)
    print("=" * 60)

    ok = True
    if want_lama:
        try:
            fetch_lama(args.lama_url)
        except Exception:
            ok = False
    if want_sttn:
        try:
            fetch_sttn()
        except Exception as err:  # noqa: BLE001
            print("[错误] STTN 权重下载失败：%s" % err)
            ok = False

    print()
    print("=" * 60)
    print("提示：权重不随仓库分发，下好之后用环境变量指过去就行（可选）：")
    print("    set LAMA_MODEL=%s" % (MODELS_DIR / LAMA_NAME))
    print("    set STTN_MODEL=%s" % (MODELS_DIR / "sttn-det.safetensors"))
    print("或者把 config.yaml 里的 sttn.model_path 指向文件。")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
