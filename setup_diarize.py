#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""安装 pyannote 说话人分割环境（venv-diarize）。

由 setup_diarize.cmd 调用；也可以直接运行：

    venv\Scripts\python.exe setup_diarize.py

若希望复用已安装的 PyTorch，可通过 JISHENG_TORCH_PYTHON 指定其解释器。
否则尝试复用项目分离环境的基底，找不到时创建普通虚拟环境。
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / "venv-diarize"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
REQ = HERE / "requirements-diarize.txt"
COMPAT = HERE / "pyannote_compat.py"


def log(msg=""):
    print(msg, flush=True)


def _has_torch(python: Path) -> bool:
    if not python.exists():
        return False
    try:
        r = subprocess.run([str(python), "-c", "import torch"],
                           capture_output=True, text=True, timeout=180)
    except Exception:
        return False
    return r.returncode == 0


def find_base_python():
    """返回 (python 路径 或 None, 说明)。优先带 torch 的解释器。"""
    preferred = os.environ.get("JISHENG_TORCH_PYTHON")
    candidates = [Path(preferred).expanduser()] if preferred else []

    # venv-sep 的配置会记录它的基底解释器。
    cfg = HERE / "venv-sep" / "pyvenv.cfg"
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().lower().startswith("home"):
                home = line.split("=", 1)[1].strip() if "=" in line else ""
                if home:
                    candidates.append(Path(home) / "python.exe")

    for c in candidates:
        if _has_torch(c):
            return c, "复用它的 torch"

    return None, "没找到带 torch 的 Python"


def _run(cmd, **kw):
    log("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def main() -> int:
    log("")
    log("  安装说话人分割环境（venv-diarize）")
    log("  ===================================")
    log("")

    if not REQ.is_file():
        log(f"[X] 找不到 {REQ}")
        return 1

    if VENV_PY.exists():
        log("  检测到已有 venv-diarize，跳过创建，直接安装/升级依赖。")
    else:
        base, why = find_base_python()
        if base is not None:
            log(f"  基底 Python : {base}")
            log(f"               （{why}，不会重复下载 torch）")
            cmd = [base, "-m", "venv", "--system-site-packages", VENV_DIR]
        else:
            log(f"  [!] {why}，改用当前解释器建环境。")
            log("      这会额外下载约 2.5 GB 的 torch，耗时较长。")
            cmd = [Path(sys.executable), "-m", "venv", VENV_DIR]

        log("")
        log("  创建 venv-diarize ...")
        if _run(cmd).returncode != 0:
            log("[X] 创建虚拟环境失败。")
            return 1

    log("")
    log("  安装 pyannote.audio 3.x ...")
    r = _run([VENV_PY, "-m", "pip", "install",
              "--disable-pip-version-check", "--no-input", "-r", REQ])
    if r.returncode != 0:
        log("[X] 依赖安装失败。")
        return 1

    log("")
    log("  自检 ...")
    # 直接跑兼容层模块的自检：它会先应用 torchaudio / weights_only 兼容处理，
    # 再 import pyannote.audio。裸 import 在这套新版依赖下是会失败的。
    if _run([VENV_PY, COMPAT]).returncode != 0:
        log("[X] 自检失败：pyannote 装上了但导入不了。")
        return 1

    log("")
    log("  安装完成。")
    log("")
    log("  模型来源（二选一）：")
    log("")
    log("  [A] 本机 models\\pyannote\\ 已放好模型（推荐，不需要 HF_TOKEN）")
    log("      从 ModelScope 取回的官方权重，合计约 31 MB：")
    log(f"        {HERE / 'models' / 'pyannote' / 'segmentation-3.0' / 'pytorch_model.bin'}")
    log(f"        {HERE / 'models' / 'pyannote' / 'wespeaker-voxceleb-resnet34-LM' / 'pytorch_model.bin'}")
    log('      两个都在的话，直接跑：jisheng.cmd "某集.mkv" --diarize')
    log("")
    log("  [B] 从 HuggingFace 在线下载（需要能打开 huggingface.co）")
    log("      1. 接受两个仓库的用户协议：")
    log("         https://huggingface.co/pyannote/speaker-diarization-3.1")
    log("         https://huggingface.co/pyannote/segmentation-3.0")
    log("      2. 将只读令牌保存在当前终端的 HF_TOKEN 环境变量中。")
    log('      3. jisheng.cmd "某集.mkv" --diarize')
    log("")
    log("  用 jisheng.cmd --check 可随时查看当前用的是哪种模型。")
    log("  详见 说话人分割说明.md")
    log("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
