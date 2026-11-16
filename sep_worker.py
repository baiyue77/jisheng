#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
sep_worker.py —— 人声分离子进程

放在独立环境里运行（该环境运行独立安装的 PyTorch），
这样切分出去的 torch CUDA 上下文在进程退出后会被完全释放，
不会长期占用显存、也不影响 ASR 用的那个环境。

用法：
    venv-sep\Scripts\python.exe sep_worker.py <输入音频> <输出目录> <模型权重文件名>

成功时向 stdout 打印：RESULT=<人声轨路径>
"""

import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "separators"


def _resolve(produced, out_dir: Path, keyword: str):
    """audio-separator 各版本返回的可能是文件名或绝对路径，统一处理。"""
    for name in produced:
        p = Path(name)
        if not p.is_absolute():
            p = out_dir / p.name
        if keyword in p.name.lower() and p.exists():
            return p
    for name in produced:
        p = Path(name)
        if not p.is_absolute():
            p = out_dir / p.name
        if p.exists():
            return p
    return None


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: sep_worker.py <输入音频> <输出目录> <模型文件名>", file=sys.stderr)
        return 2

    src = sys.argv[1]
    out_dir = Path(sys.argv[2])
    model_filename = sys.argv[3]

    if not Path(src).exists():
        print(f"输入不存在: {src}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.WARNING)
    from audio_separator.separator import Separator

    sep = Separator(
        model_file_dir=str(MODEL_DIR),
        output_dir=str(out_dir),
        output_format="wav",
        log_level=logging.ERROR,
    )
    sep.load_model(model_filename=model_filename)
    produced = sep.separate(src)

    vocal = _resolve(produced, out_dir, "vocal")
    if vocal is None:
        print(f"分离未产出人声轨，产出列表: {produced}", file=sys.stderr)
        return 3

    print(f"RESULT={vocal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
