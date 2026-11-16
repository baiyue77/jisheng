#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
zh_label.py —— 中文音频批量识别 + 术语表校正，直接产出 GPT-SoVITS 标注 list

两个后端
--------
  qwen    : Qwen3-ASR-1.7B（中文最准，默认）——自动调用 venv-diarize 里的 worker
  whisper : 本地 faster-whisper（快，可用热词 prompt）

用法
----
    python zh_label.py "samples/character-a" --speaker 角色甲
    python zh_label.py <目录> --backend whisper --speaker 角色甲

输出
----
    <目录名>.list   <绝对路径>|<说话人>|ZH|<文本>     ← GPT-SoVITS 可直接导入
    <目录名>.tsv    附时长/耗时，便于抽查
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jisheng as A  # noqa: E402

WHISPER_MODEL = "turbo"
QWEN_PY = HERE / "venv-diarize" / "Scripts" / "python.exe"
QWEN_WORKER = HERE / "qwen_worker.py"
QWEN_MODEL = HERE / "models" / "Qwen3-ASR-1.7B-hf"

# 纯笑声/语气词：对 TTS 训练无价值，默认丢弃
LAUGH = re.compile(r"^[呵哈嘿嘻呋嗯额啊哦噢诶欸呀呃唔\s，。、？！…~～]+$")


def read_wav_meta(path: Path):
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def load16k(path: Path) -> np.ndarray:
    """读任意采样率 wav -> 16kHz 单声道 float32（不依赖 ffmpeg）。"""
    sr, ch, _ = read_wav_meta(path)
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        if sr % 16000 == 0:
            f = sr // 16000
            x = x[: len(x) // f * f].reshape(-1, f).mean(axis=1)
        else:
            n = int(len(x) * 16000 / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x


def build_prompt(pairs, limit=30) -> str:
    """从术语表右列提取热词。"""
    seen, words = set(), []
    for _, right in pairs:
        w = right.strip()
        if len(w) < 2 or w in seen or " " in w:
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= limit:
            break
    return "词汇表：" + "、".join(words) + "。" if words else ""


def clean(text: str, drop_laugh: bool) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 去掉 Qwen 偶尔挂在开头的提示词回声
    t = re.sub(r"^词汇表[：:].*?。", "", t).strip()
    if drop_laugh and LAUGH.match(t) and len(t) >= 2:
        return ""
    return t


def run_whisper(files, pairs, prompt, args):
    A.register_cuda_dlls()
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(args.model or WHISPER_MODEL, device="cuda", compute_type="float16")
    print(f"[whisper] 加载 {time.time() - t0:.1f}s", flush=True)
    out = []
    for i, f in enumerate(files, 1):
        p = Path(f)
        audio = load16k(p)
        dur = len(audio) / 16000.0
        t1 = time.time()
        seg = A.transcribe_chunk(model, audio, {"start": 0, "end": len(audio)},
                                 5, pairs, args.language, prompt or None)
        text = clean(seg["text"] if seg else "", args.drop_laugh)
        out.append((p.name, dur, A.apply_glossary(text, pairs), time.time() - t1))
        print(f"[{i}/{len(files)}] {dur:5.1f}s  {text}", flush=True)
    return out


def run_qwen(files, pairs, prompt, args):
    if not QWEN_PY.exists():
        raise SystemExit(f"[X] 找不到运行环境：{QWEN_PY}")
    if not (QWEN_MODEL / "model.safetensors").exists():
        raise SystemExit(f"[X] 找不到 Qwen3-ASR 模型：{QWEN_MODEL}")
    tmp_jsonl = Path(tempfile.mkdtemp(prefix="qwen_")) / "out.jsonl"
    cmd = [str(QWEN_PY), str(QWEN_WORKER), str(Path(files[0]).parent),
           "--out", str(tmp_jsonl), "--model", str(QWEN_MODEL),
           "--batch", str(args.batch)]
    if prompt:
        cmd += ["--prompt", prompt]
    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"[X] Qwen worker 退出码 {proc.returncode}")
    rows = []
    with open(tmp_jsonl, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append((d["file"], d["sec"], d.get("text", ""), None))
    dt = time.time() - t0
    avg = dt / max(len(rows), 1)
    return [(n, s, A.apply_glossary(clean(t, args.drop_laugh), pairs), avg)
            for n, s, t, _ in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="切片音频目录")
    ap.add_argument("--backend", default="qwen", choices=["qwen", "whisper"])
    ap.add_argument("--speaker", default="", help="说话人名")
    ap.add_argument("--out", default=None, help="输出 .list 路径")
    ap.add_argument("--glossary", help="可选术语表文件")
    ap.add_argument("--language", default="zh", help="Whisper 后端使用的语言代码")
    ap.add_argument("--model", default=None, help="Whisper 模型名称或本地目录")
    ap.add_argument("--batch", type=int, default=4, help="qwen 批大小")
    ap.add_argument("--prompt", default=None, help="覆盖自动热词")
    ap.add_argument("--min-sec", type=float, default=0.4)
    ap.add_argument("--keep-laugh", action="store_true", help="保留纯笑声/语气词行")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(glob.glob(str(src / "*.wav")))
    if not files:
        print(f"[X] 目录里没有 wav：{src}", file=sys.stderr)
        return 2

    keep = []
    for f in files:
        if read_wav_meta(Path(f))[2] / read_wav_meta(Path(f))[0] >= args.min_sec:
            keep.append(f)
    skipped = len(files) - len(keep)

    pairs = []
    gp = Path(args.glossary) if args.glossary else None
    if gp and gp.is_file():
        pairs = A.load_glossary(gp)
        print(f"术语表：{gp.name} · {len(pairs)} 条")
    prompt = args.prompt if args.prompt is not None else (build_prompt(pairs) if pairs else "")
    if prompt:
        print(f"热词：{prompt[:64]}{'…' if len(prompt) > 64 else ''}")

    t0 = time.time()
    rows = (run_whisper if args.backend == "whisper" else run_qwen)(keep, pairs, prompt, args)
    rows = [r for r in rows if r[2]]
    if not rows:
        print("[X] 没有有效结果", file=sys.stderr)
        return 3

    out = Path(args.out) if args.out else src / (src.name + ".list")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for name, _, text, _ in rows:
            fh.write(f"{src / name}|{args.speaker}|ZH|{text}\n")
    tsv = out.with_suffix(".tsv")
    with open(tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("file\tsec\ttext\n")
        for name, sec, text, _ in rows:
            fh.write(f"{name}\t{sec:.2f}\t{text}\n")

    total = sum(r[1] for r in rows)
    el = time.time() - t0
    print("\n" + "=" * 56)
    print(f"  {args.backend} · 写出 {len(rows)} 条 -> {out}")
    print(f"  跳过 {skipped} 条（<{args.min_sec}s）/ 丢弃空行与笑声 "
          f"{len(keep) - len(rows)} 条")
    print(f"  音频 {A.fmt_hms(total)} · 耗时 {A.fmt_hms(el)} · {total / el:.1f}x")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
