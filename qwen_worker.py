#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
qwen_worker.py —— Qwen3-ASR 批量识别 worker（供 zh_label.py 调用）

必须在装有 torch + transformers 的环境里跑（本项目用 venv-diarize）。
输出 jsonl：{"file": "xxx.wav", "sec": 3.8, "text": "..."}

单独用法（调试）：
    venv-diarize\Scripts\python.exe qwen_worker.py <音频目录> --out out.jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3-ASR-1.7B-hf")


def to16k(src: str, dst: str) -> None:
    """wav -> 16kHz 单声道，不依赖 ffmpeg。"""
    with wave.open(src) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != 16000:
        if sr % 16000 == 0:
            f = sr // 16000
            x = x[: len(x) // f * f].reshape(-1, f).mean(axis=1).astype(np.int16)
        else:
            n = int(len(x) * 16000 / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)
    with wave.open(dst, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="", help="词汇表提示（热词）")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--tmp", default=os.path.join(HERE, "tmp", "qwen16k"))
    ap.add_argument('--manifest', help='带 file/index/start/end 的短音频 JSON 清单')
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM as ModelCls
    except ImportError:
        from transformers import Qwen3ASRForConditionalGeneration as ModelCls

    os.makedirs(args.tmp, exist_ok=True)
    manifest = None
    if args.manifest:
        with open(args.manifest, encoding='utf-8') as source:
            manifest = json.load(source)
        if not isinstance(manifest, list):
            raise ValueError('manifest 必须是数组')
        files = [row['file'] for row in manifest]
    else:
        files = sorted(glob.glob(os.path.join(args.src, "*.wav")))
    if not files:
        print(f"[X] 没有 wav：{args.src}", file=sys.stderr)
        return 2

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = ModelCls.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"[qwen] 加载 {time.time() - t0:.1f}s · {len(files)} 条", flush=True)

    out = open(args.out, "w", encoding="utf-8")
    done, t_start = 0, time.time()
    for i in range(0, len(files), args.batch):
        batch = files[i:i + args.batch]
        paths, durs = [], []
        for f in batch:
            st = os.stat(f)
            identity = json.dumps([os.path.normcase(os.path.realpath(f)), st.st_size,
                                   st.st_mtime_ns], ensure_ascii=False)
            key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            dst = os.path.join(args.tmp, key + ".wav")
            if not os.path.exists(dst):
                fd, pending = tempfile.mkstemp(suffix=".wav", dir=args.tmp)
                os.close(fd)
                try:
                    to16k(f, pending)
                    os.replace(pending, dst)
                finally:
                    if os.path.exists(pending):
                        os.unlink(pending)
            paths.append(dst)
            with wave.open(dst) as w:
                durs.append(w.getnframes() / 16000.0)

        req = {"audio": paths}
        if args.prompt:
            req["prompt"] = [args.prompt] * len(paths)
        if args.language != 'auto':
            req["language"] = [args.language] * len(paths)
        inputs = processor.apply_transcription_request(**req).to(model.device, model.dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=512)
        texts = processor.decode(ids[:, inputs["input_ids"].shape[-1]:],
                                 return_format="transcription_only")
        for offset, (f, d, t) in enumerate(zip(batch, durs, texts)):
            done += 1
            row = {"file": os.path.basename(f), "sec": round(d, 2), "text": (t or "").strip()}
            if manifest is not None:
                row.update({key: manifest[i + offset][key] for key in ('index', 'start', 'end')})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[qwen {done}/{len(files)}] {d:5.1f}s  {(t or '').strip()}", flush=True)
        out.flush()
        os.fsync(out.fileno())
    out.close()

    el = time.time() - t_start
    total = sum(1 for _ in files)
    print(f"[qwen] 完成 {total} 条 · {el / 60:.1f} 分钟", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
