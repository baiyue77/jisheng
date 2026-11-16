#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
diarize_worker.py —— 说话人分割子进程

用法：
    venv-diarize\Scripts\python.exe diarize_worker.py <输入音频.wav> <输出.json>
        [--model MODEL] [--num-speakers N] [--device cuda|cpu]
        [--embeddings-out 质心.npz]

输入音频必须是 16kHz 单声道 WAV（与 ASR 流程保持一致）。
输出为 JSON 列表：[{"start": 秒, "end": 秒, "speaker": "SPEAKER_00"}, ...]

**stdout 协议**（既有调用方按行匹配前缀，别改 RESULT= 那一行的位置与格式）：
    RESULT=<输出 JSON 路径>
    EMBEDDINGS=<质心 npz 路径>        # 只在真拿到质心时**追加**一行

质心（聚类质心）说明
--------------------
`pipeline(audio, return_embeddings=True)` 返回 `(diarization, centroids)`，
`centroids` 形状 `(说话人数, 嵌入维度)`，**逐行对齐 `list(diarization.labels())`**：
pyannote 3.4 的 `speaker_diarization.py:629-644` 先按需 `np.pad` 补零到
`len(diarization.labels())`，再执行
`centroids[[inverse_mapping[label] for label in diarization.labels()]]`
—— `inverse_mapping` 与 `mapping` 互为逆，取值都落在 `range(len(labels))` 里，
所以这是**按标签对行做置换**，返回行数恒等于 `len(labels())`。
（源码里那句注释把它描述成 "re-order ... so that they match"，
但实现是**置换**而不是「取前 N 行」，这正是不能假设前缀对齐的原因。）
实测范数 2.2~2.6，**未归一化** —— 归一化由 `speaker_registry` 负责。

三种退化的**实际**归宿（都不影响分割结果本身，区别只在 sidecar）：
* `centroids is None`            —— OracleClustering 的正常路径 → 不写 sidecar、
  不打印 `EMBEDDINGS=`，`RESULT=` 照常；
* 行数 != 标签数                  —— pyannote 3.4 下到不了（上面那个重排保证了相等），
  留作防御：**一律不写**（行数与标签数不等时「第 i 行就是第 i 个标签」不成立，
  详见 `write_embeddings` 的注释）；
* 部分行为全零 / 非有限           —— 行数相等，所以**照写**；由**读端**负责逐行丢弃
  （`speaker_registry.extract_centroids` → `normalise_vector` 对全零/NaN 返回 None）。

一个**刻意保留的不对称**：本文件对非有限行的策略是「全有全无」（任一行非有限
就整份不写，因为写一份混着 NaN 的 sidecar 不如不写），而读端是逐行丢。
两者都会让坏行进不了匹配，但严格说行为不同 —— 这里如实写下来，不假装一致。

放在独立环境里运行，避免 pyannote.audio/torch 依赖污染 ASR 环境。
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

# 兼容层：pyannote 3.x 在新版 torch / torchaudio 上的两处不兼容处理
from pyannote_compat import apply_all as _apply_compat  # noqa: E402

# sidecar 里的两个数组名（调用方靠它们取值，写死一处便于对齐）
LABELS_KEY = "labels"
CENTROIDS_KEY = "centroids"


def _load_wav(path: Path) -> dict:
    """用标准库把 16bit WAV 读成 pyannote 需要的 {"waveform", "sample_rate"}。

    自己读盘可以完全不依赖 torchaudio 的 I/O API。
    waveform 形状为 (channel, time)。
    """
    import wave

    import numpy as np
    import torch

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if width != 2:
        raise RuntimeError(f"只支持 16bit PCM WAV，实际 {width * 8}bit")

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(data)
    if channels > 1:
        waveform = waveform.view(-1, channels).mean(dim=1)
    waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": rate, "uri": path.stem}


def _configure_hf_cache():
    r"""配置 HuggingFace 缓存位置与下载镜像。

    必须在 import huggingface_hub / pyannote.audio 之前调用：
    HF_HOME 与 HF_ENDPOINT 都是 import 时读取的。

    受限模型可能需要凭据，在线请求固定使用 Hugging Face 官方端点。
    """
    if not os.environ.get("HF_HOME"):
        cache = HERE / "models"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache)
            os.environ["HF_HUB_CACHE"] = str(cache / "hub")
        except OSError as exc:
            print(f"[!] 无法创建模型缓存目录 {cache}：{exc}", file=sys.stderr)

    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    print(f"[diarize] HF_HOME={os.environ.get('HF_HOME')}", file=sys.stderr)
    print(f"[diarize] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}", file=sys.stderr)


def _load_pipeline(model: str, device: str):
    """加载模型并搬到目标设备。单独一个函数，方便离线测试替换。"""
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    try:
        try:
            pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
        except TypeError:
            # 部分 pyannote / huggingface_hub 版本改名为 token=
            pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as exc:
        raise RuntimeError(f"加载 diarization 模型失败：{exc}") from exc

    if device == "cuda" and torch.cuda.is_available():
        return pipeline.to(torch.device("cuda"))
    return pipeline.to(torch.device("cpu"))


def _segments_from_diarization(diarization) -> list:
    """diarization -> [{start, end, speaker}, ...]，按 start 升序。"""
    results = []
    for speech_turn, _track, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(float(speech_turn.start), 3),
            "end": round(float(speech_turn.end), 3),
            "speaker": str(speaker),
        })
    results.sort(key=lambda x: x["start"])
    return results


def _labels_of(diarization) -> list:
    """diarization 的说话人标签，顺序与质心行一一对应。取不到就当空。"""
    try:
        labels = list(diarization.labels())
    except Exception:
        return []
    return [str(x) for x in labels]


def write_embeddings(path, labels, centroids):
    r"""把 `(labels, centroids)` 落成 npz sidecar；不该写就返回 None（**绝不抛**）。

    规则是**严格相等**：只有 `行数 == len(labels)` 才写。返回 None 的情况
    （都是「宁可不写，也不写一份走样的对应用于匹配」）：
    1. `labels` 为空（没有说话人就谈不上质心）
    2. 行数 != 标签数
    3. 矩阵为空 / 不是二维（静音输入、形状异常）
    4. 有非有限的元素（NaN / Inf），或写盘失败（磁盘满 / 被占用）

    为什么「行数多」也不放行：放行需要一个假设 —— 「前 N 行仍按标签顺序对齐」。
    这个假设**没有依据**：pyannote 3.4 在返回前做的是
    `centroids[[inverse_mapping[label] for label in labels]]`，是按标签对行做
    **置换**，不是取前缀（源码 641-644）。真出现「行数多」时多出来的是哪些行、
    原来属于谁，从外部完全看不出来。读端（`webui.load_cluster_embeddings`）对
    行数不等也是整份不信，两边同规则才不会出现「写方说合格、读方说不要」这种
    质心躺在盘上却永远进不了标注状态的静默状态。
    """
    if centroids is None:
        return None
    labels = [str(x) for x in (labels or ())]
    if not labels:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    try:
        arr = np.asarray(centroids)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    if arr.shape[0] != len(labels):
        # 逐行对齐不成立（多出来的行是置换后的残留，少的行是补位零行）
        return None
    if not bool(np.all(np.isfinite(arr))):
        return None

    dst = Path(path)
    # 与 speaker_registry._atomic_write_npz 同样的两个坑：
    # np.savez 会自己往文件名后面补 `.npz`；临时文件必须与目标同目录（同卷）。
    tmp = dst.parent / ("%s.%d.tmp.npz" % (dst.name, os.getpid()))
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(tmp), **{LABELS_KEY: np.array(labels, dtype="<U64"),
                              CENTROIDS_KEY: arr.astype(np.float32)})
        os.replace(str(tmp), str(dst))
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return None
    return dst


def _run_and_write(diarization, centroids, out_json, out_embeddings=None) -> int:
    """把一次分割结果落盘并按协议打印。返回进程退出码。

    **顺序是协议的一部分**：先写 JSON、先打印 `RESULT=`，之后才碰 sidecar。
    sidecar 那一步无论出什么事（拿不到质心 / 形状不对 / 写盘失败）都只影响
    `EMBEDDINGS=` 这一行，**不能**让分割结果跟着失败。
    """
    results = _segments_from_diarization(diarization)

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"RESULT={out_json}")

    if out_embeddings:
        emb_path = write_embeddings(out_embeddings, _labels_of(diarization),
                                    centroids)
        if emb_path is not None:
            print(f"EMBEDDINGS={emb_path}")
    return 0


def run_pipeline(pipeline, audio, kwargs, out_json, out_embeddings=None) -> int:
    """跑一次分割（**必须开 `return_embeddings=True`**）并落盘。返回退出码。

    `pipeline` 可注入，离线单测不必起 pyannote —— 但这次调用是真的，
    所以「`return_embeddings=True` 到底有没有传下去」被断言的是真实调用参数。
    """
    diarization, centroids = pipeline(audio, return_embeddings=True, **kwargs)
    return _run_and_write(diarization, centroids, out_json, out_embeddings)


def main() -> int:
    parser = argparse.ArgumentParser(description="说话人分割子进程（pyannote.audio）")
    parser.add_argument("input_wav", help="16kHz 单声道 WAV 路径")
    parser.add_argument("output_json", help="输出 JSON 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="pyannote pipeline 模型 ID 或本地路径")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="强制说话人数（留空自动估计）")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--embeddings-out", default=None,
                        help="聚类质心 sidecar（npz）路径；拿到质心才写。"
                             "不传就完全不产生这个文件（老调用方行为不变）")
    args = parser.parse_args()

    src = Path(args.input_wav)
    dst = Path(args.output_json)
    if not src.exists():
        print(f"[X] 输入音频不存在：{src}", file=sys.stderr)
        return 2

    _configure_hf_cache()
    _apply_compat()

    try:
        import torch  # noqa: F401
        from pyannote.audio import Pipeline  # noqa: F401
    except ImportError as exc:
        print(f"[X] 缺少依赖：{exc}", file=sys.stderr)
        print("    请先运行 setup_diarize.cmd 安装 venv-diarize 环境。", file=sys.stderr)
        return 3

    try:
        pipeline = _load_pipeline(args.model, args.device)
    except RuntimeError as exc:
        print(f"[X] {exc}", file=sys.stderr)
        print("    本地模型模式下不需要 token；若走在线仓库，常见原因是"
              "未接受 HuggingFace 用户协议或未设置 HF_TOKEN。", file=sys.stderr)
        return 4

    kwargs = {}
    if args.num_speakers is not None:
        # pyannote 3.x 支持 num_speakers；若不支持会回退到 min/max
        kwargs["num_speakers"] = args.num_speakers
        kwargs["min_speakers"] = args.num_speakers
        kwargs["max_speakers"] = args.num_speakers

    try:
        audio = _load_wav(src)
    except Exception as exc:
        print(f"[X] 读取音频失败：{exc}", file=sys.stderr)
        return 6

    try:
        return run_pipeline(pipeline, audio, kwargs, dst, args.embeddings_out)
    except Exception as exc:
        print(f"[X] 说话人分割失败：{exc}", file=sys.stderr)
        return 5

if __name__ == "__main__":
    sys.exit(main())
