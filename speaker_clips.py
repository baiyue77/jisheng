#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""试听片段挑选与切分。

挑选规则（对应设计文档 §7）：
  1. 候选 = 该聚类在 segments.jsonl 里的句子，时长 3~8 秒
  2. 与其它说话人重叠 <= 0.2 秒；重叠的只在候选不足时兜底使用，并打标记
  3. 打分 = 0.5*时长 + 0.3*干净度 + 0.2*响度
  4. 把整集按 limit 等分，每个时区取一条 —— 让用户听到不同场景下的声音
  5. 仍不足则放宽时长上界到 15 秒，再不足才允许用重叠片段

切分只产出 16 kHz 单声道小 WAV（约 160 KB/条）供浏览器试听；
训练集用的是另一条路径（阶段 2 的 train_export.py，32 kHz）。
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
from pathlib import Path

import jisheng as A

CLIP_MIN_S = 3.0
CLIP_TARGET_MAX_S = 8.0
CLIP_RELAXED_MAX_S = 15.0
CLIP_OVERLAP_TOL_S = 0.2
CLIPS_PER_CLUSTER = 3
PREVIEW_SOURCES = ("vocals_16k.wav", "raw16k.wav")


def overlap_seconds(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merged_seconds(intervals) -> float:
    """区间并集的总长度（重叠部分只算一次）。"""
    items = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not items:
        return 0.0
    total = 0.0
    cur_s, cur_e = items[0]
    for s, e in items[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def other_speaker_overlap(seg: dict, dia_segments: list, cluster) -> float:
    """seg 的区间里，属于其它说话人的总时长（取并集，不重复计）。"""
    s, e = float(seg["start"]), float(seg["end"])
    spans = []
    for d in dia_segments:
        if d.get("speaker") == cluster:
            continue
        ov = overlap_seconds(s, e, float(d["start"]), float(d["end"]))
        if ov > 0:
            spans.append((max(s, float(d["start"])), min(e, float(d["end"]))))
    return min(merged_seconds(spans), e - s)


def score_candidate(dur: float, overlap_ratio: float, rms_norm: float) -> float:
    return (0.5 * min(dur, CLIP_TARGET_MAX_S) / CLIP_TARGET_MAX_S
            + 0.3 * (1.0 - max(0.0, min(overlap_ratio, 1.0)))
            + 0.2 * max(0.0, min(rms_norm, 1.0)))


def _zone_of(start: float, episode_end: float, zone_count: int) -> int:
    if episode_end <= 0 or zone_count <= 1:
        return 0
    z = int(start / episode_end * zone_count)
    return max(0, min(z, zone_count - 1))


def _spread(pool, limit: int, episode_end: float, zone_count: int):
    """每个时区取一条（区内按分数），仍不足则按分数补齐。"""
    ranked = sorted(pool, key=lambda r: (-r["score"], r["start"]))
    picked, taken = [], set()
    for z in range(zone_count):
        if len(picked) >= limit:
            break
        for rec in ranked:
            if rec["_i"] in taken:
                continue
            if _zone_of(rec["start"], episode_end, zone_count) == z:
                picked.append(rec)
                taken.add(rec["_i"])
                break
    for rec in ranked:
        if len(picked) >= limit:
            break
        if rec["_i"] not in taken:
            picked.append(rec)
            taken.add(rec["_i"])
    return picked


def pick_clip_candidates(segments, dia_segments, cluster,
                         limit: int = CLIPS_PER_CLUSTER,
                         audio_stats=None,
                         episode_end=None,
                         dur_min: float = CLIP_MIN_S,
                         dur_max: float = CLIP_TARGET_MAX_S,
                         relaxed_max: float = CLIP_RELAXED_MAX_S,
                         overlap_tol: float = CLIP_OVERLAP_TOL_S,
                         zone_count=None):
    """挑出该聚类最适合试听的片段。

    audio_stats: 可选 callable(start, end) -> (rms, peak)，两者 0~1。
                 传 None 时响度项取中立值（测试与无音频场景）。
    episode_end: 用于划分时区；None 则取所有片段的最后结束时间。
    """
    if limit <= 0:
        return []
    zone_count = zone_count or max(limit, 1)
    if episode_end is None:
        episode_end = max([float(s["end"]) for s in segments] + [0.0])
    episode_end = float(episode_end)

    pool = []
    for i, s in enumerate(segments):
        if s.get("speaker") != cluster:
            continue
        dur = float(s["end"]) - float(s["start"])
        if dur < dur_min or dur > relaxed_max:
            continue
        ov = other_speaker_overlap(s, dia_segments, cluster)
        pool.append({"_i": i, "start": float(s["start"]), "end": float(s["end"]),
                     "dur": dur, "overlap_s": ov, "overlap": ov > overlap_tol,
                     "rms_norm": 0.5, "score": 0.0, "peak_dbfs": None})
    if not pool:
        return []

    if audio_stats is not None:
        kept = []
        for rec in pool:
            rms, peak = audio_stats(rec["start"], rec["end"])
            rms = max(0.0, float(rms))
            peak = max(0.0, float(peak))
            if peak > 0:
                rec["peak_dbfs"] = round(20.0 * math.log10(peak), 1)
            # peak == 0 时 peak_dbfs 保持 None —— 不是「-inf dBFS」，而是「这一段的
            # 峰值测不出来/全静音」。这不是缺陷，是刻意的语义，别改成负数。
            if peak >= 0.99:        # 削波，宁可换一条
                continue
            rec["_rms"] = rms
            kept.append(rec)
        pool = kept
        if not pool:
            return []
        loudest = max(r["_rms"] for r in pool) or 1.0
        for rec in pool:
            rec["rms_norm"] = min(rec["_rms"] / loudest, 1.0)

    for rec in pool:
        rec["score"] = score_candidate(rec["dur"],
                                       rec["overlap_s"] / rec["dur"] if rec["dur"] else 0.0,
                                       rec["rms_norm"])

    chosen, taken = [], set()
    for max_dur, allow_overlap in ((dur_max, False),
                                   (relaxed_max, False),
                                   (relaxed_max, True)):
        if len(chosen) >= limit:
            break
        sub = [r for r in pool
               if r["dur"] <= max_dur
               and (allow_overlap or not r["overlap"])
               and r["_i"] not in taken]
        if not sub:
            continue
        for rec in _spread(sub, limit - len(chosen), episode_end, zone_count):
            taken.add(rec["_i"])
            chosen.append(rec)

    chosen.sort(key=lambda r: r["start"])
    out = []
    for n, rec in enumerate(chosen, 1):
        # dur 单独取整、start/end 保留原值，所以 dur 与 (end-start) 最多差 0.005 秒。
        # 面板只拿 dur 显示「(x.xs)」、拿 start 画时间轴，这点误差无影响，不强行对齐。
        out.append({"n": n, "start": rec["start"], "end": rec["end"],
                    "dur": round(rec["dur"], 2),
                    "overlap": bool(rec["overlap"]),
                    "overlap_s": round(rec["overlap_s"], 2),
                    "score": round(rec["score"], 4),
                    "peak_dbfs": rec["peak_dbfs"]})
    return out


# ---------------------------------------------------------------------------
# 片段缓存路径
# ---------------------------------------------------------------------------

def clip_cache_dir(work) -> Path:
    return Path(work) / "clips"


def _safe_cluster(cluster) -> str:
    # 注意这个替换会把 "SPEAKER.0" 与 "SPEAKER 0" 都变成 "SPEAKER_0"（同名）。
    # 这不是缺陷：pyannote 只产出 SPEAKER_xx 形式，没有触发路径，所以不做区分。
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(cluster or "")).strip("_")
    return s or "unknown"


def clip_path(work, cluster, n: int) -> Path:
    return clip_cache_dir(work) / f"{_safe_cluster(cluster)}_{int(n)}.wav"


def preview_source(work):
    """试听音频来源：优先分离后的人声轨（BGM/OP/ED 已被去掉）。"""
    work = Path(work)
    for name in PREVIEW_SOURCES:
        p = work / name
        if p.is_file():
            return p
    return None


def cut_clip(src, start: float, dur: float, dest) -> None:
    """切一小段 16 kHz 单声道 WAV（试听专用）。

    先写到同目录的临时名再 os.replace 成最终路径：并发 GET /api/clip 只会看到
    「旧的完整文件」或「新的完整文件」，不会读到写了一半的 WAV。
    临时名**必须保持 .wav 结尾**，否则 ffmpeg 按扩展名选不到 muxer 会直接失败。

    必须带 `-nostdin`：ffmpeg 默认会去读标准输入，被当成后台进程跑时可能因为
    等 stdin 而永远挂住（与 jisheng.extract_audio 里同一个坑）。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.{threading.get_ident()}.tmp.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-ss", f"{float(start):.3f}", "-t", f"{float(dur):.3f}",
             "-i", str(src),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             str(tmp)],
            check=True, capture_output=True,
        )
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _clip_meta(dest) -> Path:
    """片段签名文件：记录这条片段是对「哪段、哪个源」切的。"""
    return Path(dest).with_suffix(".json")


# 每个片段目标一把锁，只为把「临时文件 → os.replace」这一段串行。
# 光靠原子替换还不够：Windows 上两个线程同时 replace 同一个目标，
# 输的那个会抛 PermissionError（WinError 13/32）。并发 GET /api/speakers
# 会对同一批片段各跑一次 ensure_clips，所以这一步必须互斥。
#
# 与 webui 的 _LABEL_LOCKS 一样，这里刻意不做「清理不再使用的锁」：
# 把一把正被持有（或即将被持有）的锁从字典里删掉，另一个线程就会给同一
# 目标新建第二把锁，两个线程同时进临界区 —— 互斥失效比多占几个槽位严重。
_DEST_LOCKS = {}
_DEST_LOCKS_GUARD = threading.Lock()


def _dest_lock(path):
    key = os.path.normcase(os.path.abspath(str(Path(path))))
    with _DEST_LOCKS_GUARD:
        lock = _DEST_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DEST_LOCKS[key] = lock
        return lock


def _clip_signature(cand, src) -> dict:
    st = Path(src).stat()
    return {"start": round(float(cand["start"]), 3),
            "dur": round(float(cand["dur"]), 3),
            "src": Path(src).name,
            "size": int(st.st_size),
            "mtime": int(st.st_mtime)}


def ensure_clips(work, cluster, candidates, src) -> list:
    """按需切片段。签名对得上才复用，否则重切。

    签名里带上了候选的时间范围与源文件的 (名字, 大小, mtime)，因为：
      * 断点续跑后重挑候选，n 相同但时间段不同 —— 只靠文件名会放到旧音频
      * preview_source 可能从 raw16k.wav 换成 vocals_16k.wav
    元数据是切成功后最后写的，所以切到一半被打断的片段下次会被重切。

    src 不存在就直接返回空列表：没音频可切不是错误，但让 _clip_signature
    抛 FileNotFoundError 会把一次普通的「源文件已被清掉」变成 500。
    """
    src = Path(src)
    if not src.is_file():
        return []
    out = []
    for cand in candidates:
        dest = clip_path(work, cluster, cand["n"])
        meta = _clip_meta(dest)
        sig = _clip_signature(cand, src)
        with _dest_lock(dest):
            fresh = False
            if dest.is_file() and meta.is_file():
                try:
                    fresh = json.loads(meta.read_text(encoding="utf-8")) == sig
                except Exception:
                    fresh = False
            if not fresh:
                cut_clip(src, cand["start"], cand["dur"], dest)
                # 签名也原子写：否则并发读会读到写了一半的 JSON
                A.atomic_write_text(meta, json.dumps(sig, ensure_ascii=False))
        rec = dict(cand)
        rec["path"] = str(dest)
        out.append(rec)
    return out
