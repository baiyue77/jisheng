#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""说话人标注状态。

一个任务的标注状态存在 work\<片名>\speakers.json：
  clusters            聚类 -> {name}；两个聚类写同一个 name 就是「合并」
  segment_overrides   segments.jsonl 的行号 -> 说话人键；这就是「单句改判」
  work_name           作品名，供阶段 2 的 output\说话人\<作品名>\ 使用

名字解析优先级：单句改判 > 该句的聚类 > 聚类的名字 > 通用 [说话人N]。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import jisheng as A

LABELS_FILE = "speakers.json"

_STR_FIELDS = ("media", "job", "work_name", "source_kind")

# WebUI 的 HTTP 处理是多线程的。写侧：两次并发保存会在同一个目标文件上竞争替换，
# 即使临时文件名每次调用都唯一，Windows 上并发 replace 同一目标仍会抛
# PermissionError（WinError 32），所以写临时文件与替换必须串行化。
# 读侧：一个正在读的句柄会让写线程的 os.replace 在 Windows 上报 WinError 5
# （PermissionError），所以读也必须持同一把锁 —— 读写共用 _SAVE_LOCK。
#
# 用 RLock 而不是 Lock：锁图只有「_label_lock(work) → _SAVE_LOCK」一个方向，
# 不存在 A→B→A 的环，所以**不是**为了防环。可重入是为「同一线程嵌套获取」预留的：
# `_wipe_checkpoint` 持 `_label_lock` 时会调 `_wipe_annotation`，后者也要拿
# `_label_lock`（见 webui.py 里那对函数），普通 Lock 在那里当场自死锁。
# 本模块的 `_SAVE_LOCK` 目前没有这种嵌套调用点，用 RLock 是为了让将来
# 「持锁的辅助函数里再调 save_labels」这一改动不会变成死锁事故。
# （注意：`_wipe_annotation` 只 unlink + rmtree，**从不**获取 `_SAVE_LOCK`，
#  别把它当成 `_SAVE_LOCK` 需要可重入的理由。）
_SAVE_LOCK = threading.RLock()


def _as_str(value) -> str:
    return value if isinstance(value, str) else ""


def labels_path(work) -> Path:
    return Path(work) / LABELS_FILE


def new_labels(job_id: str = "", media: str = "",
               source_kind: str = "separated") -> dict:
    return {"version": 1, "job": job_id or "", "media": media or "",
            "work_name": "", "source_kind": source_kind or "separated",
            "clusters": {}, "segment_overrides": {}, "updated_at": ""}


def load_labels(work) -> dict:
    """读标注状态。文件缺失或损坏都退回空状态，绝不抛异常。

    这个文件是给人手改的，所以每个字段都要过一遍类型检查 —— 有一个字段写错类型
    不能让整个面板变成 500。文件名带 UTF-8 BOM 也要能读（Windows 编辑器会加）。

    这里要持写锁：HTTP 处理是多线程的，一个正在读的句柄会让写线程的
    os.replace 在 Windows 上报 WinError 5（PermissionError）。
    """
    with _SAVE_LOCK:
        p = labels_path(work)
        if not p.is_file():
            return new_labels()
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return new_labels()
        if not isinstance(data, dict) or not isinstance(data.get("clusters"), dict):
            return new_labels()
        for key in _STR_FIELDS:
            data[key] = _as_str(data.get(key))
        if not data["source_kind"]:
            data["source_kind"] = "separated"
        if not isinstance(data.get("segment_overrides"), dict):
            data["segment_overrides"] = {}
        # 手改过的值类型可能是任何东西。非字符串一律整条丢掉 —— 绝不能 str() 一下
        # 了事：["SPEAKER_01"] 会变成 "['SPEAKER_01']" 这种幽灵角色。
        overrides = {}
        for key, val in data["segment_overrides"].items():
            if not isinstance(val, str):
                continue
            val = val.strip()
            if val:
                overrides[str(key)] = val
        data["segment_overrides"] = overrides
        clusters = {}
        for cluster, rec in data["clusters"].items():
            if not isinstance(rec, dict):
                rec = {"name": rec}          # 手写成 "SPEAKER_00": "艾丽丝" 也认
            rec["name"] = _as_str(rec.get("name")).strip()
            clusters[str(cluster)] = rec     # 保留 name 以外的字段（阶段 2 会加声纹键）
        data["clusters"] = clusters
        return data


def save_labels(work, data: dict) -> None:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = labels_path(work).with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    with _SAVE_LOCK:  # 串行化 write+replace，见 _SAVE_LOCK 的注释
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(labels_path(work))


def ensure_clusters(data: dict, clusters) -> dict:
    store = data.setdefault("clusters", {})
    for c in clusters or ():
        if c:
            store.setdefault(c, {"name": ""})
    return data


def bootstrap(work, job_id: str = "", media: str = "", clusters=(),
              source_kind: str = "separated") -> dict:
    """识别流程结束时落一份骨架，让面板一打开就有聚类列表。"""
    data = load_labels(work)
    if media:
        data["media"] = str(media)
    if job_id:
        data["job"] = job_id
    if source_kind:
        data["source_kind"] = source_kind
    ensure_clusters(data, clusters)
    save_labels(work, data)
    return data


def set_names(data: dict, mapping: dict) -> None:
    """{"SPEAKER_00": "艾丽丝"}；空串或纯空白 = 清空名字。"""
    for cluster, name in (mapping or {}).items():
        if not cluster:
            continue
        rec = data.setdefault("clusters", {}).setdefault(cluster, {"name": ""})
        rec["name"] = (name or "").strip()


def cluster_name(data: dict, cluster):
    rec = (data.get("clusters") or {}).get(cluster)
    if not isinstance(rec, dict):
        return None
    name = (rec.get("name") or "").strip()
    return name or None


def set_work_name(data: dict, name: str) -> None:
    data["work_name"] = (name or "").strip()


def get_work_name(data: dict) -> str:
    return (data.get("work_name") or "").strip()


def set_override(data: dict, index, to: str) -> None:
    data.setdefault("segment_overrides", {})[str(int(index))] = str(to)


def effective_speaker(data: dict, index: int, seg: dict):
    """返回喂给 prefix_with_speaker 的值：None / "?" / 角色名 / "SPEAKER_xx"。"""
    cluster = (data.get("segment_overrides") or {}).get(str(index), seg.get("speaker"))
    if not cluster:
        return None
    if cluster == "?":
        return "?"
    return cluster_name(data, cluster) or cluster


def apply_labels(data: dict, segments) -> list:
    out = []
    for i, seg in enumerate(segments):
        rec = dict(seg)
        rec["speaker"] = effective_speaker(data, i, seg)
        out.append(rec)
    return out


def rewrite_outputs(data: dict, segments, srt_path, txt_path) -> int:
    """按当前标注重写 SRT 与 TXT，返回字幕条数。

    两者都走 A.atomic_write_text：写失败时用户手上那份好的字幕一字未动。
    """
    labeled = apply_labels(data, segments)
    n = A.write_srt(labeled, Path(srt_path))
    A.atomic_write_text(
        Path(txt_path),
        "\n".join(A.prefix_with_speaker(s["text"].strip(), s.get("speaker"))
                  for s in labeled if s["text"].strip()))
    return n
