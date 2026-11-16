#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""全局角色库：`speakers_registry.json`（人可读元数据）+ `embeddings.npz`（向量）。

为什么向量不写 JSON：pyannote 的嵌入是 float32，写进 JSON 要过一次十进制文本，
读回来就不再逐位相同 —— 而匹配靠的就是这 256 维的余弦。
所以向量单独进 npz，JSON 里只放人眼要看的东西。

npz 键（设计 §6.2，全部 L2 归一化 float32）：
  page__<作品>__<角色>__centroid   该角色的 70/30 滚动质心（**当前空间**那批）
  page__<作品>__<角色>__seg000…    每次命名存入的聚类质心（保留多条更稳）
  page__<作品>__<角色>__centroid@<空间>  同上，但属于**另一个嵌入模型空间**（停放）
  page__<作品>__<角色>__seg000…@<空间>   同上
  job__<jobid>__<cluster>__centroid 本集聚类质心（审计用）

**嵌入模型空间**（Task 7 审查 P1 的修复）：换过嵌入模型之后，旧模型算的向量与
新的根本不是一个空间，混在一起算余弦、混在一起做 70/30 加权都是错的。所以库按
「空间」分开存，**一条向量只属于一个空间**：

* 每个角色记着它**当前那批键**（`page__<作品>__<角色>__…`，没有 `@` 后缀的那批）
  是哪个模型算的（`works[W].pages[P]["space"]`）；
* 别的空间的行**不删**，而是加 `@<空间 id>` 后缀**停放**在同一份 npz 里 ——
  换回那个模型时它们照样参与匹配（`_row_participates`），换走了就不参与；
* 空间 id 是 `space_id(模型标识)`（12 位摘要）：模型标识可能带 `__`（本模块的
  分隔符）也可能几百字符长，不能直接进键；JSON 的 `spaces` 里存 id → 模型标识，
  只为人眼排障，**正确性不依赖它**；
* `n_enroll`/`total_s` 跟着向量走：它们描述「这个空间里有多少证据」，换空间时
  存进 `pages[P]["parked"][<空间>]`，换回来再取出来（跨空间的次数不混算）。
  只有 `episodes` 是审计流水（Task 5 计划书第 4 条要求留档），不跟着搬。

绝不为「换了个模型」删用户数据；也绝不把两个空间的行混着比或混着加权。

已实测的两条硬事实（Lead 在本地 pyannote 3.4 源码逐行核实 + 真音频探针）：
1. `pipeline(audio, return_embeddings=True)` 返回 `(diarization, centroids)`，
   `centroids` 形状 `(说话人数, 256)`、**逐行对齐 `list(diarization.labels())`**；
   **范数实测 2.2~2.6，未归一化** —— 所以本模块一律先 L2 归一化再存/再比。
2. `speaker_diarization.py:634-636` 在 labels 多于质心行数时用 **全零行** 补齐
   —— 那种行不是真质心，必须丢掉（`extract_centroids` 负责）。

阈值：设计原写 auto>=0.55，但实测里有一对**被 pyannote 判为不同说话人**的组合余弦
高达 **0.5194**，离 0.55 只差 0.03。所以阈值可配置（写进 JSON）、默认取保守值
`auto=0.65` / `suggest=0.45`，并提供 `health_check()` 报告
「同角色最低分 / 不同角色最高分」两个数 —— 只有这两个数之间有间隔，阈值才有依据。

本模块**纯逻辑 + 文件读写**：不依赖 torch / pyannote / ffmpeg，只用 numpy。
读写一律容错（文件损坏/手改错类型 → 退回空库），写一律原子（临时文件 + os.replace）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

REGISTRY_FILE = "speakers_registry.json"
EMBEDDINGS_FILE = "embeddings.npz"

VERSION = 1
DEFAULT_EMBEDDING_MODEL = ""
DEFAULT_DIM = 256
DEFAULT_AUTO = 0.65
DEFAULT_SUGGEST = 0.45
DEFAULT_THRESHOLDS = {"auto": DEFAULT_AUTO, "suggest": DEFAULT_SUGGEST}

# seg 键循环 0..MAX_SEGMENTS-1：每次命名存一条，攒够就从头覆盖最老的
MAX_SEGMENTS = 30

# 停放键的空间后缀分隔符：`page__W__P__seg003@<空间 id>`。
# 用 `@` 而不是再插一段 `__`：`_segment_key_index` 要求键恰好 4 段且尾段是
# `segNNN`，加后缀后自然落选（停放键永远不参与匹配），而键段数不变，
# 角色名/作品名里带 `__` 时的行为与改造前完全一样。
_SPACE_MARK = "@"

# 范数小于这个值的向量视为坏数据（零向量 / pyannote 补位零行），读回来直接丢
_MIN_USABLE_NORM = 1e-6

# 库默认落在这里：`<项目>\speakers_registry\`。
#
# **故意不放在 `output\说话人\` 里**：那是训练集的**导出根**，导出时会按作品名
# `_rebuild_work_dir` 重建目录。库是**状态**而不是产物 —— 作品名万一撞上库目录名
# （或将来有人「清理重导」整棵导出根），库会被一起删掉。
# 放在与 `work\`（检查点）同级更合适：都是应用状态，不是交付物。
# 调用方仍可显式传 root，或用 set_default_root() 改。
_STORAGE_DIRNAME = "speakers_registry"
_default_root = None

# WebUI 是多线程的：并发写同一目标时，Windows 上并发 os.replace 会抛
# PermissionError(WinError 32)，而正在读的句柄会让 replace 抛 WinError 5。
# 读写共用一把锁（沿用 speaker_labels._SAVE_LOCK 的理由）。
_SAVE_LOCK = threading.RLock()


# --------------------------------------------------------------------------
# 路径与基础类型
# --------------------------------------------------------------------------

def project_dir() -> Path:
    return Path(__file__).resolve().parent


def default_root() -> Path:
    r"""库根目录；`<项目>\speakers_registry\`。

    **不在** `output\说话人\`（导出根）里面 —— 理由见 `_STORAGE_DIRNAME` 的注释。
    """
    if _default_root is not None:
        return Path(_default_root)
    return project_dir() / _STORAGE_DIRNAME


def set_default_root(path) -> None:
    """改默认库根目录（None 恢复成项目内默认值）。"""
    global _default_root
    _default_root = None if path in (None, "") else str(path)


def _resolve_root(root) -> Path:
    if root in (None, ""):
        return default_root()
    return Path(root)


def registry_path(root=None) -> Path:
    return _resolve_root(root) / REGISTRY_FILE


def vectors_path(root=None) -> Path:
    return _resolve_root(root) / EMBEDDINGS_FILE


def _as_str(value) -> str:
    return value if isinstance(value, str) else ""


def _as_num(value, default=0.0) -> float:
    # bool 是 int 的子类，但 True 当分数是手改事故，不算数字
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(default)
    out = float(value)
    return out if math.isfinite(out) else float(default)


def _as_int(value, default=0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return int(default)
    out = float(value)
    return int(out) if math.isfinite(out) else int(default)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """临时文件 + os.replace（同目录，保证同卷）。失败时清掉临时文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / ("%s.%d.%d.tmp" % (path.name, os.getpid(),
                                           threading.get_ident()))
    with _SAVE_LOCK:
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(str(tmp), str(path))
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    """np.savez 会**自己往文件名后面加 .npz**（`savez("x.tmp")` 写的是 `x.tmp.npz`），
    所以临时名要连 `.npz` 一起给，改名时也用它 —— 否则 os.replace 会找不到源文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = "%s.%d.%d.tmp" % (path.name, os.getpid(), threading.get_ident())
    tmp = path.parent / base
    produced = path.parent / (base + ".npz")
    with _SAVE_LOCK:
        try:
            np.savez(str(tmp), **arrays)
            os.replace(str(produced), str(path))
        except Exception:
            for leftover in (tmp, produced):
                try:
                    if leftover.exists():
                        leftover.unlink()
                except Exception:
                    pass
            raise


def _drop_file(path: Path) -> None:
    try:
        Path(path).unlink()
    except Exception:
        pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------
# 向量基础运算
# --------------------------------------------------------------------------

def _vec_or_none(value):
    """转成 float64 一维有限数组；不是有限一维数组就返回 None。绝不抛。"""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def normalise_vector(value):
    """L2 归一化成 float32；全零 / NaN / Inf / 非一维 → None。绝不抛。"""
    arr = _vec_or_none(value)
    if arr is None:
        return None
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return (arr / norm).astype(np.float32)


def cosine(a, b):
    """余弦相似度；维度不同 / 含非有限值 / 有零向量 → None。绝不抛。"""
    va = _vec_or_none(a)
    vb = _vec_or_none(b)
    if va is None or vb is None or va.shape != vb.shape:
        return None
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0.0 or nb <= 0.0:
        return None
    out = float(np.dot(va, vb) / (na * nb))
    if not math.isfinite(out):
        return None
    return max(-1.0, min(1.0, out))


def top_mean(query, rows, k=3):
    """与该角色已存的每条向量算余弦，取 **top-k 平均**。

    比单次成对余弦稳：一条坏样本（换场景/串了 BGM）只压低一个分量，
    取前 k 条就把它挡在外面。一条可用向量都没有 → None。
    """
    if _vec_or_none(query) is None:
        return None
    sims = []
    for row in rows or ():
        c = cosine(query, row)
        if c is not None:
            sims.append(c)
    if not sims:
        return None
    sims.sort(reverse=True)
    take = sims[:max(1, int(k))]
    return float(sum(take) / len(take))


def extract_centroids(centroids, labels=None) -> dict:
    """把 pyannote 的 `(N, 256)` 质心矩阵变成 {标签: 单位向量}。

    三种已核实的情况都在这里处理（都退化成「该说话人纯手动命名」，不报错）：
    - `centroids is None`            → 空（OracleClustering 的正常路径）
    - 行数 < 标签数                   → pyannote 补过全零行，丢掉
    - 静音输入                       → `(0, 256)`；全零行同样丢掉
    - NaN / Inf 行                    → 丢掉
    另外不检查形状是否恰好是 256 维：换模型时维度会变，写死 256 会变成硬编码。
    """
    if centroids is None or labels is None:
        return {}
    try:
        arr = np.asarray(centroids, dtype=np.float64)
    except Exception:
        return {}
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return {}
    labels = list(labels)
    if not labels:
        return {}
    out = {}
    for i, label in enumerate(labels):
        if i >= arr.shape[0]:
            break                       # 标签多于行数：多出来的没有质心
        name = _as_str(label).strip()
        if not name:
            continue
        unit = normalise_vector(arr[i])  # 全零 / NaN 行在这里被丢掉
        if unit is not None:
            out[name] = unit
    return out


# --------------------------------------------------------------------------
# npz 键
# --------------------------------------------------------------------------

def page_centroid_key(work, page) -> str:
    return "page__%s__%s__centroid" % (work, page)


def page_segment_key(work, page, index) -> str:
    return "page__%s__%s__seg%03d" % (work, page, max(0, int(index)))


def job_centroid_key(job, cluster) -> str:
    return "job__%s__%s__centroid" % (job, cluster)


def _segment_key_index(key: str):
    """`page__W__P__seg003` → ('W', 'P', 3)；不是 seg 键就返回 None。

    停放键（`page__W__P__seg003@<空间>`）返回 None：尾段长度不再是 6，
    于是停放的行自然不参与匹配/统计 —— 它们只在 `_parked_segment_of_key`
    里被认出来。
    """
    if not isinstance(key, str) or not key.startswith("page__"):
        return None
    parts = key.split("__")
    if len(parts) != 4:
        return None
    tail = parts[3]
    if not tail.startswith("seg") or len(tail) != 6 or not tail[3:].isdigit():
        return None
    return parts[1], parts[2], int(tail[3:])


def _page_centroid_of_key(key: str):
    """`page__W__P__centroid` → ('W', 'P')；否则 None（含停放键）。"""
    if not isinstance(key, str) or not key.startswith("page__"):
        return None
    parts = key.split("__")
    if len(parts) != 4 or parts[3] != "centroid":
        return None
    return parts[1], parts[2]


# --------------------------------------------------------------------------
# 嵌入模型空间：一条向量只属于一个空间
# --------------------------------------------------------------------------

def space_id(model) -> str:
    """嵌入模型标识 → **空间 id**（12 位十六进制摘要）；空标识 → ""（通配）。

    为什么不把模型标识直接拼进 npz 键：标识是环境/用户给的任意字符串，可能含
    `__`（本模块的分隔符）也可能几百字符长 —— 拼进去既会破坏键的解析，也会让
    键无限变长。摘要定长且无歧义；id → 标识的表存在 JSON 的 `spaces` 里
    （只为人眼排障，正确性不依赖它）。
    """
    tag = _as_str(model).strip()
    if not tag:
        return ""
    return hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]


def page_parked_centroid_key(work, page, sid) -> str:
    """停放的质心键（`sid` 为另一个空间）：保留在盘上，但绝不参与匹配。"""
    return "page__%s__%s__centroid%s%s" % (work, page, _SPACE_MARK, sid)


def page_parked_segment_key(work, page, sid, index) -> str:
    """停放的 seg 键（见 `page_parked_centroid_key`）。"""
    return "page__%s__%s__seg%03d%s%s" % (work, page, max(0, int(index)),
                                           _SPACE_MARK, sid)


def _split_space_tail(key):
    """`page__W__P__<头>@<空间>` → (W, P, 头, 空间)；不是这个形状 → None。"""
    if not isinstance(key, str) or not key.startswith("page__"):
        return None
    parts = key.split("__")
    if len(parts) != 4:
        return None
    head, sep, sid = parts[3].partition(_SPACE_MARK)
    if not sep or not head or not sid:
        return None
    return parts[1], parts[2], head, sid


def _parked_centroid_of_key(key: str):
    """`page__W__P__centroid@<空间>` → ('W', 'P', '<空间>')；否则 None。"""
    parts = _split_space_tail(key)
    if parts is None or parts[2] != "centroid":
        return None
    return parts[0], parts[1], parts[3]


def _parked_segment_of_key(key: str):
    """`page__W__P__seg003@<空间>` → ('W', 'P', 3, '<空间>')；否则 None。"""
    parts = _split_space_tail(key)
    if parts is None:
        return None
    tail = parts[2]
    if len(tail) != 6 or not tail.startswith("seg") or not tail[3:].isdigit():
        return None
    return parts[0], parts[1], int(tail[3:]), parts[3]


def _key_ident(key):
    """任意页面键 → (作品, 角色, 行自带的空间 id or None)；不认识 → None。

    空间为 None 表示「看这个角色记着的那一份」（`pages[角色]["space"]`）——
    也就是**当前那批键**；停放键自带空间 id。
    `job__*` 与其它键一律返回 None（那批是审计用，不参与匹配/统计）。
    """
    ident = _page_centroid_of_key(key)
    if ident is not None:
        return ident[0], ident[1], None
    seg = _segment_key_index(key)
    if seg is not None:
        return seg[0], seg[1], None
    parked = _parked_centroid_of_key(key)
    if parked is not None:
        return parked[0], parked[1], parked[2]
    parked = _parked_segment_of_key(key)
    if parked is not None:
        return parked[0], parked[1], parked[3]
    return None


def _page_space(work_rec, page) -> str:
    """该角色**当前那批键**属于哪个空间；没记过（老库/通配）→ ""。"""
    rec = ((work_rec or {}).get("pages") or {}).get(page)
    if not isinstance(rec, dict):
        return ""
    return _as_str(rec.get("space")).strip()


def _space_matches(row_space, want) -> bool:
    """行空间能不能进 `want` 空间的匹配：任一侧为空 = 通配（老库/还没定过模型）。"""
    return not row_space or not want or row_space == want


def _row_participates(work_ok, page_space, row_space, want) -> bool:
    """一行向量该不该参与「当前模型 = `want`」这一次匹配/加权。

    * 记过空间的（当前批或停放批）：**只认同一个空间**；
    * 没记过的（本提交之前写的老库，没有任何空间信息）：沿用既有的
      「作品级身份」口径 `work_ok`（`_work_vectors_usable`）。
    """
    if row_space:
        return _space_matches(row_space, want)
    if page_space:
        return _space_matches(page_space, want)
    return bool(work_ok)


def _page_rows(vectors, work, page) -> list:
    """该角色**当前那批键**的全部行 → [(键, 向量), ...]（不含停放键）。"""
    out = []
    for key, vec in vectors.items():
        ident = _page_centroid_of_key(key)
        if ident is None:
            seg = _segment_key_index(key)
            ident = (seg[0], seg[1]) if seg is not None else None
        if ident == (work, page):
            out.append((key, vec))
    return out


def _park_page_rows(vectors, work, page, sid) -> int:
    """把该角色当前那批键整批停放到 `sid` 空间（**不删**，换回来还能搬回来）。"""
    moved = 0
    for key, vec in _page_rows(vectors, work, page):
        ident = _page_centroid_of_key(key)
        if ident is not None:
            target = page_parked_centroid_key(work, page, sid)
        else:
            target = page_parked_segment_key(work, page, sid,
                                             _segment_key_index(key)[2])
        vectors.pop(key, None)
        vectors[target] = vec
        moved += 1
    return moved


def _unpark_page_rows(vectors, work, page, sid) -> int:
    """把 `sid` 空间停放的行搬回当前那批键（该空间在盘上就没别的行了）。"""
    moved = 0
    for key in list(vectors.keys()):
        parked = _parked_centroid_of_key(key)
        if parked is not None and parked == (work, page, sid):
            vectors[page_centroid_key(work, page)] = vectors.pop(key)
            moved += 1
            continue
        parked = _parked_segment_of_key(key)
        if parked is not None and (parked[0], parked[1], parked[3]) == (work, page, sid):
            vectors[page_segment_key(work, page, parked[2])] = vectors.pop(key)
            moved += 1
    return moved


def _swap_page_stats(rec, out_space, want) -> None:
    """空间切换时统计跟着向量走：当前统计存进 `parked[out_space]`，`want` 那份取回来。

    `n_enroll`/`total_s` 描述的是「这个空间里有多少证据」：把换模型前的入库次数
    接着往上加，面板上就会出现一个不属于当前空间的虚高数字。
    `episodes` 是审计流水（Task 5 计划书第 4 条要求留档），**不跟着搬**。
    """
    parked = rec.get("parked")
    if not isinstance(parked, dict):
        parked = {}
    if out_space:
        parked[out_space] = {
            "n_enroll": max(0, _as_int(rec.get("n_enroll"), 0)),
            "total_s": max(0.0, _as_num(rec.get("total_s"), 0.0)),
        }
        rec["n_enroll"] = 0
        rec["total_s"] = 0.0
    back = parked.pop(want, None) if want else None
    if isinstance(back, dict):
        rec["n_enroll"] = max(0, _as_int(back.get("n_enroll"), 0))
        rec["total_s"] = max(0.0, _as_num(back.get("total_s"), 0.0))
    if parked:
        rec["parked"] = parked
    else:
        rec.pop("parked", None)


def _adopt_legacy_pages(reg, vectors, work, old_model, old_dim, model, dim) -> int:
    """作品级身份翻新时，把该作品**全部**「没记过空间、当前键上有行」的角色归属到旧空间。

    只在翻转的**这一刻**做：`old_model` 是那批无空间的行唯一的归属线索，翻完
    作品记录就被改成新模型了。只归属「换模型后第一个被命名的角色」是老库 P1 的
    形态 —— 之后同一作品里其它角色的旧行会以新空间的名义复活：参与新空间的
    匹配、被 `/api/speakers` 预填、再命名时被 70/30 掺进新空间质心，而且**没有
    任何停放行**，换回旧模型也拿不回来（Task 7 复审 D1）。

    没有旧空间标识（`old_model` 为空 = 连作品级身份都没记过的老库）时不归属：
    没有任何线索能说清那批行是谁算的，只能按通配语义当当前空间（与改造前一致）。
    空间 id 相同（只换了维度）时也不动：`@<同一个空间>` 的停放键是非法状态。

    返回被归属的角色数。
    """
    old_sid = space_id(old_model)
    want = space_id(model)
    if not old_sid or not want or old_sid == want:
        return 0
    if _same_space(old_model, old_dim, model, dim):
        return 0
    pages = ((reg.get("works") or {}).get(work) or {}).get("pages") or {}
    adopted = 0
    for page in list(pages):
        rec = pages.get(page)
        if not isinstance(rec, dict) or _as_str(rec.get("space")).strip():
            continue            # 记过空间的行由 `pages[P]["space"]` 精确判定，不归这条管
        if not _park_page_rows(vectors, work, page, old_sid):
            continue            # 一条当前行都没有：没有可归属的东西
        _swap_page_stats(rec, old_sid, want)
        rec["space"] = want
        rec["updated_at"] = _now()
        adopted += 1
    if adopted:
        _remember_space(reg, old_sid, old_model)
    return adopted


def _remember_space(reg, sid, model) -> None:
    """把空间 id → 模型标识记进 JSON（只为人眼排障；正确性不依赖它）。"""
    if not sid:
        return
    spaces = reg.get("spaces")
    if not isinstance(spaces, dict):
        spaces = {}
        reg["spaces"] = spaces
    spaces[sid] = _as_str(model)


def _same_space(model_a, dim_a, model_b, dim_b) -> bool:
    """两个「模型标识 + 维度」是不是同一个空间（空值 = 通配，与 `_space_matches` 同口径）。"""
    if model_a and model_b and _as_str(model_a).strip() != _as_str(model_b).strip():
        return False
    if dim_a and dim_b and _as_int(dim_a, 0) != _as_int(dim_b, 0):
        return False
    return True


def _work_vectors_usable(work_rec, model) -> bool:
    """该作品的**老库行**（没记过空间的那批）能不能参与当前模型的匹配。

    换模型后旧向量必须失效。模型身份**不写进 npz 键**：键格式是设计 §6.2 定死的
    `page__<作品>__<角色>__centroid`，把模型名塞进去就变成了
    `page__<模型>__<角色>__centroid`，"作品"这一层就没了。

    本提交之后，**记过空间**的行改由 `_row_participates` 按行判定（能精确到
    「这个角色的这批键是哪个模型算的」），这条函数只负责没有任何空间信息的
    老库行 —— 即「作品记录里记着的那一个模型」。它的行为一个字都没变。

    - 作品没记模型（老库/首次入库）= 通配，任何模型下都能用
    - 全局模型为空（还没定过模型）= 通配
    - 两边都记了且不同 = 失效
    """
    recorded = _as_str((work_rec or {}).get("model")).strip()
    current = _as_str(model).strip()
    if not recorded or not current:
        return True
    return recorded == current


def work_rows_in_space(root=None, work="", model="") -> int:
    """该作品在 `model` 这个空间里**实际可参与匹配**的行数（诊断/闸门用）。

    `/api/speakers` 用它判断「库级模型与本集的模型不同」时还有没有东西可比：
    库按空间分开存之后，「库级模型 != 本集模型」不再等于「一个候选都没有」——
    用户换回旧模型时，旧空间的行必须还能用（Task 7 审查 P1 的不变量 2）。
    """
    reg = load_registry(root)
    work = _as_str(work)
    wrec = reg["works"].get(work) or {}
    by_page = _page_vectors(load_vectors(root), wrec, model, reg["dim"])
    return sum(len(rows) for (w, _page), rows in by_page.items() if w == work)


def page_rows_in_model(vectors, work_rec, model, dim=0) -> dict:
    """→ {角色: [向量, ...]}，只含该作品在当前模型/维度下可用的向量。

    一行包装 `_page_vectors`（私有），**当前全仓（含测试）零调用方**：面板需要的
    「这个作品下能参与匹配的角色」已经由 `list_registry` 的 `vectors_usable`
    / `n_vectors` 满足（面板用的就是后者）。留着是因为它是有用的诊断口
    （`work_rows_in_space` 就是同一套判定的计数版）；真要接进面板再接，
    **不要再在文档里承诺一个不存在调用方**（旧 docstring 就是这么变成假话的）。
    """
    return {page: list(rows)
            for (work, page), rows in _page_vectors(vectors, work_rec, model, dim).items()
            if work}


# --------------------------------------------------------------------------
# 库文件读写（绝不抛）
# --------------------------------------------------------------------------

def empty_registry() -> dict:
    return {
        "version": VERSION,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "dim": DEFAULT_DIM,
        "normalized": True,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "work_hints": {},
        "works": {},
        "spaces": {},
        "updated_at": "",
    }


def _coerce_thresholds(value) -> dict:
    out = dict(DEFAULT_THRESHOLDS)
    if not isinstance(value, dict):
        return out
    for name, default in (("auto", DEFAULT_AUTO), ("suggest", DEFAULT_SUGGEST)):
        raw = value.get(name, None)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        num = float(raw)
        if math.isfinite(num) and 0.0 < num < 1.0:
            out[name] = num
    return out


def _coerce_episodes(value) -> list:
    out = []
    for item in value if isinstance(value, list) else ():
        if not isinstance(item, dict):
            continue
        rec = {
            "job": _as_str(item.get("job")),
            "cluster": _as_str(item.get("cluster")),
            "score": _as_num(item.get("score"), 0.0),
            "source": _as_str(item.get("source")) or "manual",
            "at": _as_str(item.get("at")),
        }
        secs = item.get("total_s", None)
        if secs is not None:
            rec["total_s"] = _as_num(secs, 0.0)
        out.append(rec)
    return out


def _coerce_page(rec) -> dict:
    """一个角色的记录。`space` = 它**当前那批键**是哪个模型空间算的（"" = 老库/通配）。

    `parked` 存的是「别的空间里还有多少证据」（换空间时 `n_enroll`/`total_s`
    跟着向量一起进出，见 `_swap_page_stats`）：有它才不会把换模型前的入库次数
    算进新空间，也才不会丢。
    """
    if not isinstance(rec, dict):
        rec = {"name": rec}
    out = {
        "n_enroll": max(0, _as_int(rec.get("n_enroll"), 0)),
        "total_s": max(0.0, _as_num(rec.get("total_s"), 0.0)),
        "episodes": _coerce_episodes(rec.get("episodes")),
        "updated_at": _as_str(rec.get("updated_at")),
        "note": _as_str(rec.get("note")),
        "space": _as_str(rec.get("space")).strip(),
    }
    parked = {}
    raw = rec.get("parked")
    if isinstance(raw, dict):
        for sid, prec in raw.items():
            if not isinstance(sid, str) or not sid or not isinstance(prec, dict):
                continue
            parked[sid] = {"n_enroll": max(0, _as_int(prec.get("n_enroll"), 0)),
                           "total_s": max(0.0, _as_num(prec.get("total_s"), 0.0))}
    if parked:
        out["parked"] = parked
    return out


def _coerce_work(wrec) -> dict:
    """一个作品的记录。`model` 是「这批向量是用哪个嵌入模型算的」。"""
    pages = wrec.get("pages") if isinstance(wrec, dict) else None
    out_pages = {}
    if isinstance(pages, dict):
        for page, prec in pages.items():
            if not isinstance(page, str) or not page:
                continue
            out_pages[page] = _coerce_page(prec)
    return {
        "model": _as_str(wrec.get("model")).strip() if isinstance(wrec, dict) else "",
        "dim": max(0, _as_int(wrec.get("dim"), 0)) if isinstance(wrec, dict) else 0,
        "pages": out_pages,
        "updated_at": _as_str(wrec.get("updated_at")) if isinstance(wrec, dict) else "",
    }


def load_registry(root=None) -> dict:
    """读元数据。文件缺失/损坏/手改错类型 → 退回空库，**绝不抛**。"""
    with _SAVE_LOCK:
        p = registry_path(root)
        reg = empty_registry()
        if not p.is_file():
            return reg
        try:
            text = p.read_text(encoding="utf-8-sig")
        except Exception:
            return reg
        try:
            data = json.loads(text)
        except Exception:
            return reg
        if not isinstance(data, dict):
            return reg
        reg["version"] = _as_int(data.get("version"), VERSION) or VERSION
        reg["embedding_model"] = _as_str(data.get("embedding_model"))
        dim = _as_int(data.get("dim"), DEFAULT_DIM)
        reg["dim"] = dim if dim > 0 else DEFAULT_DIM
        reg["normalized"] = bool(data.get("normalized", True))
        reg["thresholds"] = _coerce_thresholds(data.get("thresholds"))
        reg["updated_at"] = _as_str(data.get("updated_at"))

        hints = data.get("work_hints")
        reg["work_hints"] = {}
        if isinstance(hints, dict):
            for key, val in hints.items():
                if isinstance(key, str) and val:
                    name = _as_str(val).strip()
                    if name:
                        reg["work_hints"][key] = name

        spaces = data.get("spaces")
        reg["spaces"] = {}
        if isinstance(spaces, dict):
            for key, val in spaces.items():
                if isinstance(key, str) and key:
                    tag = _as_str(val).strip()
                    if tag:
                        reg["spaces"][key] = tag

        works = data.get("works")
        out_works = {}
        if isinstance(works, dict):
            for work, wrec in works.items():
                if not isinstance(work, str) or not work:
                    continue
                if not isinstance(wrec, dict):
                    continue
                out_works[work] = _coerce_work(wrec)
        reg["works"] = out_works
        return reg


def save_registry(root=None, reg=None) -> None:
    """原子写元数据。`reg` 里没给的字段用当前盘上的值补齐（部分更新）。"""
    if not isinstance(reg, dict):
        reg = {}
    path = registry_path(root)
    with _SAVE_LOCK:
        disk = load_registry(root)
        merged = dict(disk)
        merged.update({k: v for k, v in reg.items() if v is not None})
        merged["version"] = VERSION
        merged["normalized"] = True
        merged["thresholds"] = _coerce_thresholds(merged.get("thresholds"))
        merged["work_hints"] = {k: _as_str(v) for k, v in
                                (merged.get("work_hints") or {}).items()
                                if isinstance(k, str) and _as_str(v)}
        # 空间 id → 模型标识（只为人眼：npz 键里那段摘要翻回模型名）。部分更新时
        # 不能把别人记下的空间表抹掉，所以和 works 一样按当前的并集写。
        merged["spaces"] = {k: _as_str(v) for k, v in
                            (merged.get("spaces") or {}).items()
                            if isinstance(k, str) and k and _as_str(v)}
        merged["updated_at"] = _now()
        payload = {
            "version": VERSION,
            "embedding_model": _as_str(merged.get("embedding_model")),
            "dim": _as_int(merged.get("dim"), DEFAULT_DIM) or DEFAULT_DIM,
            "normalized": True,
            "thresholds": merged["thresholds"],
            "work_hints": merged["work_hints"],
            "works": merged.get("works") if isinstance(merged.get("works"), dict) else {},
            "spaces": merged["spaces"],
            "updated_at": merged["updated_at"],
        }
        # 文件要给人看：ensure_ascii=False + 缩进
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_vectors(root=None) -> dict:
    """读向量。文件缺失/损坏 → `{}`，**绝不抛**。坏条目单条丢弃。"""
    with _SAVE_LOCK:
        p = vectors_path(root)
        if not p.is_file():
            return {}
        try:
            with np.load(str(p)) as z:
                files = list(z.files)
                raw = {k: z[k] for k in files}
        except Exception:
            return {}
        out = {}
        for key, value in raw.items():
            if not _looks_like_key(key):
                continue
            arr = _loadable_vector(value)       # 零向量 / NaN / 非一维 → 丢这一条
            if arr is None:
                continue
            out[key] = arr.astype(np.float32)
        return out


def _looks_like_key(key) -> bool:
    if not isinstance(key, str):
        return False
    if key.startswith("page__"):
        return (_segment_key_index(key) is not None
                or _page_centroid_of_key(key) is not None
                # 停放键（另一个空间的行）必须认：否则 save_vectors 会把它**静默
                # 丢掉** —— 那正是「换个模型就删用户数据」
                or _parked_segment_of_key(key) is not None
                or _parked_centroid_of_key(key) is not None)
    if key.startswith("job__"):
        parts = key.split("__")
        return len(parts) == 4 and parts[3] == "centroid" and bool(parts[1]) and bool(parts[2])
    return False


def save_vectors(root=None, vectors=None) -> None:
    """原子写全部向量（整库覆盖）。空库就删掉文件。非法向量直接不写。"""
    path = vectors_path(root)
    arrays = {}
    for key, value in (vectors or {}).items():
        if not _looks_like_key(key):
            continue
        unit = normalise_vector(value)          # 全零/NaN/Inf 在这里被挡掉
        if unit is not None:
            arrays[key] = unit
    if not arrays:
        _drop_file(path)
        return
    _atomic_write_npz(path, arrays)


def _loadable_vector(value):
    """读回来的键值能不能用：必须是一维全零以外的有限数组。

    归一化到接近 0 的范数说明是**坏数据**（pyannote 的补位零行、
    手改出来的零向量），不是一条「方向很弱」的向量 —— 方向正是匹配唯一要的东西，
    范数接近 0 的向量方向完全由噪声决定。手改坏的单条丢弃，不影响其余条目。
    """
    arr = _vec_or_none(value)
    if arr is None:
        return None
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm < _MIN_USABLE_NORM:
        return None
    return arr


def _page_vectors(vectors, work_rec, model, dim=0):
    """→ {(作品, 角色): [向量, ...]}，只含该作品在当前**空间**/维度下可用的向量。

    `dim` 用**全局**那个（`reg["dim"]`）：作品记录里那份只是登记时的快照，
    用户手改全局维度或换了个不同维度的模型时，快照会过期；
    而全局 dim 是「现在这个嵌入模型是多少维」的唯一权威。

    空间（Task 7 审查 P1 的修复）：每行向量属于哪个嵌入模型空间，由键决定 ——
    当前那批键看 `pages[角色]["space"]`，停放键（`…@<空间>`）自带空间。
    只有空间对得上的行才参与余弦；换过模型的旧行留在盘上但一行都不进来
    （`_row_participates`）。
    """
    want = space_id(model)
    work_ok = _work_vectors_usable(work_rec, model)
    limit_dim = _as_int(dim, 0)
    by_page = {}
    for key, vec in vectors.items():
        ident = _key_ident(key)
        if ident is None:
            continue
        work, page, row_space = ident
        if not _row_participates(work_ok, _page_space(work_rec, page), row_space, want):
            continue
        arr = _vec_or_none(vec)
        if arr is None:
            continue
        if limit_dim and arr.shape[0] != limit_dim:
            continue
        by_page.setdefault((work, page), []).append(vec)
    return by_page


# --------------------------------------------------------------------------
# 打分 / 分档 / 一对一
# --------------------------------------------------------------------------

def _band_of(score, thresholds):
    if score is None:
        return "none"
    if score >= thresholds["auto"]:
        return "auto"
    if score >= thresholds["suggest"]:
        return "suggest"
    return "none"


def greedy_pair(by_cluster) -> dict:
    """一对一贪心配对：分数从高到低配，**一个已有角色在一集里最多配一个聚类**。

    `by_cluster` = {聚类: [候选, ...]}，候选里有 `page` / `score` / `auto`。
    返回 {聚类: 角色名 or None}；同分时按 (角色名, 聚类) 排序，结果确定可复现。
    """
    pairs = []
    for cluster, cands in (by_cluster or {}).items():
        for cand in cands or ():
            if not isinstance(cand, dict) or not cand.get("auto"):
                continue
            score = cand.get("score")
            if score is None:
                continue
            pairs.append((-float(score), _as_str(cand.get("page")), _as_str(cluster)))
    pairs.sort()
    out = {str(c): None for c in (by_cluster or {})}
    taken = set()
    for _neg, page, cluster in pairs:
        if cluster in out and out[cluster] is None and page not in taken:
            out[cluster] = page
            taken.add(page)
    return out


def match_vectors(root=None, work="", clusters=None, model=None) -> dict:
    """给一集的每个聚类找候选角色。

    返回 {聚类: {"candidates": [...], "top": ...|None, "auto_page": ...|None}}：
    - `candidates` 列出**该作品下每个有向量的角色**（按分数从高到低），
      每项 `{page, score, band, auto, n_vectors}`；`band` 是
      `auto`（>= auto 阈值，可预填）/ `suggest`（候选）/ `none`（留空）
    - `top` 是排第一的候选（含 `auto` 标志，可能不在 auto 档）
    - `auto_page` 是**一对一贪心之后**才确定的名字：两个聚类抢同一角色时，
      只有分数高的那个拿得到，另一个是 None。

    `clusters` 可以是 {聚类: 向量} 或 {聚类: {"embedding": 向量}}。

    `model` 是**这批查询向量**所属的嵌入模型标识，不给就用库级那个。库里按空间
    分开存之后这两者可以不同（用户换回了旧模型），按空间比才不会拿 A 空间的
    质心去跟 B 空间的库向量算余弦（D7 要防的静默混比）。
    """
    reg = load_registry(root)
    thresholds = reg["thresholds"]
    want = _as_str(reg.get("embedding_model")) if model is None else _as_str(model).strip()
    work = _as_str(work)
    work_rec = reg["works"].get(work) or {}
    keys = set(work_rec.get("pages") or {})
    by_page = _page_vectors(load_vectors(root), work_rec, want, reg["dim"])
    keys.update(page for (w, page) in by_page if w == work)
    rows_by_page = {}
    for key in keys:
        rows_by_page[key] = by_page.get((work, key), [])

    per_cluster = {}
    for cluster, raw in (clusters or {}).items():
        cluster = str(cluster)
        vec = raw
        if isinstance(raw, dict):
            vec = raw.get("embedding")
        unit = normalise_vector(vec)
        cands = []
        if unit is not None:
            for page, rows in rows_by_page.items():
                score = top_mean(unit, rows)
                if score is None:
                    continue
                cands.append({"page": page, "score": score,
                              "band": _band_of(score, thresholds),
                              "auto": score >= thresholds["auto"],
                              "n_vectors": len(rows)})
        cands.sort(key=lambda c: (-c["score"], c["page"]))
        per_cluster[cluster] = {"candidates": cands,
                                "top": cands[0] if cands else None,
                                "auto_page": None}

    # 一对一贪心必须在**候选列表**上做（不是 per_cluster 那个包着 candidates/auto_page
    # 的外层字典，喂错了会把所有聚类都配成 None）
    auto_pages = greedy_pair({c: info["candidates"] for c, info in per_cluster.items()})
    for cluster, page in auto_pages.items():
        if cluster in per_cluster:
            per_cluster[cluster]["auto_page"] = page
    return per_cluster


# --------------------------------------------------------------------------
# 入库 / 维护
# --------------------------------------------------------------------------

def _page_rec(reg, work, page) -> dict:
    works = reg.setdefault("works", {})
    wrec = works.get(work)
    if not isinstance(wrec, dict):
        wrec = _coerce_work({})
        works[work] = wrec
    wrec.setdefault("model", "")
    wrec.setdefault("dim", 0)
    pages = wrec.get("pages")
    if not isinstance(pages, dict):
        pages = {}
        wrec["pages"] = pages
    rec = pages.get(page)
    if not isinstance(rec, dict):
        rec = _coerce_page({})
        pages[page] = rec
    return rec


def _work_rec(reg, work) -> dict:
    """取（必要时建）一个作品的记录；调用方拿到的就是库里的那一份。"""
    works = reg.setdefault("works", {})
    wrec = works.get(work)
    if not isinstance(wrec, dict):
        wrec = _coerce_work({})
        works[work] = wrec
    wrec.setdefault("model", "")
    wrec.setdefault("dim", 0)
    wrec.setdefault("pages", {})
    return wrec


def register_page(root=None, work="", page="") -> dict:
    """只登记角色（可能还没有向量）。作品名/角色名为空则不做任何事。"""
    work = _as_str(work).strip()
    page = _as_str(page).strip()
    if not work or not page:
        return load_registry(root)
    with _SAVE_LOCK:
        reg = load_registry(root)
        rec = _page_rec(reg, work, page)
        rec["updated_at"] = _now()
        save_registry(root, reg)
    return reg


def enroll(root=None, work="", page="", vector=None, job="", cluster="",
           seconds=0.0, model=None, dim=None, source="manual") -> dict:
    """把一次命名的聚类质心并入该角色。

    - 质心 `centroid = normalize(0.7*old + 0.3*new)`，**只在同一个模型空间内**
      加权：`old` 取的是当前空间里那条质心（首次入这个空间、或维度不同 →
      退化为 `new` 本身）
    - 本次质心另存一条 `segNNN`（循环 0..MAX_SEGMENTS-1，保留多条更稳）
    - 给了 job/cluster 就再存一条 `job__<job>__<cluster>__centroid` 供审计
    - `model`/`dim` 是**全局**的嵌入身份；某个角色（先看角色、再看作品）记着的
      空间与它不一致 → 换空间：当前那批键整批**停放**到旧空间（不删），
      新空间的旧行（如果以前存过）搬回当前键位，`n_enroll`/`total_s` 跟着走
      —— 两个空间的行永远不会混着比、混着加权（Task 7 审查 P1）
    - 作品级身份（`works[W]["model"]/["dim"]`）翻新的**那一次**，该作品**所有**
      「没记过空间、当前键上有行」的角色**一次性**归属到旧空间（`@<旧空间>`，
      统计一起进停放区）—— 归属线索只有那一刻还在，翻完作品身份就被新值覆盖
      了。这一步只看作品身份、**不看这次有没有向量**：没有向量的一次登记同样会
      翻新身份、同样会丢线索（Task 7 复审 D1）

    向量非法（全零/NaN/Inf）→ 只登记角色、不写向量，返回 `vector_stored=False`。
    （只登记时不动**被登记角色自己**的空间：没有新行就没有可换的空间，动了就等于
    凭空改计数。上面那条整作品归属不在此列 —— 它由作品身份翻新触发。）
    """
    work = _as_str(work).strip()
    page = _as_str(page).strip()
    unit = normalise_vector(vector)
    if not work or not page:
        return {"ok": False, "vector_stored": False, "reason": "empty_work_or_page"}
    job = _as_str(job).strip()
    cluster = _as_str(cluster).strip()
    secs = max(0.0, _as_num(seconds, 0.0))
    model = _as_str(model).strip()
    dim = _as_int(dim, 0) if dim is not None else 0

    with _SAVE_LOCK:
        reg = load_registry(root)
        # 全局默认模型：显式给了就更新，没给就沿用库里的（首次入库时记下来）。
        # 维度可以直接从向量量出来 —— 记错了会让**全部**向量都参与不了匹配
        # （`_page_vectors` 按全局 dim 过滤），所以没传就自己数。
        if model:
            reg["embedding_model"] = model
        if not dim and unit is not None:
            dim = int(unit.shape[0])
        if dim and dim > 0:
            reg["dim"] = dim
        global_model = _as_str(reg.get("embedding_model"))
        global_dim = _as_int(reg.get("dim"), DEFAULT_DIM) or DEFAULT_DIM

        wrec = _work_rec(reg, work)
        # 这个作品的向量身份：没记过就跟随全局
        old_work_model = _as_str(wrec.get("model")).strip()
        old_work_dim = _as_int(wrec.get("dim"), 0)
        wrec["model"] = old_work_model or global_model
        wrec["dim"] = old_work_dim or global_dim
        identity_changed = (wrec["model"] != global_model) or (wrec["dim"] != global_dim)
        vectors = load_vectors(root)
        adopted = 0
        if identity_changed:
            # 换模型/换维度：作品级身份换成新的。**记过空间的行不受它管辖**
            # （那些行由 `pages[P]["space"]` 精确判定），它只管老库那批没记过
            # 空间的行 —— 行为与本提交之前完全一致。
            wrec["model"] = global_model
            wrec["dim"] = global_dim
            # 老库升级：趁旧身份还在手里，把该作品**所有**「无空间且有当前行」
            # 的角色一次性归属到旧空间。只归属当前这一页的话，同一作品里其它
            # 角色的旧行会以新空间的名义复活（Task 7 复审 D1）。
            adopted = _adopt_legacy_pages(reg, vectors, work, old_work_model,
                                          old_work_dim, global_model, global_dim)

        rec = _page_rec(reg, work, page)
        stored = False
        if unit is not None:
            # ---- 空间：换过模型的向量绝不参与当前空间的匹配与加权 ----
            want = space_id(global_model)
            page_space = _as_str(rec.get("space")).strip()
            if want and page_space and page_space != want:
                # 这个角色已经换过空间：当前那批键整批停放到旧空间，新空间以前
                # 存过的搬回当前键位。一批键**永远只属于一个空间**，所以
                # `page__W__P__centroid` 不会变成两个空间混起来的质心。
                _park_page_rows(vectors, work, page, page_space)
                _unpark_page_rows(vectors, work, page, want)
                _swap_page_stats(rec, page_space, want)
            elif want and not page_space and _page_rows(vectors, work, page):
                # 老库：这个角色有行但没记过空间。这批行要么就是当前空间的，
                # 要么是**这个作品上一个模型**算的 —— 后者必须归属过去，否则
                # 作品身份一翻新它们就会以当前空间的名义复活（正是 P1）。
                if not _same_space(old_work_model, old_work_dim,
                                   global_model, global_dim):
                    old_sid = space_id(old_work_model)
                    if old_sid:
                        _park_page_rows(vectors, work, page, old_sid)
                        _swap_page_stats(rec, old_sid, want)
                    # 连作品级身份都没记过（`old_work_model` 为空）：没有任何线索
                    # 能说清这批行是谁算的，只能当当前空间（老库的通配语义），
                    # 这条与改造前一致。
            if want:
                rec["space"] = want
                _remember_space(reg, want, global_model)
            # `want` 为空（库还没定过嵌入标识）= 通配：不记空间、不动已有的键，
            # 参与判定全按老库口径（`_row_participates` 的 work_ok 分支）。
            ckey = page_centroid_key(work, page)
            old = _vec_or_none(vectors.get(ckey))
            if old is not None and old.shape[0] != unit.shape[0]:
                old = None              # 维度不同 = 不同空间，不能加权
            if old is not None:
                blended = normalise_vector(0.7 * old + 0.3 * unit.astype(np.float64))
                vectors[ckey] = blended if blended is not None else unit
            else:
                vectors[ckey] = unit

            existing = _seg_indices(vectors, ckey[:-len("centroid")])
            idx = (existing[-1][0] + 1) % MAX_SEGMENTS if existing else 0
            vectors[ckey[:-len("centroid")] + "seg%03d" % idx] = unit
            # 超过上限的旧 seg 键清掉（保证 npz 不无限增长）
            for seg_idx, seg_key in existing:
                if seg_idx >= MAX_SEGMENTS:
                    vectors.pop(seg_key, None)

            if job and cluster:
                vectors[job_centroid_key(job, cluster)] = unit

            # episodes 里的 score = 本次质心与该角色**同一个空间里其他**已存向量
            # 的 top-3 平均（即「这次和历史的吻合程度」，首次入库没有可比对象，记 0）
            others = [v for _tag, key, v in _page_row_list(vectors, work, page)
                      if key != ckey]
            score = top_mean(unit, others)
            save_vectors(root, vectors)
            stored = True
            rec["n_enroll"] = max(0, _as_int(rec.get("n_enroll"), 0)) + 1
            episodes = rec.setdefault("episodes", [])
            episodes.append({"job": job, "cluster": cluster,
                             "score": round(float(score), 6) if score is not None else 0.0,
                             "source": _as_str(source) or "manual",
                             "at": _now(), "total_s": secs})
        elif adopted:
            # 只登记（这次没有向量）也翻新了作品级身份：整作品归属必须**落盘**，
            # 否则 JSON 说那批行属于新空间、npz 里它们还躺在当前键位上 —— 既是
            # D1 那个形态，又让停放统计与停放键对不上（一条 `@` 键都没有）。
            save_vectors(root, vectors)
        rec["total_s"] = max(0.0, _as_num(rec.get("total_s"), 0.0)) + secs
        rec["updated_at"] = _now()
        wrec["updated_at"] = _now()
        save_registry(root, reg)

    return {"ok": True, "vector_stored": stored, "work": work, "page": page,
            "model": global_model, "dim": global_dim,
            "identity_changed": identity_changed}


def _page_row_list(vectors, work, page) -> list:
    """该角色的全部向量 → [(来源标签, 键, 向量), ...]。

    来源标签：`centroid` / `segNNN`；按标签排序（centroid 在前）。
    """
    out = []
    for key, vec in vectors.items():
        ident = _page_centroid_of_key(key)
        if ident == (work, page):
            out.append(("centroid", key, vec))
            continue
        seg = _segment_key_index(key)
        if seg is not None and (seg[0], seg[1]) == (work, page):
            out.append(("seg%03d" % seg[2], key, vec))
    out.sort(key=lambda r: r[0])
    return out


def _seg_indices(vectors, prefix):
    """→ [(segIndex, key), ...]，按 index 排序。"""
    out = []
    for key in vectors:
        seg = _segment_key_index(key)
        if seg is None:
            continue
        if key.startswith(prefix):
            out.append((seg[2], key))
    return sorted(out)


def rename_page(root=None, work="", old="", new="") -> dict:
    """角色改名：把它的全部向量与统计一起搬到新名字（新名字已存在则合并统计）。

    停放键（别的空间的行）**一起搬** —— 否则改个名就把这个角色在旧模型下的
    历史丢在旧名字下面，换回旧模型时它再也回不来。
    两个角色的空间不同（A 改名并进 B 空间的角色）时，搬过去的行**进停放区**：
    否则新名字的当前那批键里就混进了 A 空间的行（等于换个名字就制造 P1）。

    **并入已存在的角色时，源的 `segNNN` 按序号续排**（Task 7 复审 D3）。面板上
    「角色名填一个已存在的名字 = 并入该角色」是文档化的正常路径，而源行的序号与
    目标已有的序号往往重合 —— 照搬就是**按键覆盖**，实测 6 行 → 4 行、5 行 → 3 行，
    覆盖掉的是用户的声纹向量，拿不回来。所以：

    * 每个**命名空间**（当前那批键算一个；`@<空间>` 停放键各算一个）单独续排，
      起点是该命名空间里**目标已有序号的最大值 + 1**；到 `MAX_SEGMENTS` 顶了就
      绕回低位找**空位**（只填空位，不覆盖）。源自己的当前行与停放行可能落进
      同一个命名空间，序号一起排，不会互相撞；
    * 序号**不越过** `MAX_SEGMENTS`：`enroll` 的下一条 seg 用
      `(existing[-1][0] + 1) % MAX_SEGMENTS` 算序号，还会把 `>= MAX_SEGMENTS` 的
      键删掉 —— 排到 30 以上等于把丢行换个地方发生；
    * 一个命名空间最多 `MAX_SEGMENTS` 行，**真的塞满**时无位可续，只能像 `enroll`
      的环那样从最老的（0 号）开始覆盖，此时在返回值的 `overwritten` 里如实计数，
      绝不假装没丢；
    * `centroid` 行不续排：一个角色只有一条质心，合并时就该覆盖（既有语义）；
      `job__*` 审计键一律不动。

    返回 `{"ok", "moved", "work", "from", "to", "overwritten"}`：`moved` 是搬走的
    源行数，`overwritten` 是上面那个「命名空间已满」兜底真正覆盖掉的目标行数
    （正常情况恒为 0）。
    """
    work = _as_str(work).strip()
    old = _as_str(old).strip()
    new = _as_str(new).strip()
    if not work or not old or not new or old == new:
        return {"ok": False}
    with _SAVE_LOCK:
        reg = load_registry(root)
        wrec = reg["works"].get(work) or {}
        pages = wrec.get("pages") or {}
        src_space = _page_space(wrec, old)
        dst_space = _page_space(wrec, new) if new in pages else src_space
        # 只有能说清「从哪个空间搬来」时才停止放（空间 id 为空 = 老库行，没有
        # 可归属的空间，只能照旧留在当前键位 —— 那批行本来就由作品级判据管）
        park = bool(src_space) and src_space != dst_space
        vecs = load_vectors(root)
        moved = 0
        # ---- 第一遍：只算「每一行去哪」，一行都还没落笔 ----
        # seg 行的序号**必须先把目标已有哪些序号算清楚再分配**：源行的序号与目标
        # 已有的序号重合时照搬就是覆盖（复审 D3）。而且不能边遍历边定 —— 源自己的
        # 当前行与停放行可能落进**同一个**命名空间（源的停放行正好是目标空间时它
        # 会被搬回当前那批键），那样源行之间也会自己撞。
        plan = {}           # 源键 → 目标键
        seg_plan = []       # [(源序号, 源键, 目标命名空间)]；命名空间 None = 当前那批键
        for key in vecs:
            if _page_centroid_of_key(key) == (work, old):
                # 质心：一个角色只有一条，合并时覆盖而不是续排（既有语义）
                plan[key] = (page_parked_centroid_key(work, new, src_space) if park
                             else page_centroid_key(work, new))
                continue
            seg = _segment_key_index(key)
            if seg is not None and (seg[0], seg[1]) == (work, old):
                seg_plan.append((seg[2], key, src_space if park else None))
                continue
            parked = _parked_centroid_of_key(key)
            if parked is not None and (parked[0], parked[1]) == (work, old):
                plan[key] = (page_centroid_key(work, new) if parked[2] == dst_space
                             else page_parked_centroid_key(work, new, parked[2]))
                continue
            parked = _parked_segment_of_key(key)
            if parked is not None and (parked[0], parked[1]) == (work, old):
                seg_plan.append((parked[2], key,
                                 None if parked[3] == dst_space else parked[3]))
        # 目标在**每个命名空间**已经占了哪些序号。只认目标的键：源的行全都要搬走，
        # 不占位；目标原有的行走不到这里，它一条都不搬。
        occupied = {}
        for key in vecs:
            ident = _segment_key_index(key)
            sid = None
            if ident is None:
                parked = _parked_segment_of_key(key)
                if parked is None:
                    continue
                ident, sid = (parked[0], parked[1], parked[2]), parked[3]
            if (ident[0], ident[1]) == (work, new):
                occupied.setdefault(sid, set()).add(ident[2])
        # 每个命名空间的续排状态：taken = 已占（含本次已分配出去的），want = 下一个候选。
        # 起点「目标最大序号 + 1」= 接着排；到顶绕回低位找空位 —— 都能保证一行不覆盖。
        state = {sid: {"taken": set(taken), "want": max(taken) + 1 if taken else 0}
                 for sid, taken in occupied.items()}
        overwritten = 0
        overwritten_in_ns = {}      # 兜底覆盖时的下一候选序号（每个命名空间各算）
        # 源行按序号排序；序号相同时先当前那批键、再各停放空间（按空间 id），最后按
        # 源键名 —— 排序键全部来自键本身，与 npz 里的键序无关：同一批键无论文件被
        # 谁重写过，分配出来的序号都一样。
        seg_plan.sort(key=lambda item: (item[0], item[2] or "", item[1]))
        for src_index, key, sid in seg_plan:
            st = state.setdefault(sid, {"taken": set(), "want": 0})
            idx = None
            for step in range(MAX_SEGMENTS):
                cand = (st["want"] + step) % MAX_SEGMENTS
                if cand not in st["taken"]:
                    idx = cand
                    break
            if idx is None:
                # 一个命名空间最多 MAX_SEGMENTS 行（与 `enroll` 的环一致）：真塞满了
                # 就没有空位可续，只能从最老的那条起覆盖 —— 如实计数，不静默。
                idx = overwritten_in_ns.get(sid, 0)
                overwritten_in_ns[sid] = idx + 1
                overwritten += 1
            st["taken"].add(idx)
            st["want"] = idx + 1
            plan[key] = (page_parked_segment_key(work, new, sid, idx) if sid
                         else page_segment_key(work, new, idx))
        # ---- 第二遍：按原始键序落笔（与改造前同一顺序，只是目标键名不同） ----
        for key in list(vecs.keys()):
            target = plan.get(key)
            if target is None:
                continue
            vecs[target] = vecs.pop(key)
            moved += 1
        # 合并/搬迁元数据
        if old in pages:
            rec = pages.pop(old)
        else:
            rec = _coerce_page({})
        if new in pages:
            dst = pages[new]
            same_space = _as_str(dst.get("space")).strip() == _as_str(rec.get("space")).strip()
            if same_space:
                # 同一个空间才把统计相加；跨空间相加等于把 A 空间的入库次数
                # 记到 B 空间头上（面板上就是个虚高数字）
                dst["n_enroll"] = max(0, _as_int(dst.get("n_enroll"), 0)) \
                    + max(0, _as_int(rec.get("n_enroll"), 0))
                dst["total_s"] = max(0.0, _as_num(dst.get("total_s"), 0.0)) \
                    + max(0.0, _as_num(rec.get("total_s"), 0.0))
            elif _as_str(rec.get("space")).strip():
                # 跨空间：被并进来的那份统计存进停放区（换回那个模型时再取出来）
                parked_stats = dst.get("parked")
                if not isinstance(parked_stats, dict):
                    parked_stats = {}
                src_sid = _as_str(rec.get("space")).strip()
                prev = parked_stats.get(src_sid)
                add_enroll = max(0, _as_int(rec.get("n_enroll"), 0))
                add_secs = max(0.0, _as_num(rec.get("total_s"), 0.0))
                if isinstance(prev, dict):
                    add_enroll += max(0, _as_int(prev.get("n_enroll"), 0))
                    add_secs += max(0.0, _as_num(prev.get("total_s"), 0.0))
                parked_stats[src_sid] = {"n_enroll": add_enroll, "total_s": add_secs}
                dst["parked"] = parked_stats
            dst["episodes"] = list(dst.get("episodes") or []) + list(rec.get("episodes") or [])
            dst["updated_at"] = _now()
            pages[new] = dst
        else:
            rec["updated_at"] = _now()
            pages[new] = rec
        _page_rec(reg, work, new)
        save_vectors(root, vecs)
        save_registry(root, reg)
    return {"ok": True, "moved": moved, "work": work, "from": old, "to": new,
            "overwritten": overwritten}


def forget_page(root=None, work="", page="") -> dict:
    """忘掉一个角色：元数据与它的向量一起删（`job__*` 审计键保留）。

    删的是这个角色的**全部**空间（当前键 + 所有停放键）—— 用户点的是「删除
    角色」，面板也写明「它的全部声纹与统计都会删掉」，留下别的模型下的历史
    不删就成了「删了还在」。
    """
    work = _as_str(work).strip()
    page = _as_str(page).strip()
    if not work or not page:
        return {"ok": False}
    with _SAVE_LOCK:
        reg = load_registry(root)
        pages = (reg["works"].get(work) or {}).get("pages") or {}
        pages.pop(page, None)
        vecs = load_vectors(root)
        removed = 0
        for key in list(vecs.keys()):
            ident = _page_centroid_of_key(key)
            if ident is None:
                seg = _segment_key_index(key)
                ident = (seg[0], seg[1]) if seg is not None else None
            if ident is None:
                parked = _parked_centroid_of_key(key)
                if parked is None:
                    parked = _parked_segment_of_key(key)
                    ident = (parked[0], parked[1]) if parked is not None else None
                else:
                    ident = (parked[0], parked[1])
            if ident == (work, page):
                vecs.pop(key)
                removed += 1
        save_vectors(root, vecs)
        save_registry(root, reg)
    return {"ok": True, "removed": removed, "work": work, "page": page}


def list_registry(root=None) -> list:
    """面板用的一行一角色列表（只含 JSON 能直接序列化的值）。

    `n_vectors` / `vectors_usable` 都按**当前空间**算：换过模型的旧行不进
    `n_vectors`（那是一句「这个角色现在有多少证据」），一行都用不上时
    `vectors_usable` 才为 False（面板据此标「旧模型，不参与匹配」）。
    """
    reg = load_registry(root)
    vecs = load_vectors(root)
    want = space_id(reg["embedding_model"])
    rows = []
    for work in sorted(reg["works"]):
        wrec = reg["works"].get(work) or {}
        pages = wrec.get("pages") or {}
        work_ok = _work_vectors_usable(wrec, reg["embedding_model"])
        for page in sorted(pages):
            rec = pages[page]
            page_space = _page_space(wrec, page)
            n_vectors = 0
            for key in vecs:
                ident = _key_ident(key)
                if ident is None or (ident[0], ident[1]) != (work, page):
                    continue
                if not _row_participates(work_ok, page_space, ident[2], want):
                    continue
                n_vectors += 1
            spaces = reg.get("spaces") if isinstance(reg.get("spaces"), dict) else {}
            rows.append({
                "work": work,
                "page": page,
                "n_enroll": int(rec.get("n_enroll") or 0),
                "total_s": float(rec.get("total_s") or 0.0),
                "n_vectors": n_vectors,
                "model": _as_str(wrec.get("model")),
                "space": page_space,
                "space_tag": _as_str(spaces.get(page_space)) if page_space else "",
                # 有参与的行 → 能匹配；一行都没有 → 沿用老口径（登记了还没入库的
                # 角色不该被标成「旧模型」，那是假话）
                "vectors_usable": bool(n_vectors > 0
                                       or (work_ok and _space_matches(page_space, want))),
                "episodes": list(rec.get("episodes") or []),
                "updated_at": _as_str(rec.get("updated_at")),
            })
    return rows


# --------------------------------------------------------------------------
# work_hints：媒体所在目录 → 作品名
# --------------------------------------------------------------------------

def media_dir(media):
    """媒体文件所在目录（规范化）。不是可用路径 → None。"""
    text = _as_str(media).strip()
    if not text:
        return None
    try:
        p = Path(text)
        target = p if p.is_dir() else p.parent
        if str(target) in ("", "."):
            return None
        return os.path.normcase(os.path.abspath(str(target)))
    except Exception:
        return None


def work_hint(reg, media):
    """同目录的其他集打开时，作品名自动带出；没有记录 → None。"""
    if not isinstance(reg, dict):
        return None
    hints = reg.get("work_hints")
    if not isinstance(hints, dict):
        return None
    key = media_dir(media)
    if key is None:
        return None
    for stored, name in hints.items():
        if not isinstance(stored, str):
            continue
        try:
            same = os.path.normcase(os.path.abspath(stored)) == key
        except Exception:
            same = stored == key
        if same:
            return _as_str(name).strip() or None
    return None


def set_work_hint(root, media, work_name) -> dict:
    """记下「这个目录 = 这个作品名」，并顺手登记作品（面板要能列出它）。"""
    key = media_dir(media)
    name = _as_str(work_name).strip()
    if key is None or not name:
        return {"ok": False, "reason": "bad_media_or_name"}
    with _SAVE_LOCK:
        reg = load_registry(root)
        reg["work_hints"][key] = name
        # 顺手建一条作品条目（面板要能列出这个作品，哪怕还没入库任何角色）
        _work_rec(reg, name)
        save_registry(root, reg)
    return {"ok": True, "dir": key, "work_name": name}


# --------------------------------------------------------------------------
# 阈值体检（纯计算）
# --------------------------------------------------------------------------

def health_check(root=None, vector=None) -> dict:
    """把库里每条 `page__*__segNNN` 向量两两算余弦，报告两个数：

    - `same_min`：**同角色**最低分（越低说明这个角色跨集越不稳）
    - `diff_max`：**不同角色**最高分（越高说明越容易认错人）

    只有 `same_min` 明显高于 `diff_max`（`gap > 0`）阈值才有依据。
    `ok` = 两者都有值且 gap > 0；`suggested` 给出一个可用的阈值建议（没有间隔时为 None）。

    只统计 `segNNN`：`centroid` 是它们的加权混合、`job__*` 是同一批聚类的副本，
    把它们算进去会让「同角色最低分」自我抬高，体检就没意义了。
    样本较少时，这些统计量只反映当前素材。

    给了 `vector` 就走另一条路：拿它跟每个角色的 seg 向量比，返回按分数排序的
    `{"clusters": [{"page", "score", "n_vectors"}, ...], "top": ...}`（诊断单条向量用）。
    """
    reg = load_registry(root)
    model = reg["embedding_model"]
    want = space_id(model)
    vecs = load_vectors(root)
    # 键里第 2 段是作品名、第 3 段是角色名。只有「该角色在当前空间下可用」的
    # seg 才参与体检 —— 换了模型的旧向量跟新质心不是一个空间，混在一起算余弦
    # 没有意义（判据与 `_page_vectors` 同一套，避免两处规则漂移）。
    works = reg["works"] or {}
    work_ok = {work: _work_vectors_usable(wrec, model) for work, wrec in works.items()}
    by_page = {}
    for key in vecs:
        ident = _key_ident(key)
        if ident is None:
            continue
        work, page, row_space = ident
        seg = _segment_key_index(key)          # **只**看 segNNN，见上方 docstring
        if seg is None:
            parked = _parked_segment_of_key(key)
            if parked is None:
                continue
            seg = parked
        wrec = works.get(work) or {}
        if not _row_participates(work_ok.get(work, True), _page_space(wrec, page),
                                 row_space, want):
            continue
        by_page.setdefault(page, []).append((seg[2], key, vecs[key]))
    for rows in by_page.values():
        rows.sort(key=lambda r: r[0])

    pages = sorted(by_page)
    same_min = None
    same_min_page = ""
    diff_max = None
    diff_pair = None
    n_same = 0
    n_diff = 0
    for i, a in enumerate(pages):
        rows_a = by_page[a]
        # 从 0 起、到 len(rows_a) 止 —— 这里的下标是**行**下标，与页下标 i 无关。
        # 误写成 range(i + 1, len(rows_a)) 时，角色多于页数就会漏掉同角色对
        # （A 有 3 条时只剩 2 对，跨角色切片也跟着错），同角色最低分会算错。
        for j in range(1, len(rows_a)):
            for t in range(j):
                c = cosine(rows_a[j][2], rows_a[t][2])
                if c is None:
                    continue
                n_same += 1
                if same_min is None or c < same_min:
                    same_min = c
                    same_min_page = a
        for b in pages[i + 1:]:
            for _ia, _ka, va in rows_a:
                for _ib, _kb, vb in by_page[b]:
                    c = cosine(va, vb)
                    if c is None:
                        continue
                    n_diff += 1
                    if diff_max is None or c > diff_max:
                        diff_max = c
                        diff_pair = [a, b]

    if vector is not None:
        unit = normalise_vector(vector)
        if unit is None:
            return {"ok": False, "reason": "bad_query_vector"}
        hits = []
        for page, rows in by_page.items():
            score = top_mean(unit, [r[2] for r in rows])
            if score is not None:
                hits.append({"page": page, "score": score, "n_vectors": len(rows)})
        hits.sort(key=lambda h: (-h["score"], h["page"]))
        return {"ok": True, "clusters": hits, "top": hits[0] if hits else None}

    gap = None
    suggested = None
    overlap = None
    if same_min is not None and diff_max is not None:
        gap = float(same_min - diff_max)
        overlap = gap <= 0.0
        if gap > 0.0:
            auto = diff_max + 0.5 * min(gap, 0.4)
            suggested = {"auto": round(auto, 4),
                         "suggest": round(max(0.0, auto - 0.10), 4)}
    return {
        "ok": same_min is not None and diff_max is not None and gap > 0.0,
        "n_pages": len(pages),
        "n_segments": sum(len(v) for v in by_page.values()),
        "n_pairs_same": n_same,
        "n_pairs_diff": n_diff,
        "same_min": same_min,
        "same_min_page": same_min_page,
        "diff_max": diff_max,
        "diff_pair": diff_pair,
        "gap": gap,
        "overlap": overlap,
        "suggested": suggested,
        "thresholds": reg["thresholds"],
        "model": model,
        # 空间 id → 模型标识：报告的这些数只统计了**当前空间**的行，把表带出来
        # 才能看出「另一个模型下的历史还留在盘上」（只为人眼，不是判据）。
        "spaces": dict(reg.get("spaces") or {}),
        "known_limit": "样本较少时，请补充更多素材后再比较分布。",
    }
