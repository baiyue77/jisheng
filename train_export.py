#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""训练集切分（设计 §8）与导出（§6.3 / §8 / §9）：句子 + 标注状态 -> 训练集。

本模块分两层：

* **上半部分是纯逻辑**（`plan_clips` / `is_exportable` / 各 bucket 常量）：不 open、
  不 Path 写、不调 ffmpeg。切分计划算出来之后，落盘由下半部分负责。
  这一层的纯净是有意的：审查者用 4000 组随机输入 + deepcopy 比对验过它不改入参，
  也让它能完全脱离音频单测。
* **下半部分是导出**（Task 2）：真实 ffmpeg `silencedetect` / 切分 / trim、峰值归一化、
  文件名净化、`.list` 原子写、导出报告。ffmpeg 与磁盘是这里唯一的副作用来源。

导出的三条源码级硬约束（GPT-SoVITS，设计 §9）：
1. `.list` 文本里不能有 `|`（`line.split("|")` 必须恰好 4 段）
2. 必须 LF：`open(..., newline="\n")`，**绝不用 `Path.write_text()`**（Windows 会写 CRLF）
3. 语言字段固定 `JA`

处理顺序（这个顺序是有意的，不是随手排的）：

1. **归类**：每个句子先按 `speaker_labels.effective_speaker` 定角色名
   （含单句改判）；聚类没起名（`effective_speaker` 会退回 `SPEAKER_00` 这种聚类键）、
   名字为空、或 `?` 的，一律归 `_未命名`。
2. **拆长**：单句 > `split_over=12 s` 的，在 `silence_finder` 报出的每个内部静音处切开
   （切点取静音段中点，两侧各留半个静音给 Task 2 trim；右边剩不到
   `split_over / 2 = 6 s` 的静音不算切点，免得切出碎尾巴）。
   必须排在合并之前 —— 反过来的话，合并上限 10 s 会先动过时间轴，
   12 s 这条拆长规则就落在错误的边界上。
3. **合并**：同角色、相邻、间隔 < `merge_gap=0.3 s`，
   且合并后的**跨度**不超过 `merge_max=10 s`。边界取「≤ 10 s 允许，> 10 s 不并」：
   上限本身是允许达到的，只有**超过**才断开（与「单句 > 12 s」同一读法）。
   判据只有**时间间隔**：空文本段在第 1 步就被整条丢弃，不参与、也不构成屏障
   （它是 ASR 的伪边界，拿来当屏障会切出碎条再被 `_短句` 丢掉）。
   注意 span 判据是跨度（首句 start 到末句 end），不是各句时长之和 ——
   中间那点 0.05 s 的空隙也算进素材长度，不算进去就会出现 10.2 s 的产物。
   合并出来的片段再走一遍拆长？**不**：合并上限 10 s < 12 s，合并产物不会超长。
4. **弃短**：最终时长 < `drop_short=2 s` 的整条归 `_短句`（边界取「正好 2 s 保留」，
   与设计「< 2 秒丢弃」一致），不进 `.list`；时间区间原样保留，便于追溯。

`silence_finder(start, end) -> [(s, e), ...]` 是注入点：返回该区间内的静音段
（升序、互不重叠）。真实的 ffmpeg `silencedetect` 实现留给 Task 2，
所以本模块可以完全脱离音频单测。
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import speaker_labels as sl
from asr_languages import LANGUAGES

#: 最终时长不足的片段归这里，不进 `.list`
SHORT_BUCKET = "_短句"
#: 没起名字的聚类 / `?` / 无说话人 归这里，不进 `.list`
UNNAMED_BUCKET = "_未命名"

DEFAULT_MERGE_GAP = 0.3
DEFAULT_MERGE_MAX = 10.0
DEFAULT_SPLIT_OVER = 12.0
DEFAULT_DROP_SHORT = 2.0

_EPS = 1e-9


def _norm_text(text) -> str:
    """片段文本。非字符串当空处理（speakers.json 是给人手改的，什么都可能出现）。"""
    return text.strip() if isinstance(text, str) else ""


def _speaker_of(data: dict, index: int, seg: dict) -> str:
    """角色名或 bucket 名。空 / `?` / None / 只有聚类键没起名 一律 _未命名。

    `effective_speaker` 在聚类没起名时会退回聚类键（`SPEAKER_01`）—— 那是给字幕打的
    「说话人N」，不是角色名，导出时必须归 `_未命名`，绝不能用它建角色目录。
    所以先判原始聚类键有没有名字，再取角色名。

    保留名（`_短句` / `_未命名`）也归 `_未命名`：角色名撞上 bucket 名时
    （用户手打出来的），`_ClipBuilder.build` 里 `bucket = self.speaker` 会让它
    冒充 bucket —— 片段被标 `short=True`、落 `_短句` 目录、又被 `is_exportable`
    挡在 `.list` 外，用户看到的是「写了 1 条、list 0 条、原因 short」。
    """
    cluster = (data.get("segment_overrides") or {}).get(str(index), seg.get("speaker"))
    if not cluster or cluster == "?":
        return UNNAMED_BUCKET
    if sl.cluster_name(data, cluster) is None:
        return UNNAMED_BUCKET
    name = sl.effective_speaker(data, index, seg)
    if not isinstance(name, str):
        return UNNAMED_BUCKET
    name = name.strip()
    if not name or name == "?" or name in (SHORT_BUCKET, UNNAMED_BUCKET):
        return UNNAMED_BUCKET
    return name


def _segment_texts(seg_start: float, text: str, span: float, cuts) -> list:
    """把一个长句的文本按切点位置分给各段：按时间比例切字符，一个字都不丢。

    只按比例定位置（不做形态素分析），所以分配点落在字与字之间即可 ——
    这样「不丢字」是构造上成立的，而不是靠事后拼接补回来的。
    """
    n = len(cuts) + 1
    if span <= 0 or n == 1:
        return [text] + [""] * (n - 1)

    ratios = [0.0]
    for cut in cuts:
        ratios.append(min(1.0, max(0.0, (cut - seg_start) / span)))
    ratios.append(1.0)
    total = len(text)
    bounds = [int(round(r * total)) for r in ratios]
    for i in range(1, len(bounds)):                 # 保证边界单调，字符不回流
        bounds[i] = max(bounds[i], bounds[i - 1])
    return [text[bounds[i]:bounds[i + 1]] for i in range(n)]


def _silences_in(silence_finder, start: float, end: float) -> list:
    """取内部静音段：裁到句子范围内，丢掉退化的，按开始时间排序。"""
    if silence_finder is None:
        return []
    try:
        found = silence_finder(start, end) or []
    except Exception:                                # noqa: BLE001 - 探测失败就不能切
        return []
    out = []
    for item in found:
        try:
            s, e = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        s, e = max(s, start), min(e, end)
        if e > s + _EPS:
            out.append((s, e))
    out.sort()
    return out


def _cut_points(seg_start: float, seg_end: float, silences, split_over: float) -> list:
    """算切点：静音段中点；只保留「右边那截仍值得单独成条」的中点。

    取中点而不是静音边界：静音段两侧各留半个静音，Task 2 再 trim ——
    既不会把大段静音塞进产物，也不会切到字上。

    「值得单独成条」= 右边那截 >= `split_over / 2`（6 s）。这条下限是必要的：
    14 s 的句子只有一处 7 s 处的静音时，从后往前按 `> split_over` 判会一刀不切，
    把整条 14 s 原样留下；按下限判则切成 7 + 7，符合「> 12 s 就按静音切开」。
    同时它挡住贴着句尾的静音把句子切出一个碎尾巴。

    下限取「>=」而不是「>」：右段正好 6 s 是**值得单独成条**的
    （13 s + 中点 7.0 → 7 + 6 两条都对），按 `>` 判会把它整体留下，
    产出一条 13 s 且 `over_limit=True` 的片段 —— 那正是这条规则要避免的。
    """
    floor = split_over / 2.0
    cuts = []
    for s, e in silences:
        mid = (s + e) / 2.0
        if not (seg_start + _EPS < mid < seg_end - _EPS):
            continue
        if seg_end - mid >= floor - _EPS:
            cuts.append(mid)
    return sorted(cuts)


def _split_one_segment(start: float, end: float, index: int, text: str, speaker: str,
                       silence_finder, split_over: float) -> list:
    """拆一条超长句子。"""
    silences = _silences_in(silence_finder, start, end)
    cuts = _cut_points(start, end, silences, split_over)
    bounds = [start] + cuts + [end]
    texts = _segment_texts(start, text, end - start, cuts)
    out = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e <= s + _EPS:
            continue
        out.append({"speaker": speaker, "index": index,
                    "start": s, "end": e,
                    "text": texts[i] if i < len(texts) else "",
                    "pieces": [index],
                    "over_limit": (e - s) > split_over + _EPS})
    return out


def _pieces_from_segments(segments, data, silence_finder, split_over) -> list:
    pieces = []
    for index, seg in enumerate(segments or ()):
        if not isinstance(seg, dict):
            continue
        text = _norm_text(seg.get("text"))
        if not text:
            # 空文本段不产出片段，也**不构成合并屏障**：它被整条丢弃，
            # 相邻同角色片段只按时间间隔判（空文本段是 ASR 的伪边界，不是真停顿）。
            # 拿它当屏障会把后半句切成碎条、随后被 `_短句` 丢掉 —— 白扔素材。
            continue
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (end > start):
            continue
        speaker = _speaker_of(data, index, seg)
        if end - start > float(split_over) + _EPS:
            pieces.extend(_split_one_segment(start, end, index, text, speaker,
                                             silence_finder, float(split_over)))
        else:
            pieces.append({"speaker": speaker, "index": index, "seq": len(pieces),
                           "start": start, "end": end, "text": text,
                           "pieces": [index], "over_limit": False})
    pieces.sort(key=lambda p: (p["start"], p["end"]))
    for seq, piece in enumerate(pieces):
        piece["seq"] = seq
    return pieces


class _ClipBuilder:
    """攒一条正在合并的片段：起点固定，每并入一段就往后延 end。"""

    def __init__(self, piece: dict):
        self.speaker = piece["speaker"]
        self.start = piece["start"]
        self.end = piece["end"]
        self.texts = [piece["text"]] if piece["text"] else []
        self.indices = list(piece["pieces"])
        self.over_limit = bool(piece.get("over_limit"))

    def add(self, piece: dict) -> None:
        self.end = piece["end"]
        if piece["text"]:
            self.texts.append(piece["text"])
        self.indices.extend(piece["pieces"])
        self.over_limit = self.over_limit or bool(piece.get("over_limit"))

    def build(self, drop_short: float) -> dict:
        dur = self.end - self.start
        bucket = SHORT_BUCKET if dur < drop_short - _EPS else self.speaker
        clip = {"speaker": bucket, "index": self.indices[0],
                "start": self.start, "end": self.end, "dur": dur,
                "text": " ".join(self.texts), "pieces": list(self.indices)}
        if bucket == SHORT_BUCKET:
            clip["short"] = True
        if self.over_limit:
            clip["over_limit"] = True
        return clip


def plan_clips(segments, labels, silence_finder=None, *, merge_gap=DEFAULT_MERGE_GAP,
               merge_max=DEFAULT_MERGE_MAX, split_over=DEFAULT_SPLIT_OVER,
               drop_short=DEFAULT_DROP_SHORT) -> list:
    """把句子 + 标注状态算成切分计划。

    segments  : segments.jsonl 的行，dict(start/end/text/speaker)；
                speaker 是聚类键（"SPEAKER_00" / "?" / None）。
    labels    : speaker_labels 的标注状态（clusters + segment_overrides）。
    silence_finder : (start, end) -> [(s, e), ...]；None = 没有静音信息（不拆长）。

    返回按 start 升序的 list[dict]，每项含：
      speaker  角色名 / `_短句` / `_未命名`（后两者不进 `.list`）
      index    最早那条原始句号（追溯用；拆出来的多段共用同一个 index）
      pieces   本条覆盖的所有原始句号
      start/end/dur/text
      short=True       该条归了 `_短句`
      over_limit=True  这一条**仍然超过** `split_over`（12 s）—— 拆无可拆（静音太疏 /
                       没有内部静音），Task 2 必须把它写进导出报告。
                       注意它不是「超过合并上限 10 s」：11 s、12 s 的单句本来就不该被拆，
                       不该报超长。
    """
    data = labels if isinstance(labels, dict) else {}
    pieces = _pieces_from_segments(segments, data, silence_finder, split_over)

    merge_gap = float(merge_gap)
    merge_max = float(merge_max)
    drop_short = float(drop_short)

    clips = []
    current = None
    for piece in pieces:
        if current is not None:
            gap = piece["start"] - current.end
            span = piece["end"] - current.start
            same = piece["speaker"] == current.speaker
            if same and gap < merge_gap - _EPS and span <= merge_max + _EPS:
                current.add(piece)
                continue
        if current is not None:
            clips.append(current.build(drop_short))
        current = _ClipBuilder(piece)
    if current is not None:
        clips.append(current.build(drop_short))

    return clips


def is_exportable(clip: dict) -> bool:
    """进 `.list` 的片段：有角色名、有文本，且不是 `_短句` / `_未命名`。"""
    if not isinstance(clip, dict):
        return False
    speaker = (clip.get("speaker") or "").strip()
    if not speaker or speaker in (SHORT_BUCKET, UNNAMED_BUCKET):
        return False
    return bool((clip.get("text") or "").strip())


# ===========================================================================
# Task 2：落盘、命名净化、`.list`、导出报告
# ===========================================================================

#: 默认采样率（设计 §8：32 kHz / 16-bit PCM）
DEFAULT_SAMPLE_RATE = 32000
#: 允许的采样率（设计 §8：默认 32 k，可切 44.1 k）
ALLOWED_SAMPLE_RATES = (32000, 44100)
#: `.list` 的语言字段（设计 §9 硬约束 3：固定 JA）
LIST_LANG = "JA"
GPT_SOVITS_LANGUAGES = ("ja", "zh", "en", "ko", "yue")
#: 台词部分在文件名里的上限（设计 §8）
MAX_TEXT_CHARS = 40
#: 整个文件名（不含扩展名）的上限（设计 §8）
MAX_STEM_CHARS = 80
#: `.list` 文本里的 `|` 换成全角竖线
FULLWIDTH_PIPE = "\uff5c"
LIST_SUFFIX = ".list"
EXPORT_FORMATS = ("gpt_sovits", "txt", "fish_speech", "indextts")
MANIFEST_SUFFIXES = {"gpt_sovits": LIST_SUFFIX, "txt": ".txt",
                     "fish_speech": "_fish_speech.txt", "indextts": "_indextts.jsonl"}
REPORT_SUFFIX = ".导出报告.json"
#: 峰值归一化目标（设计 §8：-1 dBFS）
PEAK_DBFS = -1.0
#: trim 判据：-40 dBFS 以下、峰值检测（不用 rms，短促起音会被均掉）
TRIM_THRESHOLD_DB = -40.0
#: trim 时首尾各保留一点点静音，避免削到起音/收尾
TRIM_PAD_S = 0.03
#: silencedetect 的默认判据
DEFAULT_NOISE_DB = -35.0
DEFAULT_MIN_SILENCE = 0.3
#: 导出前的磁盘预检：估算体积 * 安全系数 + 预留
DISK_SAFETY = 1.15
DISK_RESERVE_BYTES = 200 * 1024 * 1024
#: ffmpeg 可执行名（PATH 上）
FFMPEG = "ffmpeg"
_POLL_S = 0.2
_ILLEGAL_CHARS = frozenset('<>:"/\\|?*')
_WHITESPACE_RE = re.compile(r"\s+")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")
#: 未命名聚类的兜底目录名（`?` / 空聚类键）
UNNAMED_FALLBACK_LABEL = "未知"

#: `source_kind` 的 4 个合法取值（设计 §6.1）
SOURCE_KINDS = ("separated_hq", "separated_16k", "raw16k", "media")
_SOURCE_WARNINGS = {
    "separated_16k": "复用既有 16 kHz 输入分离出的人声轨：8 kHz 以上为空，克隆音色会发闷",
    "raw16k": "未分离，直接用 16 kHz 中间产物：高频已丢失且带 BGM",
    "media": "直接从原片切：未分离，带 BGM",
}


class ExportError(RuntimeError):
    """导出失败（音源缺失、采样率不合法、ffmpeg 报错……）。"""


class InsufficientSpace(ExportError):
    """目标盘剩余空间不够（设计 §12：导出前按估算体积预检）。"""


class ExportCancelled(ExportError):
    """用户中止导出。已经落盘的片段不完整，调用方应把产物当作无效。"""


def source_warning(kind) -> str:
    """`source_kind` 对应的面板警告；高质量源（`separated_hq`）没有任何警告。"""
    return _SOURCE_WARNINGS.get(kind)


def estimate_bytes(seconds, sr) -> int:
    """按 pcm_s16le 单声道估算体积（设计 §16 的 92 MB/集就是这个口径）。"""
    return int(round(max(0.0, float(seconds)) * int(sr) * 2))


def disk_free_bytes(path) -> int:
    """路径所在盘的剩余字节数（Windows 取盘符根）。"""
    p = Path(path)
    try:
        anchor = Path(p.resolve()).anchor
    except OSError:
        anchor = ""
    return int(shutil.disk_usage(anchor or str(p)).free)


# ---------------------------------------------------------------------------
# 静音检测：ffmpeg silencedetect
# ---------------------------------------------------------------------------

def parse_silencedetect(text) -> list:
    """解析 ffmpeg silencedetect 的 stderr，返回升序的 `[(start, end), ...]`。

    只认成对的 `silence_start:` / `silence_end:`。落单的 start（进程被杀、
    日志被截断）直接丢掉 —— 宁可少报一段静音，也不要编造一个区间。
    """
    spans = []
    pending = None
    for line in (text or "").splitlines():
        match = _SILENCE_START_RE.search(line)
        if match:
            try:
                pending = float(match.group(1))
            except ValueError:
                pending = None
            continue
        match = _SILENCE_END_RE.search(line)
        if match and pending is not None:
            try:
                end = float(match.group(1))
            except ValueError:
                pending = None
                continue
            if end > pending:
                spans.append((pending, end))
            pending = None
    spans.sort()
    return spans


def _run_ffmpeg(cmd, should_stop=None):
    """跑一次 ffmpeg，返回 `(returncode, stderr_text)`。

    用 `Popen` 而不是 `subprocess.run`：导出是长耗时动作，必须能被 `should_stop`
    中断（计划全局约束 3）。`stdin` 接 DEVNULL（配合 `-nostdin`，后台跑不会等 stdin）。
    """
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        try:
            _out, err = proc.communicate(timeout=_POLL_S)
            break
        except subprocess.TimeoutExpired:
            if should_stop is not None and should_stop():
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:                    # noqa: BLE001 - 已经杀掉了
                    pass
                raise ExportCancelled("导出已被停止（ffmpeg）")
    return proc.returncode, (err or b"").decode("utf-8", errors="replace")


def run_silencedetect(src, *, noise_db=DEFAULT_NOISE_DB, min_silence=DEFAULT_MIN_SILENCE,
                      should_stop=None) -> list:
    """对整条音源跑一次 silencedetect，返回升序静音段。

    注意**不能**加 `-loglevel error`：silencedetect 的结果就是 info 级别的日志。
    """
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-i", str(src),
           "-af", "silencedetect=noise=%gdB:d=%g" % (float(noise_db), float(min_silence)),
           "-f", "null", "-"]
    code, err = _run_ffmpeg(cmd, should_stop=should_stop)
    spans = parse_silencedetect(err)
    if code != 0 and not spans:
        raise ExportError("ffmpeg silencedetect 失败（退出码 %s）：%s"
                          % (code, err.strip()[-300:]))
    return spans


class FfmpegSilenceFinder:
    """`plan_clips` 的默认 silence_finder：整条音源只跑一次 silencedetect 并缓存。"""

    def __init__(self, src, *, noise_db=DEFAULT_NOISE_DB,
                 min_silence=DEFAULT_MIN_SILENCE, should_stop=None):
        self.src = Path(src)
        self.noise_db = float(noise_db)
        self.min_silence = float(min_silence)
        self.should_stop = should_stop
        self._spans = None

    def spans(self) -> list:
        if self._spans is None:
            self._spans = run_silencedetect(self.src, noise_db=self.noise_db,
                                            min_silence=self.min_silence,
                                            should_stop=self.should_stop)
        return self._spans

    def __call__(self, start, end) -> list:
        lo, hi = float(start), float(end)
        out = []
        for s, e in self.spans():
            s, e = max(s, lo), min(e, hi)
            if e > s:
                out.append((s, e))
        return out


def make_silence_finder(src, **kw) -> FfmpegSilenceFinder:
    return FfmpegSilenceFinder(src, **kw)


class _SilenceRecorder:
    """包一层：silencedetect 失败时记下原因、退化成「没有静音信息」，但不静默。"""

    def __init__(self, finder):
        self.finder = finder
        self.error = ""

    def __call__(self, start, end):
        try:
            return self.finder(start, end)
        except ExportCancelled:
            raise
        except Exception as exc:                     # noqa: BLE001 - 探不到静音不该中断导出
            self.error = "%s: %s" % (type(exc).__name__, exc)
            return []


# ---------------------------------------------------------------------------
# 文件名净化（设计 §8）
# ---------------------------------------------------------------------------

def sanitize_component(value) -> str:
    """Windows 非法字符与控制字符 -> `_`；再去掉结尾的点与空格。

    半角 `?` 必须替换（它是非法字符），全角 `？`、`・`、`…` 原样保留 ——
    它们在 Windows 上合法，日文台词里又很常见。
    """
    text = value if isinstance(value, str) else ""
    out = []
    for ch in text:
        if ch in _ILLEGAL_CHARS or ord(ch) < 32 or ord(ch) == 127:
            out.append("_")
        else:
            out.append(ch)
    component = "".join(out).strip().rstrip(". ")
    device_name = component.split(".", 1)[0].rstrip().upper()
    if re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³]", device_name):
        return "_" + component
    return component


def _filename_part(value) -> str:
    """文件名里的一个组成部分：先把换行/制表/多空格压成单空格，再净化。"""
    text = value if isinstance(value, str) else ""
    return sanitize_component(_WHITESPACE_RE.sub(" ", text).strip())


def _component(value) -> str:
    """角色名 -> 「目录名 == `.list` 说话人字段」共用的那一个字符串。

    空净化结果退回 `UNNAMED_FALLBACK_LABEL`：`.` / `..` / 全非法字符的名字
    净化后是空串，拿它当目录名会落到上一级目录（`Path("")`）或直接抛。
    """
    return sanitize_component(value) or UNNAMED_FALLBACK_LABEL


def make_stem(role, text, suffix="") -> str:
    """`<角色名>_<台词>`：台词截到 40 字符，整名（含后缀）不超过 80 字符。

    超长时**只砍台词**，角色名尽量留着 —— 文件名是给人看的，「这是谁」比
    「说了什么」重要。角色名自己就超长时才轮到砍它。
    """
    limit = MAX_STEM_CHARS - len(suffix or "")
    if limit < 1:
        return (suffix or "")[:MAX_STEM_CHARS]
    role = _filename_part(role)[:limit]
    text = _filename_part(text)[:MAX_TEXT_CHARS]
    keep = limit - len(role) - 1
    if keep < 0:
        role = role[:limit]
        keep = limit - len(role) - 1
    text = text[:keep] if keep >= 1 else ""
    stem = ("%s_%s" % (role, text)) if text else role
    return (stem[:limit] if len(stem) > limit else stem) + (suffix or "")


def assign_names(entries) -> None:
    """就地为每个 entry 写 `stem` / `name`，同一目录内撞名追加 `_2`、`_3`。

    去重顺序按 `(index, start, end)`：拆长后多段共用同一个 `index`（接缝第 1 条），
    只看 index 会把它们判成同一条；带上 start 顺序才是确定的、重复导出可复现的。
    比较用 `casefold()`：Windows 文件名不分大小写，`Al_x` 与 `AL_x` 是同一个文件。
    """
    used = set()
    ordered = sorted(entries, key=lambda e: (int(e["index"]), float(e["start"]),
                                             float(e["end"])))
    for entry in ordered:
        directory = str(entry["dir"]).casefold()
        n = 1
        while True:
            stem = make_stem(entry["name_part"], entry["text"],
                             suffix="" if n == 1 else "_%d" % n)
            key = (directory, stem.casefold())
            if key not in used:
                break
            n += 1
            if n > 10000:
                raise ExportError("文件名去重失败（同目录 10000 个同名片段）：%r"
                                  % (entry["name_part"],))
        used.add(key)
        entry["stem"] = stem
        entry["name"] = stem + ".wav"


# ---------------------------------------------------------------------------
# `.list` 与原子写
# ---------------------------------------------------------------------------

def _escape_field(value) -> str:
    """`.list` 字段的转义：换行 -> 空格，`|` -> 全角竖线（硬约束 1）。"""
    text = value if isinstance(value, str) else ""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return text.replace("|", FULLWIDTH_PIPE).strip()


def list_line(wav_path, speaker, text, language=LIST_LANG) -> str:
    """`.list` 的一行：`<wav>|<说话人>|<语言>|<文本>`。"""
    language = str(language).upper()
    if language.lower() not in GPT_SOVITS_LANGUAGES:
        raise ValueError('GPT-SoVITS 训练集支持 JA / ZH / EN / KO / YUE')
    return "|".join((str(wav_path), _escape_field(speaker), language,
                     _escape_field(text)))


def _atomic_write_lf(path, text) -> None:
    """LF 原子写（临时文件 + `os.replace`）。

    **必须 `newline="\\n"`**：Windows 上文本模式的默认行为会把 `\\n` 翻成 `\\r\\n`，
    而 GPT-SoVITS 是 `f.read().strip("\\n").split("\\n")`，CRLF 会在文本字段尾部留 `\\r`。
    同理**绝不能**用 `Path.write_text()`（它没有 newline 参数，走的就是默认翻译）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.%d.%d.tmp" % (path.name, os.getpid(), threading.get_ident()))
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_list(path, rows, language=LIST_LANG) -> int:
    """写 `.list`，返回行数。rows = `[(wav_path, speaker, text), ...]`。

    写之前逐行自校验「恰好 4 段」：与其写出一份 GPT-SoVITS 读不了的文件，
    不如当场报错（这是三条硬约束里最容易被转义漏掉的一条）。
    """
    lines = []
    for row in rows:
        wav_path, speaker, text = row[0], row[1], row[2]
        line = list_line(wav_path, speaker, text, language)
        if len(line.split("|")) != 4:
            raise ExportError("`.list` 行不是恰好 4 段（GPT-SoVITS 会解析失败）：%r"
                              % (line,))
        lines.append(line)
    _atomic_write_lf(path, "".join(line + "\n" for line in lines))
    return len(lines)


def read_list_lines(path) -> list:
    """按 GPT-SoVITS 的读法读回 `.list`（`strip("\\n").split("\\n")`）。"""
    raw = Path(path).read_bytes()
    return raw.decode("utf-8").strip("\n").split("\n")


# ---------------------------------------------------------------------------
# 音频：切分、trim、峰值归一化
# ---------------------------------------------------------------------------

def _num(value) -> str:
    return "%g" % float(value)


def trim_filter(threshold_db=TRIM_THRESHOLD_DB, pad_s=TRIM_PAD_S) -> str:
    """首尾按静音 trim 到语音起止的 ffmpeg 滤镜串。

    尾部没法直接 trim，所以「正放 trim 开头 -> 倒放 -> 再 trim 开头 -> 倒回来」。
    `detection=peak` 用峰值而不是 rms：rms 会把短促的起音平均掉，等于切字。
    首尾各留 `pad_s` 秒静音，进一步保住起音/收尾。
    """
    one = ("silenceremove=start_periods=1:start_duration=0:start_silence=%s:"
           "start_threshold=%s:detection=peak" % (_num(pad_s), _num(threshold_db) + "dB"))
    return ",".join((one, "areverse", one, "areverse"))


def _read_wav(path):
    with wave.open(str(path), "rb") as fh:
        return (fh.getnchannels(), fh.getsampwidth(), fh.getframerate(),
                fh.getnframes(), fh.readframes(fh.getnframes()))


def wav_seconds(path):
    """WAV 时长（秒）；读不出来返回 None。"""
    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            return fh.getnframes() / float(rate) if rate else 0.0
    except Exception:                                # noqa: BLE001 - 坏文件当读不出来
        return None


def _clamp16(value) -> int:
    if value < -32768:
        return -32768
    if value > 32767:
        return 32767
    return int(value)


def normalize_peak_wav(path, target_dbfs=PEAK_DBFS):
    """按峰值把 WAV 线性缩放到 `target_dbfs`，返回缩放后的峰值 dBFS。

    刻意**不用 `loudnorm`**（设计 §8）：它按感知响度做动态处理，会压动态、改音色。
    峰值归一化只是乘一个常数增益，波形形状一点不动 —— 所以放在 Python 里做：
    不必为每条片段多起一个 ffmpeg 进程，也不必两遍分析。全静音返回 None。
    """
    channels, width, rate, frames, data = _read_wav(path)
    if width != 2 or frames <= 0:
        return None
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(max(samples), -min(samples))
    if peak <= 0:
        return None
    target = (10.0 ** (float(target_dbfs) / 20.0)) * 32767.0
    gain = target / float(peak)
    if abs(gain - 1.0) > 1e-6:
        samples = array.array("h", (_clamp16(round(s * gain)) for s in samples))
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(channels)
            fh.setsampwidth(width)
            fh.setframerate(rate)
            fh.writeframes(samples.tobytes())
    new_peak = max(max(samples), -min(samples))
    return 20.0 * math.log10(new_peak / 32768.0)


def cut_segment(src, start, dur, dest, *, sr=DEFAULT_SAMPLE_RATE, trim=True,
                trim_threshold_db=TRIM_THRESHOLD_DB, should_stop=None):
    """切一段训练音频：`-ss/-t` 定位 -> 首尾 trim -> 32 k 单声道 s16le -> 峰值归一化。

    `-ss` 必须在 `-i` **前**（快速定位）；`-nostdin` 必须有，否则后台跑会等 stdin。
    先写同目录临时名再 `os.replace`：导出中途被打断也不会留下半个 WAV 冒充成品。
    临时名**必须保留 `.wav` 结尾**，否则 ffmpeg 按扩展名选不到 muxer。
    `trim_threshold_db` 由 `export_work` 按片注入（默认 -40 dBFS）。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("%s.%d.%d.tmp.wav" % (dest.stem, os.getpid(),
                                               threading.get_ident()))
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", "%.3f" % float(start), "-t", "%.3f" % float(dur),
           "-i", str(src)]
    if trim:
        cmd += ["-af", trim_filter(threshold_db=trim_threshold_db)]
    cmd += ["-ac", "1", "-ar", str(int(sr)), "-c:a", "pcm_s16le", str(tmp)]
    try:
        code, err = _run_ffmpeg(cmd, should_stop=should_stop)
        if code != 0:
            raise ExportError("ffmpeg 切分失败（退出码 %s）：%s"
                              % (code, err.strip()[-300:]))
        normalize_peak_wav(tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return dest


# ---------------------------------------------------------------------------
# 片段身份：回到同一份 segments + labels 重算（接缝第 3 条）
# ---------------------------------------------------------------------------

def unnamed_label(cluster) -> str:
    """`_未命名` 的子目录名：`SPEAKER_02` -> `说话人3`（与字幕里的 `[说话人N]` 一致）。"""
    if not cluster or cluster == "?":
        return UNNAMED_FALLBACK_LABEL
    if isinstance(cluster, str) and cluster.upper().startswith("SPEAKER_"):
        try:
            return "说话人%d" % (int(cluster.split("_", 1)[1]) + 1)
        except ValueError:
            pass
    return sanitize_component(cluster) or UNNAMED_FALLBACK_LABEL


def _describe_clip(clip, segments, data) -> dict:
    """把一条 clip 落成「放哪个目录、叫什么、进不进 `.list`」。

    clip 里**没带**角色名与聚类键（`_短句`/`_未命名` 的 `speaker` 已被 bucket 覆盖，
    接缝第 3 条），所以这里用 `clip["index"]` 回到同一份 segments + labels 重算，
    并且用与 `plan_clips` **同一个** `_speaker_of`，判据不会漂移。

    目录名与 `.list` 的说话人字段**必须来自同一个净化结果**（`_component`）：
    两边各净化一次的话，角色名里的 `|` 会变成目录 `甲_乙`、字段 `甲｜乙` ——
    GPT-SoVITS 切片时按字段去对目录，两边名字对不上就等于这条素材丢了。
    """
    index = clip.get("index")
    role = None
    label = None
    if isinstance(index, int) and 0 <= index < len(segments):
        seg = segments[index]
        if isinstance(seg, dict):
            name = _speaker_of(data, index, seg)
            if name != UNNAMED_BUCKET:
                role = name
            else:
                cluster = (data.get("segment_overrides") or {}).get(
                    str(index), seg.get("speaker"))
                label = unnamed_label(cluster)
    if role is None and label is None:
        # segments.jsonl 被改过、index 对不上了：能落盘，但不冒充某个角色
        label = UNNAMED_FALLBACK_LABEL

    short = bool(clip.get("short"))
    # 目录名与 `.list` 字段共用的净化结果：算一次，两边都用它（见上面的文档）。
    component = _component(role) if role is not None else _component(label)
    if role is None:
        # 未命名优先：一条 1.5 s 的未命名片段在 plan_clips 里已被标 `short`，
        # 但它没有角色名，只能进 `_未命名\<标签>\`。
        # `report["short_clips"]` 数的是**计划里路由到 `_短句\` 的条数**
        # （即「有角色名 + 短」），与片段是否真落盘无关：`drop_short=True` 时
        # `_短句\` 一条都不写，这个数照样不变；未命名的短片段路由到 `_未命名\`，
        # 也不算在里面。想数「被 drop_short 丢掉的短句」请看 `short_skipped`。
        subdir = [UNNAMED_BUCKET, _component(label)]
        name_part = label
        bucket = "unnamed"
    elif short:
        subdir = [SHORT_BUCKET]
        name_part = role
        bucket = "short"
    else:
        # 同一个字符串既当目录名又当 `.list` 字段（见上面的文档）
        subdir = [component]
        name_part = role
        bucket = "named"

    text = (clip.get("text") or "").strip() if isinstance(clip.get("text"), str) else ""
    return {
        "index": int(index) if isinstance(index, int) else -1,
        "start": float(clip.get("start") or 0.0),
        "end": float(clip.get("end") or 0.0),
        "dur": float(clip.get("dur") or 0.0),
        "text": text,
        "role": role,
        "label": label,
        "bucket": bucket,
        "subdir": subdir,
        "dir": "/".join(subdir),
        "name_part": name_part,
        "short": short,
        "over_limit": bool(clip.get("over_limit")),
        # 说话人字段是纯角色名；角色名认不出来时退回聚类标签，绝不写空字段
        "list_speaker": _escape_field(component),
        "eligible": is_exportable(clip),
    }


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------

def _rebuild_work_dir(out_root, work_dir) -> None:
    """按角色目录重建：整棵 `<作品名>` 目录删掉重来，不残留旧名字的目录。

    删之前先证明目标确实在导出根目录**里面**（`..`、空作品名、盘符根都能被这一步挡住）。
    """
    root = Path(out_root).resolve()
    target = Path(work_dir).resolve()
    if target == root or root not in target.parents:
        raise ExportError("拒绝删除导出根目录之外的路径：%s" % (target,))
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def cleanup_partial_export(out_root, work_name) -> dict:
    """删掉一次**被中止**的导出留在导出根里的那棵半截音频目录。

    为什么可以删：`export_work` 是「先删旧 `.list`/旧报告，再整棵重建
    `<作品名>\\`」，所以中断时留下的那棵目录**只有**半截 wav，没有任何
    `.list` 指向它（GPT-SoVITS 拿到也不会读它）。

    为什么敢删：
    * 只在 `<out_root>\\<净化后的作品名>\\` 这一棵里动手，别的作品目录一字不碰；
    * **`.list` 还在就一个字节都不删** —— `.list` 存在意味着这棵目录（可能是
      上一次成功导出的）仍被引用。中断发生在「删旧 `.list`」之前时就是这种
      情况（例如预检/准备音源阶段被停），此时旧产物必须原封不动；
    * 目标必须**严格在导出根里面**（与 `_rebuild_work_dir` 同一判据）：
      `..`、空作品名、盘符根都被这一步挡住。

    返回 `{"removed": bool, "code": str, "path": str, "freed_mb": float,
    "reason": str}`；`code` 是给调用方分支用的（`removed` / `no_dir` /
    `list_present` / `outside_root` / `empty_name` / `error`），`reason` 是
    写进日志的人话 —— 别让用户以为清了其实没清。
    """
    work = sanitize_component(work_name or "")
    root = Path(out_root)
    result = {"removed": False, "code": "", "path": "", "freed_mb": 0.0,
              "reason": ""}
    if not work:
        result["code"] = "empty_name"
        result["reason"] = "作品名为空，无法定位本次导出的目录"
        return result

    target = root / work
    result["path"] = str(target)
    try:
        resolved_root = root.resolve()
        resolved = target.resolve()
    except OSError as exc:                           # noqa: BLE001 - 定位不了就别删
        result["code"] = "error"
        result["reason"] = "路径无法解析：%s" % exc
        return result
    if resolved == resolved_root or resolved_root not in resolved.parents:
        result["code"] = "outside_root"
        result["reason"] = "目标不在导出根之内，拒绝删除"
        return result
    for suffix in MANIFEST_SUFFIXES.values():
        manifest = root / (work + suffix)
        if manifest.exists():
            result["code"] = "list_present"
            result["reason"] = "`%s` 还在，这棵目录可能仍被引用" % manifest.name
            return result
    if not resolved.exists():
        result["code"] = "no_dir"
        result["reason"] = "本次导出没有在导出根里留下目录"
        return result

    freed = 0
    for path in resolved.rglob("*"):
        try:
            if path.is_file():
                freed += path.stat().st_size
        except OSError:
            continue
    try:
        shutil.rmtree(resolved)
    except OSError as exc:                           # noqa: BLE001 - 删不掉不算致命
        result["code"] = "error"
        result["reason"] = "删除失败：%s" % exc
        return result
    result["removed"] = True
    result["code"] = "removed"
    result["freed_mb"] = round(freed / 1e6, 3)
    return result


def _read_previous(report_path, list_path) -> dict:
    """覆盖前的情况：上一次报告里的片段数 + 现有 `.list` 的行数。"""
    previous = {"report_exists": False, "clip_count": 0, "list_lines": 0}
    count = None
    if Path(report_path).is_file():
        try:
            data = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))
        except Exception:                            # noqa: BLE001 - 坏报告不能挡住导出
            data = None
        if isinstance(data, dict):
            previous["report_exists"] = True
            for key in ("written_clips", "total_clips"):
                value = data.get(key)
                if isinstance(value, int):
                    count = value
                    break
    if Path(list_path).is_file():
        try:
            previous["list_lines"] = len([ln for ln in read_list_lines(list_path)
                                          if ln.strip()])
        except Exception:                            # noqa: BLE001
            previous["list_lines"] = 0
    previous["clip_count"] = int(count) if count is not None else previous["list_lines"]
    return previous


def plan_export_space(planned_seconds, sr, *, safety=DISK_SAFETY,
                      reserve=DISK_RESERVE_BYTES) -> int:
    """导出所需的磁盘字节数：估算体积 * 安全系数 + 预留。

    `export_work` 与 `precheck_work` 共用它，两边都用默认的 `safety`/`reserve`
    —— 预检口径必须和真正落盘时的口径一致，否则会出现「预检说够、落盘说不够」
    （或反过来白拦一次）。

    注意它只保证**同一个 `planned_seconds` 得到同一个字节数**；「预检算出来的
    `planned_seconds` 是不是落盘的那一个」由调用方负责：生产路径要把
    `precheck_work(...)["plan"]` 交给 `export_work(plan=...)`，别再算一遍。
    """
    return int(estimate_bytes(planned_seconds, sr) * float(safety)) + int(reserve)


def _nothing_listable_message(entries) -> str:
    """一个片段都进不了 `.list` 时的报错文案：说清原因，并给**真的有用**的下一步。

    刻意不给「取消勾选『丢弃短句』」这类建议：那个开关只决定短片段切不切，
    而 `is_exportable()` 里短句本来就不进 `.list` —— 勾不勾都一样。
    给一个按了也没用的建议，比不给建议更糟。
    """
    unnamed = sum(1 for e in entries if e.get("role") is None)
    textless = sum(1 for e in entries if not e.get("text"))
    short = sum(1 for e in entries if e.get("short"))
    reasons = []
    if unnamed:
        reasons.append("%d 个片段没有角色名" % unnamed)
    if textless:
        reasons.append("%d 个片段没有文本" % textless)
    if short:
        reasons.append("%d 个片段是短句（短句不进 `.list`）" % short)
    why = "、".join(reasons) if reasons else "没有任何可用片段"

    if unnamed:
        todo = "请先在面板上给角色命名，然后重试。"
    elif short and not textless:
        todo = "本集没有够长的句子，没有可导出的训练素材。"
    else:
        todo = "请检查这一集的识别结果后重试。"

    return ("没有任何片段能进训练集：%s。写出 0 字节的 `.list` 会让 GPT-SoVITS "
            "解包时报错，所以这次一个字节都没写。%s" % (why, todo))


def _plan_export(segments, labels, out_root, *, work_name=None, sr=None,
                 drop_short=False, silence_finder=None, should_stop=None,
                 export_format="gpt_sovits"):
    """**纯计划**：算出片段、命名、体积估算与各种路径，不写盘、不删盘。

    抽出来是为了让磁盘预检（`precheck_work`）走**同一条**计划路径：
    预检要是另写一套「大概估一下」，它拦下的东西和 export_work 拦下的东西
    就会慢慢漂开，最终变成「预检通过、导出当场失败」。

    刻意**不接音源路径**：计划本身不需要它（静音探测由调用方包好传进来）。
    少一个「其实用不到」的参数，就少一处「预检与落盘各用了不同音源」的隐患。

    `silence_finder` 为 None 时由调用方先包好（成功/失败/中断的语义都在
    `export_work` 里），这里只管用。
    """
    sr = int(sr) if sr is not None else DEFAULT_SAMPLE_RATE
    work = sanitize_component(work_name if work_name else sl.get_work_name(labels))
    if not work:
        raise ExportError("作品名为空，无法导出")

    out_root = Path(out_root)
    work_dir = out_root / work
    list_path = out_root / ("%s%s" % (work, LIST_SUFFIX))
    report_path = out_root / ("%s%s" % (work, REPORT_SUFFIX))

    segments = list(segments or ())
    labels = labels if isinstance(labels, dict) else {}
    previous = _read_previous(report_path, list_path)

    clips = plan_clips(segments, labels, silence_finder)
    entries = [_describe_clip(clip, segments, labels) for clip in clips]
    assign_names(entries)

    # 一个片段都进不了 `.list` → **在这里就拒绝**（计划阶段，早于任何写盘）。
    # 放在计划里而不是落盘处，是为了让磁盘预检也走同一道门：用户点「导出」
    # 得到的是一句能看懂的话，而不是几分钟后一个 0 字节的坏文件。
    if not any(e.get("eligible") for e in entries):
        raise ExportError(_nothing_listable_message(entries))

    planned = [e for e in entries if e["text"] and not (drop_short and e["short"])
               and (export_format != "fish_speech" or e["eligible"])]
    planned_seconds = round(sum(e["dur"] for e in planned), 3)
    estimated_bytes = estimate_bytes(planned_seconds, sr)
    return {"work": work, "out_root": out_root, "work_dir": work_dir,
            "list_path": list_path, "report_path": report_path,
            "segments": segments, "labels": labels, "previous": previous,
            "clips": clips, "entries": entries, "n_planned": len(planned), "format": export_format,
            "planned_seconds": planned_seconds, "estimated_bytes": estimated_bytes}


def plan_silence_finder(src, *, should_stop=None, silence_finder=None):
    """生产路径用的**唯一**静音探测器：同一个实例喂给预检与落盘。

    为什么要共用：`plan_clips` 的拆长判据完全取决于静音段。预检和落盘各建一个
    探测器时，两边虽然都跑 ffmpeg，但**计划本身**是各算一次的；只要调用方
    一不小心给预检传了 `None`（= 没有静音信息 = 不拆长），预检的
    `n_total/n_planned/estimated_bytes` 就与真正落盘的数字不是一回事
    （实测 2 vs 3 条片段：落盘多拆出一条）。同一个实例还顺带只跑一次
    `silencedetect`（`FfmpegSilenceFinder` 缓存结果），不是两次。

    返回的 callable 带 `.error`：探测失败的原因（写进导出报告），失败时
    退化成「没有静音信息」而不是中断整趟导出。
    """
    if silence_finder is not None:
        return silence_finder
    return _SilenceRecorder(make_silence_finder(src, should_stop=should_stop))


def precheck_work(segments, labels, source, out_root, *, work_name=None,
                  sr=DEFAULT_SAMPLE_RATE, source_kind="separated_hq",
                  drop_short=False, silence_finder=None, should_stop=None,
                  free_bytes=None, safety=DISK_SAFETY,
                  reserve=DISK_RESERVE_BYTES, format="gpt_sovits") -> dict:
    """导出的**干跑**：校验 + 计划 + 磁盘预检，一个字节都不写。

    WebUI 的任务队列里先跑它，再跑 `export_work`，并且把这里算出来的
    `plan` 原样交给它（`export_work(..., plan=pre["plan"])`）—— 这样预检的数字
    与真正要写的量来自**同一次** `_plan_export`，不是两次各算一遍。
    不写盘是硬要求：预检失败必须留下**原封不动**的上一次产物。

    注意 `silence_finder=None` 时这里**不会**自己去建 ffmpeg 探测器（这一层
    不碰音源以外的东西，也不该悄悄起进程）：那样算出来的计划是「不拆长」，
    与 `export_work` 默认建探测器的结果不同。生产路径请用
    `plan_silence_finder()` 建好同一个实例再传进来。

    返回 `{ok, needed_bytes, free_bytes, estimated_bytes, planned_seconds,
    work, work_dir, list_path, report_path, n_total, n_planned, plan}`；
    `plan` 是给 `export_work(plan=...)` 用的**同一个**计划对象，本函数不改它。
    空间不够抛 `InsufficientSpace`（与 `export_work` 同一个异常类型）。
    """
    if format not in EXPORT_FORMATS:
        raise ExportError("训练格式只支持 %s，收到 %r" % (list(EXPORT_FORMATS), format))
    if int(sr) not in ALLOWED_SAMPLE_RATES:
        raise ExportError("采样率只支持 %s，收到 %r" % (list(ALLOWED_SAMPLE_RATES), sr))
    sr = int(sr)

    src = Path(source) if source else None
    if src is None or not src.is_file():
        raise ExportError("音源不存在：%s" % (source,))
    if should_stop is not None and should_stop():
        raise ExportCancelled("导出已被停止（预检之前）")

    plan = _plan_export(segments, labels, out_root, work_name=work_name,
                        sr=sr, drop_short=drop_short,
                        silence_finder=silence_finder, should_stop=should_stop,
                        export_format=format)
    free = disk_free_bytes(plan["out_root"]) if free_bytes is None else int(free_bytes)
    need = plan_export_space(plan["planned_seconds"], sr, safety=safety,
                             reserve=reserve)
    if free < need:
        raise InsufficientSpace(
            "磁盘空间不足：需要约 %.1f MB（含预留 %.0f MB），%s 只剩 %.1f MB"
            % (need / 1e6, reserve / 1e6,
               Path(plan["out_root"]).anchor or plan["out_root"], free / 1e6))
    return {"ok": True, "needed_bytes": need, "free_bytes": free,
            "estimated_bytes": plan["estimated_bytes"],
            "planned_seconds": plan["planned_seconds"],
            "work": plan["work"], "work_dir": str(plan["work_dir"]),
            "list_path": str(plan["list_path"]),
            "report_path": str(plan["report_path"]),
            "source_kind": source_kind,
            "n_total": len(plan["entries"]),
            "n_planned": plan["n_planned"],
            # 预检算出来的计划原样带走，落盘复用它 —— 见函数文档。
            "plan": plan}


def export_work(segments, labels, source, out_root, *, work_name=None,
                sr=DEFAULT_SAMPLE_RATE, source_kind="separated_hq", drop_short=False,
                silence_finder=None, should_stop=None, free_bytes=None,
                trim=True, trim_threshold_db=TRIM_THRESHOLD_DB, plan=None,
                format="gpt_sovits") -> dict:
    """把一集的切分计划落成训练集 + `.list` + 导出报告。

    segments / labels : 与 `plan_clips` 同一份（`segments.jsonl` 与 `speakers.json`）
    source            : 音源（`sep_hq` 人声轨 / 既有 16 k 轨 / 原片）
    out_root          : 导出根目录（`output\\说话人`）；作品名与 `.list`/报告都在它下面
    work_name         : 缺省取 `labels["work_name"]`
    drop_short        : True = 短句只记数不落盘（默认落 `_短句\\`）
    silence_finder    : 注入点；None = 用真实的 ffmpeg silencedetect
    should_stop       : 可中断回调；None = 不中断
    free_bytes        : 覆盖磁盘剩余空间（测试用）；None = 查真实磁盘
    trim              : 是否做首尾 trim
    trim_threshold_db : trim 的静音判据（dBFS，默认 -40）。按片可调：-40 dB 在
                        本机实测素材上只留 10.4 dB 余量（最静一句峰值 -29.6 dBFS），
                        换成低语/远处对白这种更安静的音源时，-40 会把起音甚至整条吃掉 ——
                        Task 8 需要按片校，所以这个口子开在 `export_work` 上。
    plan              : `precheck_work(...)["plan"]`。给了就**不再算第二遍计划**，
                        预检的 `n_planned` / `estimated_bytes` 与真正落盘的就是同一次
                        计算的结果（P4：预检与落盘不是同一条计划）。plan 里的
                        `entries` 会就地被写上 path/seconds/in_list，所以同一个 plan
                        只能给一次 `export_work`（预检那边不碰它，重复用是安全的，
                        但第二次落盘会复用上一轮的对象）。

    只允许用默认阈值调 `plan_clips`（接缝第 10 条）：阈值组合一变会造出
    「14.1 s 且 over_limit=False」这种无法解释的产物，导出层不给自己留这个口子。

    返回 `{ok, list, list_path, report_path, work_dir, folders, n_clips, n_total,
    n_written, n_list, n_short, n_unknown, total_mb, estimated_mb, source_kind,
    sample_rate, warning, report}`。`warning` 是音源类型的警告；一句片段都没进
    `.list`（`n_list == 0`）时还会追加「`.list` 是空文件」的警告。
    """
    if format not in EXPORT_FORMATS:
        raise ExportError("训练格式只支持 %s，收到 %r" % (list(EXPORT_FORMATS), format))
    if int(sr) not in ALLOWED_SAMPLE_RATES:
        raise ExportError("采样率只支持 %s，收到 %r" % (list(ALLOWED_SAMPLE_RATES), sr))
    sr = int(sr)

    src = Path(source) if source else None
    if src is None or not src.is_file():
        raise ExportError("音源不存在：%s" % (source,))

    recorder = None
    finder = silence_finder
    if finder is None:
        recorder = _SilenceRecorder(make_silence_finder(src, should_stop=should_stop))
        finder = recorder

    if plan is None:
        plan = _plan_export(segments, labels, out_root, work_name=work_name,
                            sr=sr, drop_short=drop_short, silence_finder=finder,
                            should_stop=should_stop, export_format=format)
    elif Path(plan.get("out_root", "")) != Path(out_root):
        # 传错计划会静默写到另一个导出根去 —— 宁可当场报错
        raise ExportError("预检计划与本次导出根不一致：%s != %s"
                          % (plan.get("out_root"), out_root))
    if plan.get("format", "gpt_sovits") != format:
        raise ExportError("预检计划的训练格式与本次导出不一致")
    work = plan["work"]
    out_root = plan["out_root"]
    work_dir = plan["work_dir"]
    list_path = plan["list_path"]
    manifest_path = out_root / (work + MANIFEST_SUFFIXES[format])
    report_path = plan["report_path"]
    segments = plan["segments"]
    labels = plan["labels"]
    language = export_language(labels.get("language"), format)
    previous = plan["previous"]
    entries = plan["entries"]
    planned_seconds = plan["planned_seconds"]
    estimated_bytes = plan["estimated_bytes"]
    # 探测器可能是调用方传进来的（生产路径 = 预检与落盘共用的那一个），
    # 它的 `.error` 就是这次探测的失败原因，报告里要如实写。
    silence_error = recorder.error if recorder is not None else (
        getattr(finder, "error", "") or "")

    free = disk_free_bytes(out_root) if free_bytes is None else int(free_bytes)
    need = plan_export_space(planned_seconds, sr)
    if free < need:
        raise InsufficientSpace(
            "磁盘空间不足：需要约 %.1f MB（含预留 %.0f MB），%s 只剩 %.1f MB"
            % (need / 1e6, DISK_RESERVE_BYTES / 1e6, Path(out_root).anchor or out_root,
               free / 1e6))

    if should_stop is not None and should_stop():
        raise ExportCancelled("导出已被停止（落盘之前）")

    # F1：音频目录马上要被整棵重建，旧 `.list` / 旧报告里指向的 wav 随即消失 ——
    # 先把它们删掉，让「音频目录被重建」与「`.list` 存在」同生共死。
    # 否则用户在循环中途点停止（或磁盘写失败抛 OSError）时，盘上会留下
    # 「已被清空的音频目录 + 上一次的旧 `.list`」：目录看起来是一次成功导出，
    # 而旧 `.list` 里有行指向已删的 wav，GPT-SoVITS 的 `1-get-text.py` 建库时会直接炸。
    # 位置很关键：必须排在**所有**预检（采样率、音源存在、作品名、磁盘空间、
    # 上面的 should_stop）之后 —— 预检失败/落盘前就中止时，上一次的产物一字不动。
    if any((out_root / (work + suffix)).exists() for kind, suffix in MANIFEST_SUFFIXES.items()
           if kind != format):
        raise ExportError("已存在其他格式的训练集，请使用新的导出目录。")
    for stale in (manifest_path, report_path):
        try:
            stale.unlink()
        except OSError:
            pass

    _rebuild_work_dir(out_root, work_dir)

    skipped = []
    not_in_list = []
    ffmpeg_errors = []
    written = []
    empty_text_skipped = 0
    short_skipped = 0
    mismatch = 0

    for entry in entries:
        if should_stop is not None and should_stop():
            raise ExportCancelled("导出已被停止（第 %d 条片段）" % (len(written) + 1,))
        # 接缝第 5 条：空文本一律不落盘（任何 bucket），只记数
        if not entry["text"]:
            empty_text_skipped += 1
            entry["skip_reason"] = "empty_text"
            skipped.append(_skip_record(entry, "empty_text"))
            continue
        if drop_short and entry["short"]:
            short_skipped += 1
            entry["skip_reason"] = "short"
            skipped.append(_skip_record(entry, "short"))
            continue
        if format == "fish_speech" and not entry["eligible"]:
            skipped.append(_skip_record(entry, "not_exportable"))
            continue

        dest = work_dir.joinpath(*entry["subdir"]).joinpath(entry["name"])
        try:
            cut_segment(str(src), entry["start"], entry["dur"], dest, sr=sr,
                        trim=trim, trim_threshold_db=trim_threshold_db,
                        should_stop=should_stop)
        except ExportCancelled:
            raise
        except Exception as exc:                     # noqa: BLE001 - 一条失败不中断整批
            detail = "%s: %s" % (type(exc).__name__, exc)
            ffmpeg_errors.append({"index": entry["index"], "start": entry["start"],
                                  "error": detail})
            entry["skip_reason"] = "ffmpeg_error"
            skipped.append(_skip_record(entry, "ffmpeg_error"))
            continue

        seconds = wav_seconds(dest)
        if not seconds:
            # trim 把整条都当静音吃掉了：留一个 0 秒的空壳没有意义
            try:
                dest.unlink()
            except OSError:
                pass
            entry["skip_reason"] = "empty_after_trim"
            skipped.append(_skip_record(entry, "empty_after_trim"))
            continue

        entry["path"] = dest
        entry["rel_path"] = dest.relative_to(work_dir).as_posix()
        entry["seconds"] = seconds
        written.append(entry)
        if entry["eligible"] != (entry["role"] is not None and not entry["short"]):
            # 落盘位置与 `is_exportable()` 不一致 = segments.jsonl 与计划不同源
            # （被人手改过 / index 对不上）。这**不是**第二道门，只如实记进报告。
            mismatch += 1
        # 接缝第 8 条：进不进 `.list` **只用** `is_exportable()` 一道门，不再各写一套判断
        entry["in_list"] = bool(entry["eligible"])
        if not entry["in_list"]:
            not_in_list.append({
                "index": entry["index"], "start": round(entry["start"], 3),
                "dir": entry["dir"], "path": entry["rel_path"],
                # 未命名优先：没有角色名是更根本的事实，哪怕它同时也短。
                # 一条 1.5 s 的未命名片段要是写 "short"，会让人以为它有角色名。
                "reason": "unnamed" if entry["role"] is None else (
                    "short" if entry["short"] else "not_exportable"),
            })

    rows = [e for e in written if e["in_list"]]
    if not rows:
        # 第二道门（`_plan_export` 已经在计划阶段拒过一次）。防的是「调用方传进来一份
        # 手搓的 plan」这类绕过：**任何路径**都不许产出 0 字节的 `.list`。
        raise ExportError(_nothing_listable_message(entries))
    if should_stop is not None and should_stop():
        raise ExportCancelled("导出已被停止（发布清单之前）")
    n_list = len(rows)
    _write_manifest(manifest_path, rows, language, format, should_stop)

    dirs = {}
    for entry in written:
        dirs[entry["dir"]] = dirs.get(entry["dir"], 0) + 1
    total_seconds = round(sum(e["seconds"] for e in written), 3)
    list_seconds = round(sum(e["seconds"] for e in written if e.get("in_list")), 3)
    actual_bytes = sum(e["path"].stat().st_size for e in written)

    report = {
        "version": 1,
        "work": work,
        "work_dir": str(work_dir),
        "format": format,
        "language": language,
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "source": str(src),
        "source_kind": source_kind,
        "sample_rate": sr,
        "channels": 1,
        "sample_width_bytes": 2,
        "trim": bool(trim),
        "trim_threshold_db": float(trim_threshold_db),
        "peak_norm_dbfs": PEAK_DBFS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_clips": len(entries),
        "written_clips": len(written),
        "list_clips": n_list,
        "named_clips": sum(1 for e in entries if e["bucket"] == "named"),
        # 按**计划路由桶**统计（`_短句\`），不是数磁盘上的文件：
        # `drop_short=True` / 空文本 / ffmpeg 失败都会让它与落盘数不一致。
        # 未命名的短片段路由到 `_未命名\`，不算在这里 —— 想数「被丢弃的短句」
        # 看 `short_skipped`。
        "short_clips": sum(1 for e in entries if e["bucket"] == "short"),
        "unnamed_clips": sum(1 for e in entries if e["bucket"] == "unnamed"),
        "short_skipped": short_skipped,
        "over_limit_clips": sum(1 for e in entries if e["over_limit"]),
        "empty_text_skipped": empty_text_skipped,
        "skipped_clips": len(skipped),
        "skipped": sorted(skipped, key=lambda r: (r["index"], r["start"])),
        "not_in_list": not_in_list,
        "ffmpeg_errors": ffmpeg_errors,
        "silence_error": silence_error,
        "routing_mismatch": mismatch,
        "planned_seconds": round(planned_seconds, 3),
        "total_seconds": total_seconds,
        "list_seconds": list_seconds,
        "estimated_bytes": estimated_bytes,
        "estimated_mb": round(estimated_bytes / 1e6, 2),
        "actual_bytes": actual_bytes,
        "actual_mb": round(actual_bytes / 1e6, 2),
        "disk_free_bytes": free,
        "dirs": dirs,
        "previous": previous,
        "previous_clip_count": previous["clip_count"],
        "written": [{"index": e["index"], "start": round(e["start"], 3),
                     "end": round(e["end"], 3), "dir": e["dir"],
                     "path": e["rel_path"], "seconds": round(e["seconds"], 3),
                     "in_list": bool(e.get("in_list")),
                     "short": e["short"], "over_limit": e["over_limit"]}
                    for e in written],
    }
    if format == "gpt_sovits":
        report["list_path"] = str(list_path)
    _atomic_write_lf(report_path,
                     json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # 这里**不再**有 `n_list == 0` 的警告分支：那种情况在计划阶段就被
    # `_plan_export` 拒绝（`ExportError`），0 字节的 `.list` 从任何路径都产不出来。
    # 以前是「照写 + 警告」，但 0 字节 `.list` 要到 GPT-SoVITS 解包时才炸
    # （`f.read().strip("\n").split("\n")` 得到 `['']`），那时已经没有线索了。
    warning = source_warning(source_kind)

    return {
        "ok": True,
        "format": format,
        "manifest_path": str(manifest_path),
        **({"list": str(list_path), "list_path": str(list_path)} if format == "gpt_sovits" else {}),
        "report_path": str(report_path),
        "work_dir": str(work_dir),
        "folders": dirs,
        "n_clips": len(written),
        "n_written": len(written),
        "n_total": len(entries),
        "n_list": n_list,
        "n_short": report["short_clips"],
        "n_unknown": report["unnamed_clips"],
        "n_skipped": len(skipped),
        "total_mb": report["actual_mb"],
        "estimated_mb": report["estimated_mb"],
        "total_seconds": total_seconds,
        "source_kind": source_kind,
        "sample_rate": sr,
        "warning": warning,
        "report": report,
    }


def export_language(language, export_format):
    language = str(language or LIST_LANG).strip().lower()
    if language == "auto":
        language = "und"
    if export_format == "gpt_sovits" and language not in GPT_SOVITS_LANGUAGES:
        raise ExportError("GPT-SoVITS 请使用明确的日语、中文、英语、韩语或粤语识别结果，也可选择其他标注格式。")
    if language != "und" and language not in LANGUAGES:
        raise ExportError("未知的识别语言：%s" % language)
    return language


def _write_manifest(path, rows, language, export_format, should_stop):
    if export_format == "gpt_sovits":
        write_list(path, [(str(e["path"]), e["list_speaker"], e["text"]) for e in rows], language)
        return

    is_fish = export_format == "fish_speech"
    fields = ["audio", "speaker", "language", "text"]
    if is_fish:
        fields.append("label")
    lines = [] if export_format == "indextts" else ["\t".join(fields)]
    for entry in rows:
        if should_stop is not None and should_stop():
            raise ExportCancelled("导出已被停止（写入标注时）")
        audio = entry["path"]
        relative = audio.relative_to(path.parent).as_posix()
        text = " ".join(entry["text"].split())
        if export_format == "indextts":
            sample_id = hashlib.sha256(str(audio.resolve()).encode("utf-8")).hexdigest()[:24]
            lines.append(json.dumps({"id": sample_id, "text": text, "audio": relative,
                                     "speaker": entry["role"], "language": language,
                                     "duration": round(entry["seconds"], 3)}, ensure_ascii=False))
        else:
            row = [relative, " ".join(entry["role"].split()), language, text]
            if is_fish:
                label = audio.with_suffix(".lab")
                _atomic_write_lf(label, text + "\n")
                row.append(label.relative_to(path.parent).as_posix())
            lines.append("\t".join(row))
    if should_stop is not None and should_stop():
        raise ExportCancelled("导出已被停止（发布清单之前）")
    _atomic_write_lf(path, "\n".join(lines) + "\n")


def _skip_record(entry, reason) -> dict:
    return {"index": entry["index"], "start": round(entry["start"], 3),
            "end": round(entry["end"], 3), "dir": entry["dir"], "reason": reason}
