#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Task 3：训练集专用的**高质量分离**（从原片全采样率重新分离）。

## 为什么需要它

现有 `sep\` 人声轨是**从 16 kHz 单声道**输入分离出来的（`webui.py` 把
`raw16k.wav` 喂给人声分离），所以那条「44.1 kHz 立体声」人声轨的**内容带宽只有
8 kHz** —— 44.1 kHz 只是上采样出来的壳。实测 10~16 kHz 能量占比：

| 音频 | 10~16 kHz 能量占比 |
|---|---|
| 原片抽出的 raw44k | 5.748e-03 |
| 现有 `sep\*_(Vocals)*.wav`（16 k 输入分离） | 2.492e-07 |
| 从这里重新分离的 HQ 人声 | 6.101e-03 |

拿现有轨切训练集会得到**发闷**的数据。本模块从**原片全采样率**（mono 44.1 kHz）
重新走同一套 BS-Roformer 权重，把高频完整保住。

## 分层

* `extract_full_rate` / `ffmpeg_convert`：ffmpeg 边界（`Popen` + `should_stop`）。
* `separate_hq`：`venv-sep\Scripts\python.exe sep_worker.py <in> <outdir> <model>`，
  成功时子进程 stdout 打印 `RESULT=<人声轨路径>`；stderr 的 tqdm 噪音（含 `%|`）要过滤。
  产物路径用 **glob `(Vocals)`** 定位，**不拼死文件名** —— audio-separator 会把模型名
  截断（`model_bs_roformer_ep_317_sdr_12.9755.ckpt` -> `..._sdr_12`）。
* `pick_source`：选音源，返回 `(路径, source_kind)`，`source_kind` 的 4 个取值与
  `train_export.SOURCE_KINDS` 一致。
* `cleanup_intermediates`：导出成功后删 `raw44k.wav` 与 `sep_hq\` 的**伴奏轨**，
  **保留 HQ 人声轨**（重导可省一次 3 分钟分离）。

## 磁盘（实测，比计划里估的小）

| 产物 | 实测 |
|---|---|
| `raw44k.wav`（mono 44.1 k / 180 s） | 15.9 MB |
| `sep_hq\*_(Vocals)*.wav`（**stereo** 44.1 k / 180 s） | 31.8 MB |
| `sep_hq\*_(Instrumental)*.wav` | 31.8 MB |

峰值约 **79 MB/集**（计划里按 380 MB 估的，那是按 24 分钟集长算的）。

## 复用判断

`sep_hq\.state.json` 以「原片 size + mtime + 模型名」为键；命中且 HQ 人声轨还在
就不重抽、不重分。状态文件在**分离成功之后**才原子写入，所以中断/失败不会留下
「看起来可复用」的状态；读坏了（人手改、写了一半）一律当作未命中，绝不抛。

## 中断

长耗时步骤一律 `Popen` + 轮询 + `should_stop`，禁止不可中断的 `subprocess.run`。
被停掉时抛 `SourceCancelled`，并**清掉这次分离写出来的半成品**（伴奏轨等），
避免下次误用半个产物。

临时文件只写调用方给的 `work`（工作台使用项目的 `work/` 目录）。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

import jisheng as A
import train_export as te

HERE = Path(__file__).resolve().parent

#: 分离跑在独立环境（运行独立安装的 PyTorch），与 `webui.py` 同一套机制
SEP_PYTHON = HERE / "venv-sep" / "Scripts" / "python.exe"
SEP_WORKER = HERE / "sep_worker.py"

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: 默认分离模型（与 `webui.SEP_MODEL_CHOICES` 的推荐项一致）
DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

#: 训练用音源的默认采样率（与 `train_export.DEFAULT_SAMPLE_RATE` 一致）
DEFAULT_SR = te.DEFAULT_SAMPLE_RATE

#: 高质量中间产物（都在 work 下）
RAW_HQ_NAME = "raw44k.wav"
RAW_FALLBACK_NAME = "raw32k.wav"
SEP_HQ_DIRNAME = "sep_hq"
STATE_NAME = ".state.json"

_POLL_S = 0.5
_TERM_GRACE_S = 10.0
_JOIN_S = 5.0

#: stderr 里这些行是环境噪音，不该被当成失败原因贴进异常
_NOISE_MARKERS = (
    "WARNING: The installed PyTorch",
    "Failed to load cublas",
    "Failed to load cufft",
    "Failed to load cudart",
    "Please follow https://onnxruntime.ai",
)


class SourceError(RuntimeError):
    """音源准备失败（原片缺失、ffmpeg 报错、分离失败、环境缺失……）。"""


class SourceCancelled(SourceError):
    """用户中止。已经写出来的中间产物不完整，调用方应把它们当作无效。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _progress(cb, message):
    """进度回调刻意做得最松：WebUI 传 `log`，测试传 `list.append`，也可以传 None。"""
    if cb is None:
        return
    try:
        cb(message)
    except Exception:                             # noqa: BLE001 - 进度不该影响主流程
        pass


def _remove_quietly(path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


def work_audio(work, name) -> Path:
    """`work` 下的中间音频路径。

    `Path(work) / name` 是必须的：`work` 传 str 时 `str / name` 会 `TypeError`，
    传 `Path` 时才是拼接 —— 实测一个 str 折进 `Path()` 会让整条路径变成
    **相对当前工作目录**（当时是 C: 盘），于是「文件明明在 work 下，
    `raw.exists()` 却是 False」，静默走错分支。参数一律先过 `Path()`。
    """
    return Path(work) / name


def sep_hq_dir(work) -> Path:
    return Path(work) / SEP_HQ_DIRNAME


def state_path(work) -> Path:
    return sep_hq_dir(work) / STATE_NAME


def _atomic_write_text(path, text) -> None:
    """临时文件 + `os.replace`，避免状态文件写一半被读到。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def state_key(media, model=DEFAULT_MODEL) -> list:
    """复用键：原片 size + mtime + 模型名。

    用 `st_size` / `st_mtime_ns` 而不是哈希 —— 实拍 3 分钟样片要读 20 MB，
    对每集几百 MB 的原片做全量哈希不值得，而「同 size 同 mtime 却换了内容」
    在这里不是要防的场景（真换了内容 Windows 会改 mtime）。
    """
    media = Path(media)
    st = media.stat()
    return [str(media.resolve()), str(st.st_size), str(st.st_mtime_ns),
            str(model)]


def read_state(work):
    """读状态文件；不存在 / 读坏 / 类型不对一律返回 `{}`（沿用 `load_labels` 的容错风格）。"""
    path = state_path(work)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(work, key, *, vocals=None, model=DEFAULT_MODEL) -> Path:
    """原子写状态文件（分离**成功之后**才调）。父目录不存在会顺带建出来。"""
    path = state_path(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "key": list(key),
        "model": str(model),
        "vocals": str(vocals) if vocals else "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return path


def _within(path, root) -> bool:
    """`path` 是否**严格落在** `root` 里面（root 自己、`..`、盘符根都不算）。

    与 `train_export._rebuild_work_dir` 同一个判据（「目标必须在根的里面」），
    只是这里先 `realpath` 把 `..` 与符号链接折平，再按「根 + 分隔符」做前缀比较；
    并用 `normcase` 归一大小写：Windows 上 `D:\\WORK\\EP01` 与 `d:\\work\\ep01`
    是同一个目录，只比字符串会把合法路径误判成越界（那会让复用静默失效）。
    判不出来（路径非法）时返回 False —— 宁可不复用，也不越界读音频。
    """
    try:
        target = os.path.normcase(os.path.realpath(str(path)))
        base = os.path.normcase(os.path.realpath(str(root)))
    except OSError:
        return False
    if not base or target == base:
        return False
    return target.startswith(base.rstrip("\\/") + os.sep)


def reuse_hq(work, key):
    """命中复用则返回 HQ 人声轨路径，否则 None。

    三个条件都要满足：状态文件的键**完全相同**、键里记的人声轨**还在盘上**
    （`cleanup_intermediates` 会删 `raw44k.wav`，但绝不删人声轨）、
    且它**落在 `work` 之内**。

    最后一条不能省：`.state.json` 是给人看也可以手改的 JSON，里面的绝对路径
    完全可以指到**别的作品目录**（审查者用合成数据复现过：导出会从另一个 work
    读音频，训练集里混进别的集的声音）。越界一律当未命中，退回 glob ——
    glob 是在 `work/sep_hq` 里找的，结果必然在 `work` 之内。
    状态文件里记的路径丢了也退回 glob：audio-separator 的命名会随版本变。
    """
    state = read_state(work)
    if not state or list(state.get("key") or []) != list(key or []):
        return None
    recorded = state.get("vocals") or ""
    if recorded:
        p = Path(recorded)
        if p.exists() and _within(p, work):
            return p
    return find_vocals(sep_hq_dir(work))


def _decoded(stream) -> str:
    """把子进程的字节流解成文本（编码容错，Windows 上不能假设 UTF-8）。"""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _clean_noise(text) -> list:
    """去掉 tqdm 进度条（`%|`）与环境噪音行，只留真正像错误的信息。"""
    keep = []
    for line in _decoded(text or "").replace("\r", "\n").splitlines():
        line = line.strip()
        if not line or "%|" in line or "it/s]" in line or "s/it]" in line:
            continue
        if line.startswith(_NOISE_MARKERS):
            continue
        keep.append(line)
    return keep


class _ProcRunner:
    """`Popen` + 两条读线程 + `should_stop` 轮询。

    为什么不用 `subprocess.run`：抽取与分离都是分钟级动作，必须可中断
    （计划全局约束 3）。为什么不用 `communicate()` 一把梭：那样在子进程结束前
    拿不到 `should_stop` 的机会，且 stderr 写满管道会死锁。
    """

    def __init__(self, cmd, should_stop=None, progress=None, note=""):
        self.cmd = [str(c) for c in cmd]
        self.should_stop = should_stop
        self.progress = progress
        self.note = note
        self.out_lines = []
        self.err_lines = []
        self.proc = None

    def run(self):
        """返回 `(returncode, stdout_text, stderr_text)`；被停掉时抛 `SourceCancelled`。"""
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise SourceError("无法启动子进程 %s：%s" % (self.cmd[0], exc))

        try:
            return self._pump()
        finally:
            # 管道不关会漏句柄（实测每次调用两个 ResourceWarning），
            # 长流程里反复抽取/分离会一直攒着。
            for stream in (self.proc.stdout, self.proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def _pump(self):
        threads = []
        for pipe, sink in ((self.proc.stdout, self.out_lines),
                           (self.proc.stderr, self.err_lines)):
            t = threading.Thread(target=self._reader, args=(pipe, sink), daemon=True)
            t.start()
            threads.append(t)

        started = time.time()
        while self.proc.poll() is None:
            if self.should_stop is not None and self.should_stop():
                self._terminate()
                raise SourceCancelled("已被停止（%s）" % (self.note or self.cmd[0]))
            time.sleep(_POLL_S)
            if self.progress is not None:
                _progress(self.progress,
                          "%s已用时 %s" % (self.note, A.fmt_hms(time.time() - started)))

        for t in threads:
            t.join(timeout=_JOIN_S)
        return (self.proc.returncode,
                "".join(self.out_lines), "".join(self.err_lines))

    @staticmethod
    def _reader(pipe, sink):
        try:
            for line in pipe:
                sink.append(_decoded(line))
        except Exception:                         # noqa: BLE001 - 管道被关掉是正常的
            pass

    def _terminate(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=_TERM_GRACE_S)
        except Exception:                         # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:                     # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# ffmpeg 边界
# ---------------------------------------------------------------------------

def ffmpeg_convert(src, dest, *, sr=DEFAULT_SR, ac=1, should_stop=None,
                   progress=None) -> Path:
    """把任意音视频转成 `pcm_s16le` WAV。

    `-nostdin` 必须带：后台跑的时候 ffmpeg 会去读 stdin 然后卡死
    （`train_export._run_ffmpeg` 里踩过同一个坑）。
    """
    src, dest = Path(src), Path(dest)
    if not src.exists():
        raise SourceError("音频源不存在：%s" % src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", str(src), "-vn", "-ac", str(int(ac)), "-ar", str(int(sr)),
           "-c:a", "pcm_s16le", str(dest)]
    runner = _ProcRunner(cmd, should_stop=should_stop, progress=progress,
                         note="抽取音频中，")
    try:
        code, _out, err = runner.run()
    except SourceCancelled:
        # 半条 raw44k.wav 留下会被下次误当成品（体积/时长都不对），当场清掉
        _remove_quietly(dest)
        raise
    if code != 0:
        _remove_quietly(dest)
        detail = " | ".join(_clean_noise(err)[-3:]) or "退出码 %s" % code
        raise SourceError("ffmpeg 抽取失败：%s" % detail)
    if not dest.exists() or dest.stat().st_size == 0:
        # 磁盘空间不足可能留下零字节产物，需要显式检查
        raise SourceError("ffmpeg 未产出有效文件：%s" % dest)
    return dest


def extract_full_rate(media, work, *, sr=44100, should_stop=None,
                      progress=None, out_name=RAW_HQ_NAME) -> Path:
    """从**原片**抽全采样率单声道音频（默认 44.1 kHz / mono / pcm_s16le）。

    单声道是有意的：实测 audio-separator 接受单声道输入 —— 现有 16 k 流程喂进去的
    就是单声道，这里只是把采样率提到全带宽。
    """
    return ffmpeg_convert(media, work_audio(work, out_name), sr=sr, ac=1,
                          should_stop=should_stop, progress=progress)


def probe_sample_rate(path):
    """`ffprobe` 读采样率；读不到返回 None（调用方自己决定要不要当错误）。"""
    try:
        proc = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of",
             "default=noprint_wrappers=1:nokey=1", str(path)],
            stdin=subprocess.DEVNULL, capture_output=True,
        )
    except OSError:
        return None
    text = _decoded(proc.stdout).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 分离
# ---------------------------------------------------------------------------

def find_vocals(sep_dir):
    """在分离输出目录里找人声轨（glob `(Vocals)`，**不拼死文件名**）。

    audio-separator 会把模型名截断，所以 `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
    产出的文件名只有 `..._sdr_12`。候选里优先带 `(Vocals)` 的，其次任何含
    `vocal` 的；排序取第一条，保证同一目录多次调用结果稳定。
    """
    sep_dir = Path(sep_dir)
    if not sep_dir.is_dir():
        return None
    ranked = []
    for f in sorted(sep_dir.glob("*.wav")):
        low = f.name.lower()
        if "(vocals)" in low:
            ranked.append((0, f))
        elif "vocal" in low:
            ranked.append((1, f))
    return ranked[0][1] if ranked else None


def find_instrumental(sep_dir):
    """伴奏轨（清理用）。同样不拼死文件名。"""
    sep_dir = Path(sep_dir)
    if not sep_dir.is_dir():
        return []
    out = []
    for f in sorted(sep_dir.glob("*.wav")):
        low = f.name.lower()
        if "(instrumental)" in low or "instrumental" in low:
            out.append(f)
    return out


def _snapshot(sep_dir) -> set:
    """分离前先记下目录里已有哪些 wav，被停掉时只删**这次新增**的。"""
    sep_dir = Path(sep_dir)
    if not sep_dir.is_dir():
        return set()
    return {f.name for f in sep_dir.glob("*.wav")}


def _resolve_result(stdout, sep_dir):
    """先认 `RESULT=`，再 glob。返回 Path 或 None。"""
    for line in _decoded(stdout).splitlines():
        line = line.strip()
        if not line.startswith("RESULT="):
            continue
        raw = line[len("RESULT="):].strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = Path(sep_dir) / p.name
        if p.exists():
            return p
    return find_vocals(sep_dir)


def separate_hq(work, *, model=DEFAULT_MODEL, should_stop=None, progress=None,
                media=None, key=None) -> Path:
    """用 `venv-sep` + `sep_worker.py` 从 `work\\raw44k.wav` 分出人声（44.1 kHz）。

    成功时写 `sep_hq\\.state.json`（键 = 原片 size+mtime+模型），供下次复用。
    """
    work = Path(work)
    raw = work_audio(work, RAW_HQ_NAME)
    if not raw.exists():
        raise SourceError("缺少分离输入：%s（先调 extract_full_rate）" % raw)
    if not Path(SEP_PYTHON).exists():
        raise SourceError("分离环境不存在：%s" % SEP_PYTHON)
    if not Path(SEP_WORKER).exists():
        raise SourceError("分离脚本不存在：%s" % SEP_WORKER)

    out_dir = sep_hq_dir(work)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _snapshot(out_dir)

    _progress(progress, "分离模型：%s（独立进程，全采样率输入）" % model)
    cmd = [SEP_PYTHON, SEP_WORKER, raw, out_dir, model]
    runner = _ProcRunner(cmd, should_stop=should_stop, progress=progress,
                         note="分离中，")
    if key is None and media is not None:
        key = state_key(media, model)
    try:
        code, out_text, err_text = runner.run()
    except SourceCancelled:
        # 被停掉：清掉这次新写的半成品 + 作废「不是这一次」的状态
        _discard_partials(work, before, key)
        raise

    if code != 0:
        _discard_partials(work, before, key)
        detail = " | ".join(_clean_noise(err_text)[-3:]) or "退出码 %s" % code
        raise SourceError("分离失败：%s" % detail)

    vocal = _resolve_result(out_text, out_dir)
    if vocal is None or not vocal.exists():
        _discard_partials(work, before, key)
        tail = " | ".join(_clean_noise(err_text)[-3:])
        raise SourceError("分离未产出人声轨（目录 %s）%s"
                          % (out_dir, ("：" + tail) if tail else ""))

    if key is None:
        # 没给原片也没给键：至少按模型名留个状态，让「带 media 的调用」能超集化——
        # 两条键不相等就不会误命中，只是从裸调用复用不到而已。
        key = [str(model)]
    write_state(work, key, vocals=vocal, model=model)
    return vocal


def _discard_partials(work, before, key) -> None:
    """分离没成功时的收尾：删这次新写的 wav；状态文件**只在键对不上时**才删。

    为什么不是无脑删状态：状态文件可能正记着上一次**成功**分离产出的、现在还
    完好躺在盘上的 HQ 人声轨。这次重分离失败（或用户中途停掉）并不代表那条旧轨
    失效了 —— 无脑删会把「重导可省一次 3 分钟分离」这个卖点白扔掉，
    下次还得白跑一遍（实测过这条路径：重分离失败 -> 下次重跑一次分离）。
    真正失效的证据是「键变了」或「键里记的人声轨不在了」，那两条由 `reuse_hq`
    自己验，所以这里只清掉键对不上的那条。
    """
    _remove_new_wavs(sep_hq_dir(work), before)
    state = read_state(work)
    if not state:
        return
    if key is not None and list(state.get("key") or []) == list(key):
        return                                    # 这条状态描述的就是本次要的轨，留着
    _remove_quietly(state_path(work))


def _remove_new_wavs(sep_dir, before) -> int:
    """删掉分离目录里**这次新增**的 wav（不动之前留下的成品）。返回删掉的条数。"""
    sep_dir = Path(sep_dir)
    n = 0
    for f in sorted(sep_dir.glob("*.wav")):
        if f.name in before:
            continue
        _remove_quietly(f)
        n += 1
    return n


# ---------------------------------------------------------------------------
# 选音源
# ---------------------------------------------------------------------------

def pick_source(work, media, mode="hq", *, model=DEFAULT_MODEL,
                should_stop=None, progress=None):
    """按 `mode` 挑一条训练集音源，返回 `(Path, source_kind)`。

    * `mode="hq"`   -> 跑/复用 `sep_hq`，`source_kind="separated_hq"`（无警告）
    * `mode="fast"` -> 既有 `sep\\*Vocals*`，`"separated_16k"`（**警告：高频已丢失**）
    * 没 `sep\\`     -> `raw16k.wav`，`"raw16k"`
    * 连 16 k 都没有 -> 直接从原片抽 32 k，`"media"`

    `source_kind` 的取值与 `train_export.SOURCE_KINDS` 一一对应，
    面板警告直接用 `train_export.source_warning(kind)`。
    """
    work, media = Path(work), Path(media)
    if not media.exists():
        raise SourceError("原片不存在：%s" % media)
    if mode == "hq":
        key = state_key(media, model)
        hit = reuse_hq(work, key)
        if hit is not None:
            _progress(progress, "复用已有的高质量人声轨：%s" % hit.name)
            return hit, "separated_hq"
        raw = work_audio(work, RAW_HQ_NAME)
        if not raw.exists():
            extract_full_rate(media, work, sr=44100, should_stop=should_stop,
                              progress=progress)
        vocal = separate_hq(work, model=model, should_stop=should_stop,
                            progress=progress, key=key)
        return vocal, "separated_hq"

    # 快速/回退：只认 sep\ 里既有的人声轨（16 kHz 输入分离出来的那条）
    legacy = find_vocals(work / "sep")
    if legacy is not None:
        return legacy, "separated_16k"
    raw16 = work_audio(work, "raw16k.wav")
    if raw16.exists():
        return raw16, "raw16k"
    out = extract_full_rate(media, work, sr=DEFAULT_SR, should_stop=should_stop,
                            progress=progress, out_name=RAW_FALLBACK_NAME)
    return out, "media"


# ---------------------------------------------------------------------------
# 清理与自检
# ---------------------------------------------------------------------------

def cleanup_intermediates(work) -> float:
    """导出成功后清理中间产物，返回释放的 MB。

    删：`raw44k.wav`（可再抽，代价几十秒）、`sep_hq\\` 里的**伴奏轨**（训练集用不到）。
    留：`sep_hq\\` 的人声轨（重导可省一次约 3 分钟分离）、`.state.json`。

    **不变量：只删 `work` 之内的文件。** 这个判据刻意用「测试钉住」而不是在函数里
    再加一道静默过滤：`targets` 里混进越界路径时，静默跳过会让「越界」变成不可
    观察的等价改动（变异实验里改坏这一处、测试全绿）。钉住它的用例是
    `tests/test_train_source.py::TestStateAndCleanup::
    test_cleanup_never_touches_files_outside_work` ——
    往 work 外面放一个同名文件，它必须还在。
    """
    work = Path(work)
    targets = [work_audio(work, RAW_HQ_NAME)] + find_instrumental(sep_hq_dir(work))
    freed = 0
    for path in targets:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        freed += size
    return freed / (1024.0 * 1024.0)


def high_band_ratio(path, lo=10000.0, hi=16000.0, *, start=0.0, seconds=1.0):
    """`lo~hi` Hz 频段能量占全带能量的比例（自检 / 诊断用）。

    取 `start` 起的 `seconds` 秒、加汉宁窗做 FFT、按功率谱求占比。
    注意**必须先把被比较的音轨都弄到 44.1 kHz 再算**：16 kHz 内容的 Nyquist
    就是 8 kHz，不上采样的话它根本没有 10~16 kHz 这个频段，比较没有意义。

    读不出音频时返回 0.0（这是诊断工具，不该把调用方炸掉）。
    """
    lo, hi = float(lo), float(hi)
    try:
        with wave.open(str(path), "rb") as fh:
            sr = fh.getframerate()
            channels = fh.getnchannels()
            width = fh.getsampwidth()
            if width != 2:
                return 0.0
            offset = max(0, int(float(start) * sr))
            fh.setpos(min(offset, max(0, fh.getnframes() - 1)))
            n = int(float(seconds) * sr)
            frames = fh.readframes(max(1, n))
    except (OSError, wave.Error, EOFError):
        return 0.0
    if not frames:
        return 0.0

    data = np.frombuffer(frames, dtype="<i2")
    if data.size == 0:
        return 0.0
    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    sig = data.astype(np.float64)
    sig = sig - sig.mean()
    if float(np.max(np.abs(sig))) <= 0.0:
        return 0.0

    sig = sig * np.hanning(sig.size)
    spec = np.fft.rfft(sig)
    power = (spec.real ** 2 + spec.imag ** 2)
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / float(sr))
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    band = float(power[(freqs >= lo) & (freqs <= hi)].sum())
    return band / total


def high_band_ratio_median(path, lo=10000.0, hi=16000.0, *, spots=None,
                           seconds=1.0):
    """在多个位置各算一次 `high_band_ratio`，返回**中位数**。

    为什么要多点：10~16 kHz 的能量高度依赖那 1 秒里有没有齿音/气声。
    实测同一台机器同一条 HQ 轨（lead 样片）：

    | 位置 | 10~16 kHz 占比 |
    |---|---|
    | 0 s | 2.889e-04 |
    | 20 s | 1.125e-03 |
    | 40 s | 1.433e-02 |
    | 100 s | 6.390e-02 |
    | 中位数 | **8.065e-03** |

    单点相差两个数量级，**只有中位数能跟 lead 给的基准值（5.7e-03 / 6.1e-03）对上**：
    lead 就是在这条轨上量到的，本函数在 raw44k 上给出中位 **5.692e-03**（lead 5.748e-03）。
    采样点刻意避开最后一秒（`spots` 不传时按 20 s 间隔取到 `时长-2 s`）。
    """
    if spots is None:
        total = _wav_seconds(path)
        if total is None:
            return high_band_ratio(path, lo, hi, seconds=seconds)
        spots = list(range(0, max(1, int(total) - 2), 20))
    values = []
    for s in spots:
        values.append(high_band_ratio(path, lo, hi, start=s, seconds=seconds))
    if not values:
        return 0.0
    values.sort()
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _wav_seconds(path):
    try:
        with wave.open(str(path), "rb") as fh:
            return fh.getnframes() / float(fh.getframerate())
    except (OSError, wave.Error, EOFError, ZeroDivisionError):
        return None
