#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
webui.py —— 动漫 ASR 本地可视化界面（零额外依赖，只用标准库起服务）

启动：
    在项目目录双击 jisheng-webui.cmd
    或运行 .\venv\Scripts\python.exe .\webui.py

浏览器打开 http://127.0.0.1:8760

特性
----
* 拖拽或填路径输入；可自定义输出目录
* 流水线：抽取音频 -> 可选人声分离 -> 识别 -> 预览/下载字幕
* 分块检查点：每识别完一段就落盘，随时可停、可续
* 手动停止 / 意外关机 / 断电后，重启界面即可「继续上次进度」
* 进度条带百分比与剩余时间估算（ETA）
"""

import hashlib
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jisheng as A  # noqa: E402
import speaker_clips  # noqa: E402
import speaker_labels  # noqa: E402
import speaker_registry  # noqa: E402
import train_export as te  # noqa: E402
import train_source as ts  # noqa: E402

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------

UI_FILE = HERE / "ui.html"
BACKGROUND_DIR = HERE / "assets" / "backgrounds"
BACKGROUND_FILES = {"megumin.webp", "eris.webp", "klee.webp"}
UPLOAD_DIR = HERE / "uploads"
WORK_DIR = HERE / "work"                      # 检查点与中间产物
SEP_MODEL_DIR = HERE / "models" / "separators"
OUTPUT_ROOT = A.DEFAULT_OUTPUT_ROOT
SEP_OUT_ROOT = OUTPUT_ROOT / "分离"            # 分离结果（供试听）
# 训练集的**导出根**：`output\说话人\<作品名>\<角色名>\`（设计 §6.3）。
# 注意 `speaker_registry` 的库**故意不放在**这个目录里（见 speaker_registry.py 的
# `_STORAGE_DIRNAME`）：导出时会按作品名整棵重建目录，库落在里面会被一起删掉。
TRAIN_EXPORT_DIRNAME = "说话人"
# 音源档位：`hq` = 从原片全采样率重新分离（贵，但高频完整）；
# `fast` = 复用既有 16 kHz 产物（快，但会有 train_export 的降级警告）。
EXPORT_SOURCE_MODES = ("hq", "fast")
HOST, PORT = "127.0.0.1", 8760

# 分离跑在独立环境：运行独立安装的 PyTorch。
# 用子进程调用可让 torch 显存随进程结束释放，不跟 ASR 抢。
SEP_PYTHON = HERE / "venv-sep" / "Scripts" / "python.exe"
SEP_WORKER = HERE / "sep_worker.py"

SEP_MODEL_CHOICES = [
    {"value": "", "label": "（请在上方勾选「先做人声分离」）"},
    {"value": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
     "label": "BS-Roformer Viperx 1297 — 已就绪，人声 SDR 11.77 ★推荐"},
    {"value": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
     "label": "MDX23C InstVoc HQ — 轻量快速，人声 SDR 10.56"},
    {"value": "vocals_mel_band_roformer.ckpt",
     "label": "MelBand Roformer (KimberleyJensen) — SDR 12.60 最高（需下载权重）"},
    {"value": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
     "label": "Mel-Roformer Denoise (aufr33) — 只去噪不分离，副作用最小"},
]

DIARIZE_MODEL_CHOICES = [
    {"value": "pyannote/speaker-diarization-3.1",
     "label": "自动：优先本地 models\\pyannote，没有则走在线（推荐）"},
    {"value": "pyannote/speaker-diarization",
     "label": "pyannote/speaker-diarization（旧版，强制在线下载）"},
]

# 声纹库里的「用了哪个嵌入模型」标记。库靠它决定旧向量还参不参与匹配
# （换模型后向量空间变了，混着比会系统性认错人）。
#
# 这是**回退值**，不是实际写进库的值：实际身份由 `embedding_model_tag()`
# 从这次真正跑过的分割模型推导（Task 5 审查 D7：写死一个常量时，「换模型后
# 旧向量失效」这条**永远触发不了**；而 `DIARIZE_MODEL_CHOICES` 里有旧版 2.x
# 流水线、`--model` 还接受本地目录，同维不同模型光靠维度检查拦不住）。
# 回退值描述的是本项目自带的那条流水线：3.1 + wespeaker-voxceleb-resnet34-LM。
EMBEDDING_MODEL_TAG = "pyannote/speaker-diarization-3.1/wespeaker-voxceleb-resnet34-LM"

# 推导缓存：key = 解析后的模型引用，value = ((size, mtime_ns), tag)。
# 没有它，每敲一次回车命名都要读盘解析一遍 pipeline 的 config.yaml。
_EMBEDDING_TAG_CACHE = {}
_EMBEDDING_TAG_LOCK = threading.Lock()

# config 里 `embedding:` 指的是**权重文件**时，身份取它所在的**目录名**
# （本地生成的 config 会把 HF 仓库名替换成权重路径，目录名才是那个仓库名）。
_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx")


def _embedding_tag_cache_clear() -> None:
    """清掉推导缓存（本地模型目录换过、或测试要重算时用）。"""
    with _EMBEDDING_TAG_LOCK:
        _EMBEDDING_TAG_CACHE.clear()


def _as_model_ref(model_ref) -> str:
    """模型引用归一成字符串；其它类型当成「没给」（走默认解析）。"""
    if isinstance(model_ref, Path):
        return str(model_ref).strip()
    if isinstance(model_ref, str):
        return model_ref.strip()
    return ""


def _resolve_diarize_ref(model_ref) -> str:
    """这次实际会交给 pyannote 的模型引用（在线仓库 id / 本地 config.yaml 路径）。"""
    ref = _as_model_ref(model_ref)
    try:
        return str(A.resolve_diarize_model(ref or None) or "")
    except Exception as exc:                 # 解析失败不能影响入库
        _emit_quiet(f"解析分割模型失败（按原样用）：{exc}")
        return ref


def _parse_pipeline_identity(path) -> tuple:
    """从 pipeline 的 `config.yaml` 读 (流水线身份, 嵌入模型身份)；读不到给空串。

    只认 `version:` 与 `pipeline.params.embedding:` 两行 —— 不引 yaml 依赖
    （项目约束是「不引入新依赖」），这两行的格式由 pyannote 自己固定。

    - `version: 3.1.0` → `pyannote/speaker-diarization-3.1`（与旧常量逐字对齐，
      这样默认这条路升级前后身份不变，用户已有的库不会整体失效）；
      没有 version 就退回用 config 所在目录名。
    - `embedding:` 是权重文件（`.bin`/`.safetensors`…）时取**所在目录名**；
      是 HF 仓库 id（`pyannote/wespeaker-...`）时取仓库名 —— 两种写法是同一个
      向量空间，必须给同一个身份。
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return ("", "")
    pipeline = ""
    m = re.search(r"^[ \t]*version[ \t]*:[ \t]*(.*?)[ \t]*$", text, re.M)
    if m:
        raw = m.group(1).strip().strip("\"'")
        if re.fullmatch(r"\d+(\.\d+)*", raw):
            pipeline = ("pyannote/speaker-diarization-"
                        + ".".join(raw.split(".")[:2]))
    if not pipeline:
        pipeline = path.parent.name
    embedding = ""
    m = re.search(r"^[ \t]*embedding[ \t]*:[ \t]*(.*?)[ \t]*$", text, re.M)
    if m:
        value = m.group(1).strip()
        if " #" in value:                    # 行尾注释（不是路径里的 #）
            value = value.split(" #", 1)[0].rstrip()
        value = value.strip().strip("\"'").replace("\\", "/").rstrip("/")
        if value and value.lower() not in ("null", "~", "none"):
            if value.lower().endswith(_WEIGHT_SUFFIXES):
                head = value.rsplit("/", 1)[0] if "/" in value else ""
                value = head or value
            embedding = value.rsplit("/", 1)[-1]
    return (pipeline, embedding)


def _derive_embedding_model_tag(ref: str) -> str:
    """推导标识；推不出来给空串（调用方回退常量）。`ref` 已经解析过。"""
    if not ref:
        return ""
    try:
        is_file = Path(ref).is_file()
    except Exception:
        is_file = False
    if is_file:
        pipeline, embedding = _parse_pipeline_identity(ref)
        if pipeline and embedding:
            return "%s/%s" % (pipeline, embedding)
        if embedding:
            return embedding
        # 配置读得到、但认不出嵌入模型：拿配置路径本身当身份。
        # **不能**回退常量 —— 那等于宣称「它就是 wespeaker」，正是要防的静默混比。
        return ref
    if ref == A.DEFAULT_DIARIZE_MODEL:
        return ""
    # 换成别的在线仓库 / 自定义路径：它必须有自己的身份，否则与默认流水线
    # 混成同一个空间 —— 那正是 D7 要修的「同维不同模型静默混比」。
    return ref


def embedding_model_tag(model_ref=None) -> str:
    """这次分割用的**嵌入模型身份**（写进库的 `embedding_model`）。

    规则（按优先级）：
    1. 解析出来的引用是本地 `config.yaml` → `version` 给出流水线身份、
       `pipeline.params.embedding` 给出嵌入模型身份；
    2. 默认在线仓库（本地没有可用 pipeline）→ 空串 → `EMBEDDING_MODEL_TAG`；
    3. 其它引用（旧版在线流水线 / 自定义目录）→ 引用自身当身份。

    **绝不抛、绝不返回空串**：推导失败只是让身份退化成常量，不影响入库
    （声纹是锦上添花，不能让一次命名失败）。
    """
    ref = _resolve_diarize_ref(model_ref)
    stat_key = None
    try:
        st = Path(ref).stat()
        stat_key = (int(st.st_size), int(st.st_mtime_ns))
    except Exception:
        stat_key = None
    with _EMBEDDING_TAG_LOCK:
        hit = _EMBEDDING_TAG_CACHE.get(ref)
        if hit is not None and hit[0] == stat_key:
            return hit[1]
    try:
        tag = _derive_embedding_model_tag(ref)
    except Exception as exc:
        _emit_quiet(f"推导嵌入模型标识失败（回退常量）：{exc}")
        tag = ""
    tag = (tag or "").strip() or EMBEDDING_MODEL_TAG
    with _EMBEDDING_TAG_LOCK:
        if len(_EMBEDDING_TAG_CACHE) > 64:   # 试过几十个模型也不至于无限涨
            _EMBEDDING_TAG_CACHE.clear()
        _EMBEDDING_TAG_CACHE[ref] = (stat_key, tag)
    return tag


def _work_embedding_tag(work) -> str:
    """这个 work 目录里的聚类质心是**哪个模型**算出来的。

    只有 `state.json` 记得这次用的 `diarize_model`（用户可能改过参数、
    也可能是从别处拷来的检查点）—— 没有就按默认解析走。
    """
    ref = None
    try:
        st = load_state(Path(work)) or {}
        opts = st.get("opts")
        if isinstance(opts, dict):
            ref = opts.get("diarize_model")
    except Exception:
        ref = None
    try:
        return embedding_model_tag(ref)
    except Exception as exc:
        _emit_quiet(f"推导嵌入模型标识失败（回退常量）：{exc}")
        return EMBEDDING_MODEL_TAG


SEP_READY = False
SEP_NOTE = ""
DIARIZE_READY = False
DIARIZE_NOTE = ""

STAGES = ("抽取音频", "人声分离", "说话人分割", "识别中", "完成")


def safe_name(s: str) -> str:
    """把文件名变成可用作目录名的安全字符串（保留中文）。"""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")
    return (s or "unnamed")[:80]


def _detect_separation():
    global SEP_READY, SEP_NOTE
    if not SEP_PYTHON.exists():
        SEP_READY, SEP_NOTE = False, f"分离环境缺失：{SEP_PYTHON}"
        return False
    if not SEP_WORKER.exists():
        SEP_READY, SEP_NOTE = False, f"分离脚本缺失：{SEP_WORKER}"
        return False
    try:
        probe = subprocess.run(
            [str(SEP_PYTHON), "-c",
             "import audio_separator, torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:
        SEP_READY, SEP_NOTE = False, f"分离环境探测失败：{exc}"
        return False
    if probe.returncode != 0:
        SEP_READY = False
        SEP_NOTE = "分离运行时不完整：" + (probe.stderr or "").strip()[:160]
        return False
    SEP_READY = True
    SEP_NOTE = "分离将使用 GPU" if "True" in probe.stdout else "分离将使用 CPU（很慢）"
    return True


def _detect_diarization():
    """真探测 venv-diarize：跑 pyannote_compat.py 自检，而不是只查文件存在。

    旧实现只查 DIARIZE_PYTHON / DIARIZE_WORKER 两个文件在不在，torch 与 pyannote
    一次都不导入，于是环境坏掉时面板照样显示「已就绪」，用户白跑一整趟才发现
    分割被跳过、字幕里一个说话人标签都没有。

    必须走 pyannote_compat.py：裸 ``import pyannote.audio`` 会失败
    （torchaudio 2.11 删掉了 AudioMetaData），那个脚本的 ``__main__`` 里先
    apply_all() 再 import，成功打印两行（pyannote / torch 版本）并退出 0。
    """
    global DIARIZE_READY, DIARIZE_NOTE
    if not A.DIARIZE_PYTHON.exists():
        DIARIZE_READY, DIARIZE_NOTE = False, f"说话人分割环境缺失：{A.DIARIZE_PYTHON}"
        return False
    if not A.DIARIZE_WORKER.exists():
        DIARIZE_READY, DIARIZE_NOTE = False, f"说话人分割脚本缺失：{A.DIARIZE_WORKER}"
        return False
    shim = A.DEFAULT_HOME / "pyannote_compat.py"
    try:
        probe = subprocess.run(
            [str(A.DIARIZE_PYTHON), str(shim)],
            capture_output=True, text=True, timeout=240,
        )
    except Exception as exc:
        DIARIZE_READY, DIARIZE_NOTE = False, f"说话人分割环境探测失败：{exc}"
        return False
    if probe.returncode != 0:
        DIARIZE_READY = False
        detail = (probe.stderr or probe.stdout or "").strip()[:160]
        DIARIZE_NOTE = "说话人分割运行时不完整：" + detail
        return False
    DIARIZE_READY = True
    ver = " ".join((probe.stdout or "").split())[:120]
    # 本地模型齐不齐决定「要不要 token」—— 本机 models\pyannote\ 三个子模型齐全时
    # 走的是本地生成的 config，跟 HuggingFace 授权毫无关系。旧提示把
    # 「不需要做的事」写成了前提条件。
    if A._local_pyannote_pipeline() is not None:
        DIARIZE_NOTE = "使用本地模型，无需 token" + (f"（{ver}）" if ver else "")
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        DIARIZE_NOTE = "说话人分割已就绪（HF_TOKEN 已设置）" + (f"（{ver}）" if ver else "")
    else:
        DIARIZE_NOTE = ("说话人分割已就绪，但未设置 HF_TOKEN —— "
                        "没有本地 pyannote 模型时，在线仓库需要 token"
                        + (f"（{ver}）" if ver else ""))
    return True


def _detect_environments():
    """并发跑两个人声探测再各自 join。

    两个探测都是「起子进程等它退出」的纯阻塞操作，串行跑会让启动白白多等它们
    耗时之和（实测分离 2.5 秒 + 分割 10.3 秒）。两个函数本身保持同步实现，
    只有调用点并发化；它们写的是不同全局（SEP_* / DIARIZE_*），无竞争。
    """
    threads = [threading.Thread(target=_detect_separation),
               threading.Thread(target=_detect_diarization)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ----------------------------------------------------------------------------
# 检查点
# ----------------------------------------------------------------------------

def _opts_key(opts: dict) -> str:
    values = {k: opts.get(k) for k in
                       ("asr_model", "separate", "sep_model", "beam", "glossary", "vad",
                        "diarize", "diarize_model", "num_speakers",
                        "diarize_min_overlap")}
    values["vad"] = opts.get("vad", True)
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def _checkpoint_opts_match(state: dict, opts: dict) -> bool:
    try:
        stored = json.loads(state.get("opts_key", ""))
    except (ValueError, TypeError):
        return False
    if not isinstance(stored, dict):
        return False
    # Before this fix every WebUI run used VAD, regardless of its checkbox.
    # Preserve those checkpoints only when the current request also uses VAD.
    stored.setdefault("vad", True)
    return stored == json.loads(_opts_key(opts))


def load_state(work: Path):
    p = work / "state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(work: Path, state: dict):
    tmp = work / "state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(work / "state.json")


def load_segments(work: Path):
    p = work / "segments.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue       # 坏行本身丢掉即可：断电截断的末行自然也被丢掉，
            #                但中间某一行坏掉不能连累它后面的所有句子
    return out


def append_segment(work: Path, seg: dict):
    with open(work / "segments.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())      # 断电也不丢


def list_resumable():
    """扫描 work 目录，列出可以继续的任务。"""
    out = []
    if not WORK_DIR.exists():
        return out
    for work in sorted(WORK_DIR.iterdir()):
        if not work.is_dir():
            continue
        st = load_state(work)
        if not st or st.get("status") == "done":
            continue
        segs = load_segments(work)
        total = st.get("total_chunks") or 1
        out.append({
            "work": work.name,
            "media": st.get("media", ""),
            "media_name": Path(st.get("media", "")).name,
            "done": len(segs),
            "total": total,
            "percent": round(100.0 * len(segs) / total, 1),
            "out_dir": st.get("out_dir", ""),
            "opts": st.get("opts", {}),
        })
    return out


# ----------------------------------------------------------------------------
# 任务执行
# ----------------------------------------------------------------------------

JOBS = {}
JOBS_LOCK = threading.Lock()
SUBTITLE_OUTPUT_LOCK = threading.Lock()
TASK_QUEUE = queue.Queue()

_model_cache = {"key": None, "model": None}
_model_lock = threading.Lock()
_cuda_state = {"ok": None}
# 导出任务串行化：一次导出会整棵重建 `<作品名>\` 目录，并删掉自己抽出来的
# `raw44k.wav`；两个导出同时跑会互相删对方的输入（`cleanup_intermediates`），
# 也会对着同一个 `.list` 抢 `os.replace`。worker 是单线程、本来就不会并发，
# 这把锁是为「测试/将来有人直接调 api_export+run_export_job」准备的兜底。
_export_lock = threading.Lock()


def _cuda_ok() -> bool:
    if _cuda_state["ok"] is None:
        try:
            import ctranslate2
            _cuda_state["ok"] = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _cuda_state["ok"] = False
    return _cuda_state["ok"]


def _emit(job, line):
    with JOBS_LOCK:
        job["logs"].append(line.rstrip())
        if len(job["logs"]) > 600:
            del job["logs"][:300]


def _set(job, **kw):
    with JOBS_LOCK:
        job.update(kw)


def _job_view(j) -> dict:
    """给前端的状态快照（过滤掉 Popen 句柄等不可序列化的内部字段）。"""
    return {
        "id": j["id"], "name": j.get("name", ""), "media": j.get("media", ""),
        "kind": j.get("kind") or "asr",
        "status": j["status"], "stage": j["stage"],
        "progress": j.get("progress", 0), "eta": j.get("eta"),
        "progress_note": j.get("progress_note", ""),
        "duration": j.get("duration"), "processed_sec": j.get("processed_sec"),
        "logs": j["logs"][-200:], "files": j.get("files", []),
        "error": j.get("error", ""), "out_dir": j.get("out_dir", ""),
        "opts": j.get("opts", {}),
        "can_label_speakers": _can_label_speakers(j),
        # 导出任务的结果快照：`.list` 路径 + 各计数 + 报告（面板靠它渲染）。
        # 别的任务没有这个键，前端按 `undefined` 处理即可。
        "result": j.get("result"),
    }


def stop_requested(job) -> bool:
    with JOBS_LOCK:
        return bool(job.get("stop"))


def abandon_requested(job) -> bool:
    """放弃 = 停止 + 删除检查点。比 stop 更彻底，不可恢复。"""
    with JOBS_LOCK:
        return bool(job.get("abandon"))


def _discard_job(job) -> None:
    """删掉该任务的检查点目录（放弃任务时调用）。

    只删 work 下的中间产物；已经写出的字幕/文本在 output 目录，不受影响。

    **导出任务一个字节都不删。** 导出任务不拥有检查点：它只是借某个任务
    （`job["work"]`）的标注结果去落盘，而 `_discard_job` 是按
    `job["media"]` 推检查点目录的 —— 导出任务的 `media` 就是那个 ASR 任务的
    原片，于是「放弃导出」会顺手删掉别人（可能已经 `status=done`）的
    `speakers.json` / `segments.jsonl` / `diarization.json`：用户命名好的
    角色全丢，且不可恢复。放弃导出只该丢弃这次导出，不该碰任何别的任务。
    """
    if (job.get("kind") or "") in ('export', 'download'):
        return
    media = job.get("media") or ""
    if not media:
        return
    try:
        import stage_jobs
        target = (Path(job['work']) if job.get('work') else
                  stage_jobs.work_path(sys.modules[__name__], job)
                  if job.get('opts', {}).get('mode') in ('extract', 'separate', 'recognize', 'diarize')
                  or job.get('opts', {}).get('backend') == 'qwen'
                  else _legacy_work_path(Path(media)))
        if not job.get('work'):
            saved_media = (load_state(target) or {}).get('media')
            if saved_media and Path(saved_media).resolve() != Path(media).resolve():
                return
        target.resolve().relative_to(WORK_DIR.resolve())
        if target.resolve() == WORK_DIR.resolve():
            return
        with JOBS_LOCK:
            shared = any(other.get('id') != job.get('id')
                         and other.get('status') == 'run' and other.get('work')
                         and Path(other['work']).resolve() == target.resolve()
                         for other in JOBS.values())
        if shared:
            _emit(job, '检查点正被另一项任务使用，已保留共享数据。')
            return
        shutil.rmtree(target,
                      ignore_errors=True)
    except Exception:
        pass


def _get_model(model_ref, device, compute_type, log):
    key = (model_ref, device, compute_type)
    with _model_lock:
        if _model_cache["key"] == key:
            return _model_cache["model"]
        from faster_whisper import WhisperModel
        log(f"加载模型：{model_ref}")
        m = WhisperModel(model_ref, device=device, compute_type=compute_type)
        _model_cache.update(key=key, model=m)
        return m


def _pick_stem(produced, out_dir: Path, keyword: str):
    """audio-separator 各版本返回的可能是文件名或绝对路径，统一处理。"""
    for name in produced:
        p = Path(name)
        if not p.is_absolute():
            p = out_dir / p.name
        if keyword in p.name.lower() and p.exists():
            return p
    return None


def _existing_exports(outdir: Path):
    """列出已经导出过的试听文件。"""
    if not outdir.exists():
        return []
    out = []
    for f in sorted(outdir.glob("*.mp3")):
        kind = "人声" if "人声" in f.name else ("伴奏" if "伴奏" in f.name else "文件")
        out.append({"name": f.name, "path": str(f), "kind": kind})
    return out


def _export_mp3(src: Path, dest: Path):
    """把分离出来的一轨转成 mp3，方便试听（体积远小于 wav）。"""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(src), "-b:a", "320k", str(dest)],
        check=True, capture_output=True,
    )


def separate_vocals(wav_in: Path, sep_dir: Path, stem: str, model_filename: str,
                    log, job=None) -> tuple:
    """独立进程抽出人声轨。

    返回 (人声16k wav, 导出文件列表)。
    同时把人声/伴奏导出成 mp3 放到 output\\分离\\<文件名>\\，供用户先试听效果。
    """
    if not SEP_PYTHON.exists():
        raise RuntimeError(f"分离环境不存在：{SEP_PYTHON}")
    sep_dir.mkdir(parents=True, exist_ok=True)
    log(f"分离模型：{model_filename}（独立进程）")

    started = time.time()
    proc = subprocess.Popen(
        [str(SEP_PYTHON), str(SEP_WORKER), str(wav_in), str(sep_dir), model_filename],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )
    if job is not None:
        _set(job, _proc=proc)

    # 边等边刷新已用时间，并允许中途停止
    out_lines, err_lines = [], []
    import threading as _th

    def _reader(pipe, sink):
        for ln in pipe:
            sink.append(ln)

    t_out = _th.Thread(target=_reader, args=(proc.stdout, out_lines), daemon=True)
    t_err = _th.Thread(target=_reader, args=(proc.stderr, err_lines), daemon=True)
    t_out.start(); t_err.start()

    while proc.poll() is None:
        if stop_requested(job):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError("__STOPPED__")
        time.sleep(1.0)
        _set(job, progress_note=f"分离中，已用时 {A.fmt_hms(time.time() - started)}")

    t_out.join(timeout=5); t_err.join(timeout=5)
    # /api/stop 会直接结束子进程，循环会以「进程退出」的方式结束而不是
    # 走上面的 __STOPPED__ 分支，所以这里要统一补一次判定。
    if stop_requested(job):
        raise RuntimeError("__STOPPED__")
    _set(job, _proc=None)
    noise = ("WARNING: The installed PyTorch", "Failed to load cublas",
             "Failed to load cufft", "Failed to load cudart",
             "Please follow https://onnxruntime.ai")
    vocal = None
    for line in out_lines:
        if line.startswith("RESULT="):
            vocal = Path(line[7:].strip())
        elif line.strip() and not line.strip().startswith(noise):
            log("  " + line.strip())
    if proc.returncode != 0 or vocal is None or not vocal.exists():
        # 过滤掉 tqdm 进度条这类噪音行
        tail = [l.strip() for l in err_lines
                if l.strip() and "%|" not in l][-3:]
        raise RuntimeError("分离失败：" + (" | ".join(tail) or f"退出码 {proc.returncode}"))

    # 导出可试听的 mp3（人声 + 伴奏）
    exported = []
    try:
        root = (Path(job['out_dir']) / '分离') if job and job.get('out_dir') else SEP_OUT_ROOT
        media = job.get('media') if job else None
        folder = available_output_stem(root, safe_name(stem), ('',), media or wav_in)
        outdir = root / folder
        outdir.mkdir(parents=True)
        instrumental = _pick_stem(out_lines and [vocal.name] or [], sep_dir, "instrumental")
        # 重新在目录里找伴奏轨
        for f in sep_dir.glob("*.wav"):
            if "instrumental" in f.name.lower():
                instrumental = f
                break
        for track, label in ((vocal, "人声"), (instrumental, "伴奏")):
            if track and track.exists():
                dest = outdir / f"{stem}_{label}.mp3"
                _export_mp3(track, dest)
                exported.append({"name": dest.name, "path": str(dest), "kind": label})
                log(f"  已导出试听：{dest.name}")
    except Exception as exc:
        log(f"  [!] 导出试听文件失败（不影响识别）：{exc}")

    # 归一化成 16kHz 单声道喂给 ASR（-nostdin 理由同上）
    normalized = sep_dir / "vocals_16k.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(vocal), "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le", str(normalized)],
        check=True, capture_output=True,
    )
    log("人声轨已归一化为 16kHz 单声道")
    return normalized, exported


CHECKPOINT_FILES = ("state.json", "segments.jsonl", "raw16k.wav", "vocals_16k.wav",
                    "diarization.json")
# 聚类质心 sidecar：重新分割后 SPEAKER_xx 的含义变了，旧质心跟新聚类不是一回事，
# 所以它跟 diarization.json 同生共死（`_wipe_annotation` 里删，见那里的注释）。
EMBEDDINGS_FILE = "embeddings.npz"


def _wipe_annotation(work) -> None:
    """只清标注状态与片段缓存。

    重新分割后 SPEAKER_xx 的含义变了，旧名字再套上去就是错的，必须连缓存一起清。
    聚类质心 sidecar（`embeddings.npz`）**也必须一起清**：那是「当时的 SPEAKER_01」
    的质心，跟这次重新分割出来的 SPEAKER_01 不是同一个东西 —— 留着它会把声纹
    配到另一个人的名字上。speakers.json 一删，写在里面的 embedding/match 也没了。

    必须持该 work 的标注锁：api_assign/api_reassign 正在同一目录上做「读-改-写」，
    不持锁重叠时会「刚保存的名字被删掉」或「删完又写回半份旧状态」。
    """
    work = Path(work)
    with _label_lock(work):
        try:
            (work / "speakers.json").unlink()
        except OSError:
            pass
        try:
            (work / EMBEDDINGS_FILE).unlink()
        except OSError:
            pass
        shutil.rmtree(work / "clips", ignore_errors=True)


def _wipe_checkpoint(work) -> None:
    """参数或输入变了：清掉检查点，连标注状态与片段缓存一起清。

    speakers.json 必须一起清 —— 否则新分割出来的 SPEAKER_00 会套上上一轮的
    角色名，字幕会被写成错的角色。

    这里持的是 _label_lock（RLock）：下面调用 _wipe_annotation 会在同一线程
    再获取一次，普通 Lock 会自死锁。
    """
    work = Path(work)
    with _label_lock(work):
        for name in CHECKPOINT_FILES:
            try:
                (work / name).unlink()
            except OSError:
                pass
        _wipe_annotation(work)


def load_cluster_embeddings(work) -> dict:
    """读 `work\\embeddings.npz` -> {聚类键: 单位向量}。

    矩阵是分割子进程落下的**原始**质心（未归一化），归一化与「丢掉补位零行」
    都由 `speaker_registry.extract_centroids` 负责 —— 这里不重复实现一遍规则。

    读不到 / 损坏 / 版本不含在盘上 → 空字典，**绝不抛**：
    声纹是锦上添花，拿不到就退化成纯手动命名，不能让面板变成 500。
    """
    path = Path(work) / EMBEDDINGS_FILE
    if not path.is_file():
        return {}
    try:
        with np.load(str(path)) as data:
            files = set(data.files)
            if not {"labels", "centroids"} <= files:
                return {}
            labels = [str(x) for x in data["labels"]]
            centroids = data["centroids"]
    except Exception as exc:
        _emit_quiet(f"读取聚类质心失败（忽略）：{exc}")
        return {}
    # 逐行对齐是匹配的地基：行数与标签数不等就整份不信（labels 多于行数时
    # pyannote 补的是全零行，`labels[i]` 与第 i 行不是一回事）。
    # 正常情况下 worker 已经拦住了这种文件；这里防的是盘上被改坏/截断的 sidecar。
    try:
        rows = int(np.asarray(centroids).shape[0])
    except Exception:
        return {}
    if rows != len(labels):
        _emit_quiet("聚类质心行数与标签数不一致（忽略这份质心）："
                    "%d 行 / %d 标签" % (rows, len(labels)))
        return {}
    try:
        return speaker_registry.extract_centroids(centroids, labels)
    except Exception as exc:
        _emit_quiet(f"解析聚类质心失败（忽略）：{exc}")
        return {}


def _store_cluster_embeddings(work, labels) -> dict:
    """把每个聚类的质心写进 `speakers.json` 的 `clusters[<id>]["embedding"]`。

    `speaker_labels.load_labels` 从阶段 1 起就保留 name 以外的键，这个口就是为它留的。
    返回写成功的 {聚类: 向量}（调用方要拿它去匹配/入库，不必再读一次盘）。

    **没有做分割、或这次没拿到质心时不写任何 embedding** —— 尤其是**不**去猜：
    宁可让面板显示「这个聚类没有声纹」，也不能把上一次的 embedding 留在
    新聚类上面（那会把声纹配到另一个人的名字上）。所以上一轮的 `embedding`/`match`
    一律先清掉，只留这一次真正读到的。

    全程持该 work 的标注锁，并且**锁内重新 load**（不拿调用方那份快照写盘）：
    调用方手里的 `labels` 是「它读盘那一刻」的快照，而 `run_job` 从读盘到调这里
    中间隔着整个分割子进程（几分钟）。用户完全可能在这中间命名
    （`api_assign` 持锁读-改-写、字幕已按新名字重写、库里已入库）。
    拿那份旧快照 save 回去就是一次 lost update：刚命的名被回退，
    于是 speakers.json / 字幕 / 声纹库三处互相矛盾。
    参数 `labels` 只为兼容老调用方保留，**不参与**写盘。
    """
    work = Path(work)
    # 读-改-写必须整体持锁，否则别人（api_assign 命名 / 面板保存）在中间改了状态，
    # 我们就会拿旧内容覆盖回去 —— 那次「刚命的名」当场丢失，而字幕与声纹库里
    # 都已经是新名字，三处状态互相矛盾（lost update）。
    #
    # 分两段拿锁、不嵌套成一整段：`_label_lock` 是可重入的，但把整段包进一个 with
    # 只会让内层多一级缩进，可读性更差；两段之间那个窗口很短，而且第二段是
    # **重读后逐键合并**（不是覆盖），所以即使窗口里有人保存过也不会丢东西。
    # 锁序只有「先 _label_lock，再 speaker_labels._SAVE_LOCK」这一种，不会死锁。
    with _label_lock(work):
        vecs = load_cluster_embeddings(work)
        fresh = speaker_labels.load_labels(work)
    clusters = fresh.setdefault("clusters", {})
    for rec in clusters.values():
        if isinstance(rec, dict):
            rec.pop("embedding", None)
            rec.pop("match_at_enroll", None)
    stored = {}
    for cluster, vec in vecs.items():
        rec = clusters.get(cluster)
        if not isinstance(rec, dict):
            rec = {"name": "" if rec is None else str(rec)}
            clusters[cluster] = rec
        rec["embedding"] = [float(x) for x in np.asarray(vec).reshape(-1)]
        stored[cluster] = vec
    # 第二段：重读一次状态再合并写回。
    # 重读是必须的 —— `fresh` 只是第一段那一刻的内容，而第一段到这里之间
    # 用户可能刚保存了别的改动（转录可以跑几分钟，这不是理论上的窗口）。
    # 刚读到的 `fresh` 里真正的改动只有 embedding/match_at_enroll 两类键，所以把它们
    # 逐键并进重读出来的状态即可，既不会把别人的改动覆盖掉，也不会留下上一轮
    # 的质心。
    with _label_lock(work):
        current = speaker_labels.load_labels(work)
        cur_clusters = current.setdefault("clusters", {})
        for cluster, rec in clusters.items():
            if not isinstance(rec, dict):
                cur_clusters[cluster] = rec
                continue
            target = cur_clusters.get(cluster)
            if not isinstance(target, dict):
                target = {"name": "" if target is None else str(target)}
                cur_clusters[cluster] = target
            target.pop("embedding", None)
            target.pop("match_at_enroll", None)
            if "embedding" in rec:
                target["embedding"] = rec["embedding"]
        speaker_labels.save_labels(work, current)
    return stored


def _media_suffix(media):
    identity = os.path.normcase(str(Path(media).resolve()))
    return hashlib.sha256(os.fsencode(identity)).hexdigest()[:10]


def available_output_stem(directory, stem, extensions, media):
    """Choose a name for a group of outputs written by the single task worker."""
    def occupied(candidate):
        return any(os.path.lexists(directory / (candidate + ext)) for ext in extensions)

    if not occupied(stem):
        return stem
    base = f"{stem[:180]}__{_media_suffix(media)}"
    candidate = base
    index = 2
    while occupied(candidate):
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def publish_subtitles(out, stem, language, media, labels, segments):
    """Allocate the subtitle pair while also excluding speaker-panel writes."""
    with SUBTITLE_OUTPUT_LOCK:
        stem = available_output_stem(out, stem, (f'.{language}.srt', f'.{language}.txt'), media)
        srt, txt = out / f'{stem}.{language}.srt', out / f'{stem}.{language}.txt'
        speaker_labels.rewrite_outputs(labels, segments, srt, txt)
    return stem, srt, txt


def _legacy_work_path(media):
    name = safe_name(media.stem)
    original = WORK_DIR / name
    distinct = WORK_DIR / f"{name[:100]}__{_media_suffix(media)}"
    if distinct.is_dir():
        return distinct
    state = load_state(original) or {}
    owner = state.get("media") or speaker_labels.load_labels(original).get("media")
    if owner and Path(owner).resolve() != media.resolve():
        return distinct
    return original


def run_job(job):
    import stage_jobs
    mode, backend, language = stage_jobs.validate(job['opts'])
    if mode != 'asr' or backend == 'qwen':
        return stage_jobs.run(sys.modules[__name__], job)
    media = Path(job["media"])
    opts = job["opts"]
    log = lambda s: _emit(job, s)

    out_dir = Path(job["out_dir"]) if job.get("out_dir") else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    work = _legacy_work_path(media)
    work.mkdir(parents=True, exist_ok=True)
    state = load_state(work)

    # 源文件或参数变了就重来，否则复用检查点
    try:
        st_size = media.stat().st_size
        st_mtime = media.stat().st_mtime
        st_mtime_ns = media.stat().st_mtime_ns
    except OSError:
        raise RuntimeError(f"读不到输入文件：{media}")

    same = (state and state.get("media") == str(media)
            and state.get("media_size") == st_size
            and state.get("media_mtime") == st_mtime
            and state.get("media_mtime_ns", st_mtime_ns) == st_mtime_ns
            and _checkpoint_opts_match(state, opts))

    if not same:
        if state:
            log("输入或参数已变化，重新开始（旧检查点已清除）")
        _wipe_checkpoint(work)
        state = {
            "media": str(media), "media_size": st_size, "media_mtime": st_mtime,
            "media_mtime_ns": st_mtime_ns,
            "opts_key": _opts_key(opts), "opts": opts,
            "out_dir": str(out_dir), "status": "running",
            "total_chunks": 0, "duration": 0.0,
        }
        save_state(work, state)
    else:
        log(f"发现检查点：已完成 {len(load_segments(work))}/{state.get('total_chunks', '?')} 段，继续")

    _set(job, work=str(work), out_dir=str(out_dir))

    def finish_if_cancelled() -> bool:
        """放弃优先于停止。返回 True 表示本任务到此结束。

        「放弃」会额外删除检查点，所以必须由 worker 来做——
        请求线程不删，避免和正在写盘的 worker 抢同一个目录。
        """
        if abandon_requested(job):
            _discard_job(job)
            _set(job, status="abandoned", stage="已放弃",
                 progress_note="", eta=None)
            log("任务已放弃，检查点已删除（已生成的字幕文件不受影响）。")
            return True
        if stop_requested(job):
            try:
                state["status"] = "stopped"
                save_state(work, state)
            except Exception:
                pass
            _set(job, status="stopped", stage="已停止", progress_note="")
            return True
        return False

    if finish_if_cancelled():
        return

    # ---- 1) 抽取音频（可复用）----
    if not shutil.which("ffmpeg"):
        raise RuntimeError("找不到 ffmpeg")
    raw_wav = work / "raw16k.wav"
    if raw_wav.exists() and same:
        duration = state.get("duration") or A.audio_duration(raw_wav)
        log(f"复用已抽取的音频，时长 {A.fmt_hms(duration)}")
        _set(job, stage="抽取音频", progress=0.10)
    else:
        _set(job, stage="抽取音频", progress=0.02)
        log("抽取 16kHz 单声道音频 ...")
        duration = A.extract_audio(media, raw_wav)
        state["duration"] = duration
        save_state(work, state)
        log(f"抽取完成，时长 {A.fmt_hms(duration)}")
        _set(job, progress=0.10)

    if finish_if_cancelled():
        return

    # ---- 2) 可选人声分离（可复用）----
    vocals_wav = work / "vocals_16k.wav"
    sep_root = Path(job['out_dir']) / '分离'
    sep_outdir = Path(state['separation_dir']) if state.get('separation_dir') else None
    exported = []
    if opts.get("separate") and opts.get("sep_model"):
        if sep_outdir and sep_outdir.parent.resolve() == sep_root.resolve():
            exported = _existing_exports(sep_outdir)
        if vocals_wav.exists() and same and exported:
            log("复用已分离的人声轨（检查点）")
            src_wav = vocals_wav
            log(f"已有试听文件 {len(exported)} 个：{sep_outdir}")
        else:
            _set(job, stage="人声分离", progress=0.12, progress_note="分离中…")
            log("开始人声分离（首次使用该模型会先下载权重）...")
            log(f"试听文件将导出到：{sep_root}")
            try:
                src_wav, exported = separate_vocals(
                    raw_wav, work / "sep", media.stem, opts["sep_model"], log, job)
            except RuntimeError as exc:
                if "__STOPPED__" in str(exc):
                    _set(job, _proc=None)
                    if abandon_requested(job):
                        _discard_job(job)
                        _set(job, status="abandoned", stage="已放弃",
                             progress_note="")
                        log("任务已放弃，检查点已删除。")
                    else:
                        state["status"] = "stopped"
                        save_state(work, state)
                        _set(job, status="stopped", stage="已停止",
                             progress=0.12, progress_note="")
                        log("已在人声分离阶段停止。分离结果未完成，下次会重新分离。")
                    return
                raise
            _set(job, progress_note="")
            if exported:
                state["separation_dir"] = str(Path(exported[0]["path"]).parent)
                save_state(work, state)
            try:
                shutil.move(str(src_wav), str(vocals_wav))
            except Exception:
                pass
            src_wav = vocals_wav if vocals_wav.exists() else src_wav
        if exported:
            _set(job, files=exported + job.get("files", []))
        _set(job, progress=0.45)

        # 仅分离模式：到此为止，不进识别
        if opts.get("mode") == "separate":
            state["status"] = "done"
            save_state(work, state)
            _set(job, status="done", stage="完成", progress=1.0, eta=0.0)
            log(f"仅分离模式，已导出 {len(exported)} 个试听文件，跳过识别。")
            return
    else:
        src_wav = raw_wav
        _set(job, progress=0.45)

    # ---- 2.5) 可选说话人分割（可复用）----
    dia_results = None
    if opts.get("diarize"):
        dia_path = work / "diarization.json"
        if dia_path.exists() and same:
            try:
                dia_results = json.loads(dia_path.read_text(encoding="utf-8"))
                log(f"复用已完成的说话人分割结果（{len(dia_results)} 段）")
            except Exception as exc:
                log(f"[!] 说话人分割检查点损坏，将重新分割：{exc}")
                dia_results = None
        if dia_results is None:
            # 复用检查点上重新分割：聚类含义已变，旧标注与旧质心必须作废。
            #
            # **只能在这里清**，两个方向都有硬理由：
            # * 放到分割之后（旧实现的位置）会连**刚写好的新质心**一起删掉
            #   —— `_wipe_annotation` 删的正是 sidecar，而子进程已经在
            #   `run_diarization` 里把它写好了。表现是质心永远进不了标注状态，
            #   而且因为文件此刻已不存在，连「读取失败」那句告警都不会打印（全静默）。
            # * 不清则更糟：旧 sidecar 是「上一次的 SPEAKER_01」的质心，
            #   跟这次重新分割出来的 SPEAKER_01 不是同一个人，留着就是拿别人的
            #   声纹去配名字。
            #
            # 复用已有 diarization.json 时**不清**：那些聚类键的含义没变，
            # 质心仍然有效。
            _wipe_annotation(work)
            _set(job, stage="说话人分割", progress=0.46,
                 progress_note="说话人分割中…")
            log("开始说话人分割（pyannote）...")
            dargs = types.SimpleNamespace(
                diarize_model=opts.get("diarize_model") or A.DEFAULT_DIARIZE_MODEL,
                device="cuda" if _cuda_ok() else "cpu",
                num_speakers=opts.get("num_speakers") or None,
            )
            emb_path = work / EMBEDDINGS_FILE
            try:
                dia_results = A.run_diarization(
                    src_wav, dargs, out_json=dia_path,
                    on_proc=lambda p: _set(job, _proc=p),
                    should_stop=lambda: (stop_requested(job)
                                         or abandon_requested(job)),
                    out_embeddings=emb_path)
            except RuntimeError as exc:
                if "__STOPPED__" in str(exc):
                    _set(job, _proc=None, progress_note="")
                    if not finish_if_cancelled():
                        _set(job, status="stopped", stage="已停止",
                             progress_note="")
                    return
                log(f"[!] 说话人分割异常，将输出不带标签：{exc}")
                dia_results = None
            except Exception as exc:
                log(f"[!] 说话人分割异常，将输出不带标签：{exc}")
                dia_results = None
            if dia_results:
                speakers = _dia_speakers(dia_results)
                log(f"说话人分割完成：{len(dia_results)} 段，"
                    f"识别到 {len(speakers)} 个说话人")
            else:
                log("[!] 说话人分割无结果，字幕将不带说话人标签。"
                    "详细原因见服务端控制台；可用 jisheng.cmd --check 自检环境。")
            _set(job, progress_note="", progress=0.48)

    # 说话人分割结束后落一份标注骨架，让面板一打开就有聚类列表
    if dia_results:
        try:
            speakers = _dia_speakers(dia_results)
            labels = speaker_labels.bootstrap(
                work, job.get("id", ""), str(media),
                [c for c in speakers if c and c != "?"],
                "separated" if (work / "vocals_16k.wav").is_file() else "media")
            # 质心进标注状态（声纹匹配的原料）。拿不到就什么都不写，不报错。
            vecs = _store_cluster_embeddings(work, labels)
            if vecs:
                log(f"聚类质心已写入标注状态：{len(vecs)} 个")
            elif (work / EMBEDDINGS_FILE).is_file():
                log("[!] 聚类质心读取失败或形状不可用，本次仅手动命名。")
        except Exception as exc:
            log(f"[!] 写入标注骨架失败（不影响识别，已跳过）：{exc}")

    if finish_if_cancelled():
        return

    # ---- 3) 识别（分块 + 检查点）----
    _set(job, stage="识别中", progress=0.48)
    model_ref = A.resolve_model(opts.get("asr_model"))
    device = "cuda" if _cuda_ok() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = _get_model(model_ref, device, compute, log)
    log(f"设备：{device} / {compute}")

    # 模型加载可能几十秒，加载完再确认一次是否已被要求结束
    if finish_if_cancelled():
        return

    glossary = []
    gpath = opts.get("glossary")
    if gpath:
        gp = Path(gpath)
        if gp.is_file():
            glossary = A.load_glossary(gp)
            log(f"术语表：{gp.name}（{len(glossary)} 条）")
        else:
            log(f"[!] 术语表不存在，已忽略：{gpath}")

    audio = A.read_wav_float32(src_wav)
    chunks = (A.vad_chunks(audio, vad_min_silence=500) if opts.get("vad", True)
              else [{"start": 0, "end": len(audio)}])
    if not chunks:
        _set(job, status="error", stage="完成", progress=1.0,
             error="没有检测到语音段。可以试着关闭 VAD 或换一段素材。")
        return

    # VAD 可能是整段最耗时的一步（长片尤其），跑完立刻确认一次
    if finish_if_cancelled():
        return

    state["total_chunks"] = len(chunks)
    save_state(work, state)

    done = load_segments(work)
    # Empty transcriptions still consume a VAD chunk. Persist the cursor separately
    # and recover appended segments if a crash happened before state was saved.
    saved_indices = [s["chunk_index"] for s in done
                     if isinstance(s.get("chunk_index"), int)]
    if saved_indices or "next_chunk" in state:
        start_i = max(state.get("next_chunk", 0), max(saved_indices, default=-1) + 1)
    else:
        # Older checkpoints have timestamps but no chunk indices. Their last
        # subtitle's end identifies the consumed chunks even when earlier ones
        # produced no subtitle.
        last_end = max((s["end"] for s in done), default=0.0)
        start_i = sum(ch["end"] / 16000.0 <= last_end + 0.000001 for ch in chunks)
    start_i = min(max(0, start_i), len(chunks))
    if start_i:
        log(f"从第 {start_i + 1} / {len(chunks)} 段继续")

    started = time.time()
    base = 0.48 + 0.47 * (start_i / len(chunks))
    _set(job, progress=base)

    for i in range(start_i, len(chunks)):
        if stop_requested(job) or abandon_requested(job):
            if not finish_if_cancelled():
                _set(job, progress=0.48 + 0.47 * (i / len(chunks)))
            else:
                log(f"已在第 {i} / {len(chunks)} 段结束本任务。")
            return

        ch = chunks[i]
        seg = A.transcribe_chunk(model, audio, ch, opts.get("beam") or 5, glossary)
        if seg:
            seg["chunk_index"] = i
            if dia_results:
                A.assign_speakers([seg], dia_results,
                                  opts.get("diarize_min_overlap") or 0.3)
            append_segment(work, seg)
            done.append(seg)
            log(f"[{A.fmt_ts(seg['start'])}] "
                f"{A.prefix_with_speaker(seg['text'], seg.get('speaker'))}")

        state["next_chunk"] = i + 1
        save_state(work, state)

        frac = (i + 1) / len(chunks)
        elapsed = time.time() - started
        done_here = (i + 1) - start_i
        if done_here > 0 and frac > 0.001:
            total_est = elapsed / done_here * (len(chunks) - start_i)
            eta = max(0.0, total_est - elapsed)
        else:
            eta = None
        _set(job,
             progress=0.48 + 0.47 * frac,
             eta=eta,
             processed_sec=sum(s["end"] - s["start"] for s in done),
             duration=duration)

    # ---- 4) 写出 ----
    if not done:
        _set(job, status="error", stage="完成", progress=1.0,
             error="没有识别到任何语音内容。")
        return

    stem, srt_path, txt_path = publish_subtitles(
        out_dir, media.stem, 'ja', media, speaker_labels.load_labels(work), done)
    _set(job, subtitle_stem=stem)

    state["status"] = "done"
    state["subtitle_stem"] = stem
    state["out_dir"] = str(out_dir)
    if exported:
        state["separation_dir"] = str(Path(exported[0]["path"]).parent)
    save_state(work, state)

    all_files = list(job.get("files", []))          # 含已导出的分离试听文件
    all_files += [{"name": srt_path.name, "path": str(srt_path), "kind": "字幕"},
                  {"name": txt_path.name, "path": str(txt_path), "kind": "文本"}]
    _set(job, status="done", stage="完成", progress=1.0, eta=0.0,
         progress_note="", files=all_files)
    log(f"完成：{len(done)} 条字幕 -> {srt_path}")


def worker_loop():
    while True:
        job = TASK_QUEUE.get()
        if job is None:
            break
        with JOBS_LOCK:
            abandoned = bool(job.get("abandon"))
            stopped = bool(job.get("stop"))
        if abandoned:
            # 还没开始跑就被放弃了：直接删检查点，别浪费一次加载
            _discard_job(job)
            job.update(status="abandoned", stage="已放弃")
            continue
        if stopped:
            job.update(status="stopped", stage="已停止")
            continue
        _set(job, status="run")
        try:
            # 按 kind 分发。没有 kind 的是既有 ASR 任务（`_enqueue` 不写这个字段），
            # 所以缺省必须是 `run_job` —— 把老任务判成未知 kind 会让它们全都不跑。
            kind = job.get("kind")
            if kind == "export":
                run_export_job(job)
            elif kind == 'download':
                import download_jobs
                download_jobs.run(sys.modules[__name__], job)
            elif kind in (None, "", "asr"):
                run_job(job)
            else:
                # 未知 kind 不能静默当 ASR 任务跑（那会白跑一整趟识别）：
                # 明确报错，让人一眼看见是分发没接对。
                raise RuntimeError(f"未知的任务类型：{kind!r}")
        except Exception as exc:
            _emit(job, traceback.format_exc(limit=3))
            _set(job, status="error", error=f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------
# 说话人标注接口（模块级实现，便于脱离 HTTP 测试）
# ----------------------------------------------------------------------------

class ApiError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.msg = msg
        self.code = code


# ----------------------------------------------------------------------------
# 导出训练集（设计 §11 的 POST /api/export、§12 的磁盘预检）
#
# 导出是**分钟级**动作（HQ 音源要现抽 + 分离，实测冷跑 188 秒），所以它一律走
# 既有任务队列：`api_export` 只把任务塞进 `TASK_QUEUE` 就返回，`worker_loop`
# 按 `job["kind"]` 分发到 `run_export_job`。进度用既有的 `_set`、日志用既有的
# `_emit`、停止/放弃用既有的 `stop_requested` / `abandon_requested` ——
# 这里刻意**不另起**一套队列或进度机制，否则 `/api/status`、停止、放弃
# 三处都要各写一遍导出专属分支。
# ----------------------------------------------------------------------------

def training_export_root() -> Path:
    """训练集导出根：`OUTPUT_ROOT\\说话人\\`。每次读 OUTPUT_ROOT（测试会改它）。"""
    return Path(OUTPUT_ROOT) / TRAIN_EXPORT_DIRNAME


def _export_target(payload: dict):
    """定位导出对象。返回 `(work, media, labels, stem)`；失败抛 ApiError。

    与 `_api_open` 分开写，因为判据不同：那边要的是「有句子可改判」，
    这边要的是「有分割结果可导出」—— 没有分割结果就没有聚类、没有角色名，
    导出只会得到一整个 `_未命名` 目录。校验顺序刻意是
    「任务存在 -> 目录存在 -> 有分割结果 -> 有识别结果」，
    这样 404 与 409 的边界对调用方是稳定的。
    """
    job_id = payload.get("job") or ""
    work_name = (payload.get("work_name") or "").strip()
    job = None
    if job_id:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            raise ApiError("没有这个任务", 404)
        # 有 job 时**只能**用任务自己记的 work 目录：`work_name` 在面板上
        # 是「作品名」输入框，不是检查点名。拿它去找目录会在
        # 「作品名 != 检查点名」时把请求打成 404（面板上这两个名字本来就可以不同）。
        work = _job_work_dir(job, job_id, None)
    else:
        target_work = payload.get("work") or work_name
        if "/" in target_work or "\\" in target_work:
            raise ApiError("检查点名不能带路径分隔符", 400)
        work = _job_work_dir(None, job_id, target_work)
    if work is None:
        raise ApiError("找不到该任务的中间产物目录（可能已被放弃）", 404)
    if not (work / "diarization.json").is_file():
        raise ApiError("该任务没有做说话人分割，没有可导出的说话人", 409)
    if not _coerce_times(load_segments(work), "识别"):
        raise ApiError("该任务还没有识别结果", 409)
    labels = speaker_labels.load_labels(work)
    # 面板上现填的作品名要落进标注状态：导出目录名、`.list` 名都跟着它走，
    # 下次打开面板还得记得。只在真的变了时写盘（写盘要持锁）。
    if work_name and work_name != speaker_labels.get_work_name(labels):
        with _label_lock(work):
            labels = speaker_labels.load_labels(work)
            speaker_labels.set_work_name(labels, work_name)
            speaker_labels.save_labels(work, labels)
    st = load_state(work) or {}
    media = ((job or {}).get("media") or labels.get("media")
             or st.get("media") or "")
    return work, media, labels, (Path(media).stem if media else work.name)


def api_export(payload: dict) -> tuple:
    """`POST /api/export`：校验后入队，立即返回任务 id。

    请求线程**只读**、只入队：真正的音源准备（可能是一整趟 HQ 分离）与落盘
    全部留给 worker。返回里没有 `.list` / 计数 —— 那些要等 `job["result"]`
    （`/api/status` 拿得到），面板也是轮询 `result` 的。
    """
    try:
        if not isinstance(payload, dict):
            raise ApiError("请求体必须是对象", 400)
        sr_raw = payload.get("sr")
        if sr_raw is None or sr_raw == "":
            sr = te.DEFAULT_SAMPLE_RATE
        else:
            # bool 是 int 的子类，但 `"sr": true` 是手写坏掉的请求，不是 1 Hz
            if isinstance(sr_raw, bool):
                raise ApiError("采样率不合法：%r" % (sr_raw,), 400)
            try:
                sr = int(sr_raw)
            except (TypeError, ValueError):
                raise ApiError("采样率不合法：%r" % (sr_raw,), 400)
        if sr not in te.ALLOWED_SAMPLE_RATES:
            raise ApiError("采样率只支持 %s，收到 %r"
                           % (list(te.ALLOWED_SAMPLE_RATES), sr_raw), 400)
        source = payload.get("source") or "hq"
        if source not in EXPORT_SOURCE_MODES:
            raise ApiError("音源只支持 %s，收到 %r"
                           % (list(EXPORT_SOURCE_MODES), source), 400)
        export_format = payload.get("format", "gpt_sovits")
        if export_format not in te.EXPORT_FORMATS:
            raise ApiError("训练格式不支持：%r" % (export_format,), 400)
        # 校验顺序：采样率/音源 -> 目标。这样「sr 坏」永远是 400，
        # 不会因为在没有 work 的情况下先判目标而变成 404。
        work, media, labels, stem = _export_target(payload)
        source_state = load_state(work) or {}
        language = source_state.get('opts', {}).get('language') or labels.get('language')
        if language == 'auto':
            language = source_state.get('detected_language') or 'und'
        try:
            language = te.export_language(language, export_format)
        except te.ExportError as exc:
            raise ApiError(str(exc), 400) from exc
        export_base = (Path(source_state.get('out_dir') or OUTPUT_ROOT)
                       if source_state.get('fingerprint') else OUTPUT_ROOT)
        jid = uuid.uuid4().hex[:12]
        job = {
            "id": jid, "kind": "export", "name": stem, "media": str(media),
            "work": str(work), "stage": "准备导出", "status": "wait",
            "progress": 0.0, "eta": None, "duration": None, "processed_sec": None,
            "logs": [f"已入队：导出训练集（{stem}）"],
            "files": [], "error": "", "stop": False, "abandon": False,
            "out_dir": str(export_base),
            "out_root": str(export_base / TRAIN_EXPORT_DIRNAME),
            "work_name": (payload.get("work_name") or "").strip(),
            "sr": sr, "source": source, "format": export_format, "language": language,
            "drop_short": bool(payload.get("drop_short", False)),
            "opts": {"sr": sr, "source": source, "format": export_format,
                     "drop_short": bool(payload.get("drop_short", False))},
        }
        with JOBS_LOCK:
            JOBS[jid] = job
        # 请求线程到此结束：真正的音源准备（可能是一整趟 HQ 分离，冷跑实测 188 秒）
        # 与落盘全在 worker 里。此刻被放弃的任务由 `worker_loop` 直接标 abandoned。
        TASK_QUEUE.put(job)
        return {"id": jid, "status": "wait", "kind": "export"}, 200
    except ApiError as exc:
        return {"error": exc.msg}, exc.code


def _export_cancel_requested(job):
    """传给 `train_source` / `train_export` 的 `should_stop`。

    放弃优先于停止（与 `run_job.finish_if_cancelled` 同一读法），但对
    「要不要中断当前 ffmpeg」这个问题来说两者都是「是」。
    """
    return stop_requested(job) or abandon_requested(job)


def _finish_export_cancelled(job, message: str, *, out_root=None,
                             work_name=None) -> None:
    """导出被中止 / 放弃时的收尾：与既有任务同一套状态字。

    日志必须说**事实**：`export_work` 是「先删旧 `.list`/报告，再整棵重建
    `<作品名>\\`」，所以中断时盘上确实可能留着半截 wav。以前这里写「已作废」
    就完事，文件却还在 —— 现在改成：能安全清掉的就清掉并报告释放量，清不掉
    的把「为什么没清」写出来（用户才不会以为盘上干净了）。

    `out_root` / `work_name` 只有 `_run_export_job_locked` 解析出作品名之后才
    传得进来；在这之前被停（还没碰过盘）时两者都是 None，此时**一个字节都不动**。
    """
    manifest_name = "`.list`" if job.get("format", "gpt_sovits") == "gpt_sovits" else "标注清单"
    _emit(job, message)
    if abandon_requested(job):
        _set(job, status="abandoned", stage="已放弃", progress_note="", eta=None,
             _export_silence_finder=None, _export_source=None)
        note = ""
        if out_root is not None and work_name:
            try:
                info = te.cleanup_partial_export(out_root, work_name)
            except Exception as exc:                    # noqa: BLE001 - 清理失败不影响状态
                info = {"removed": False, "code": "error", "freed_mb": 0.0,
                        "reason": str(exc)}
            if info.get("code") == "removed":
                note = ("已删除本次导出的半截产物（%.1f MB）；没有删除任何别的东西。"
                        % (info.get("freed_mb", 0.0),))
            elif info.get("code") == "no_dir":
                note = "本次导出没有在导出根里留下产物。"
            else:
                note = "已保留本次导出目录（%s）。" % (info.get("reason") or "未清理")
        else:
            note = "本次导出还没开始写盘，盘上没有本次的产物。"
        _emit(job, note)
        return
    _set(job, status="stopped", stage="已停止", progress_note="", eta=None,
         _export_silence_finder=None, _export_source=None)
    _emit(job, f"导出已停止。已经写出的片段可能不完整，请检查{manifest_name}或重新导出。")


def run_export_job(job, source_picker=None) -> None:
    """worker 侧的导出任务。

    阶段与进度：
      0.05 准备音源（`train_source.pick_source`，HQ 走现抽 + 分离）
      0.55 磁盘预检（按计划体积算，不够就抛，**一个字节都不写**）
      0.60~0.95 切分落盘
      1.00 完成 + 清理中间产物

    `source_picker` 是注入点（默认 `train_source.pick_source`），签名
    `(work, media, mode, model=..., should_stop=..., progress=...) -> (Path, kind)`。
    `job["_export_source"]` 与 `job["_export_silence_finder"]` 是给测试用的
    旁路：给了就不再准备音源 / 不跑真 ffmpeg 静音探测。

    预检与落盘**共用同一个静音探测器与同一份计划**（`te.plan_silence_finder`
    + `precheck_work(...)["plan"]` 交给 `export_work(plan=...)`）：两边各建一个
    探测器、各算一遍计划时，预检报的 `n_planned` / `estimated_bytes` 与真正落盘的
    不是一回事（实测 2 vs 3 条片段）。
    """
    with _export_lock:
        _run_export_job_locked(job, source_picker=source_picker)


def _run_export_job_locked(job, source_picker=None) -> None:
    out_root = Path(job.get("out_root") or training_export_root())
    work = Path(job["work"])
    media = job.get("media") or ""
    work_name = (job.get("work_name") or "").strip()
    sr = int(job.get("sr") or te.DEFAULT_SAMPLE_RATE)
    source_mode = job.get("source") or "hq"
    export_format = job.get("format", "gpt_sovits")
    drop_short = bool(job.get("drop_short"))
    log = lambda s: _emit(job, s)                          # noqa: E731
    cancelled = lambda: _export_cancel_requested(job)      # noqa: E731

    if cancelled():
        _finish_export_cancelled(job, "导出在开始前已被停止。")
        return

    labels = speaker_labels.load_labels(work)
    if job.get("language"):
        labels["language"] = job["language"]
    if not work_name:
        work_name = speaker_labels.get_work_name(labels)
    if not work_name:
        _set(job, status="error",
             error="作品名为空：请先在上方「作品名」里填一个（导出目录与 `.list` 都用它命名）。")
        return
    work_component = te.sanitize_component(work_name)
    previous_names = [work_component, work_component + te.REPORT_SUFFIX]
    previous_names.extend(work_component + suffix for suffix in te.MANIFEST_SUFFIXES.values())
    if work_component and any(os.path.lexists(out_root / name) for name in previous_names):
        run_root = out_root / ("export-" + uuid.uuid4().hex[:12])
        while os.path.lexists(run_root):
            run_root = out_root / ("export-" + uuid.uuid4().hex[:12])
        out_root = run_root
    segments = _coerce_times(load_segments(work), "识别")

    # 解析出作品名之后，中断收尾才有能力定位「本次导出自己的那棵目录」。
    def finish_cancelled(message):
        _finish_export_cancelled(job, message, out_root=out_root,
                                 work_name=work_name)

    # ---- 1) 音源 ----
    picked = job.get("_export_source")
    if picked:
        src, source_kind = Path(picked[0]), picked[1]
    else:
        _set(job, stage="准备音源", progress=0.05, progress_note="准备音源…")
        log("准备音源：%s（%s）" % (source_mode,
                                   "从原片全采样率重新分离" if source_mode == "hq"
                                   else "复用既有产物"))
        picker = source_picker or ts.pick_source
        try:
            src, source_kind = picker(work, Path(media), source_mode,
                                      model=ts.DEFAULT_MODEL,
                                      should_stop=cancelled,
                                      progress=lambda m: log(m))
        except ts.SourceCancelled as exc:
            finish_cancelled("音源准备被中止：%s" % exc)
            return
        except Exception as exc:                        # noqa: BLE001 - 收进任务错误
            _set(job, status="error", stage="失败",
                 error="%s: %s" % (type(exc).__name__, exc))
            log("音源准备失败：%s" % exc)
            return
    warn = te.source_warning(source_kind)
    if warn:
        log("[!] 音源降级（%s）：%s" % (source_kind, warn))
    log("音源：%s（%s）" % (src, source_kind))

    if cancelled():
        finish_cancelled("音源准备完成后收到停止请求。")
        return

    # ---- 2) 磁盘预检（干跑，不写盘）----
    # 只建**一个**静音探测器：它同时给预检与落盘用。各建一个时，预检与落盘
    # 算的是两份不同的计划（拆长判据完全取决于静音段），预检报的
    # `n_planned`/`estimated_bytes` 就不是真正要写的量。
    _set(job, stage="磁盘预检", progress=0.55, progress_note="检查磁盘空间…")
    finder = te.plan_silence_finder(
        src, should_stop=cancelled,
        silence_finder=job.get("_export_silence_finder") or None)
    try:
        pre = te.precheck_work(segments, labels, src, out_root,
                               work_name=work_name, sr=sr,
                               source_kind=source_kind, drop_short=drop_short,
                               silence_finder=finder, should_stop=cancelled,
                               format=export_format)
    except te.ExportCancelled as exc:
        finish_cancelled("导出被中止：%s" % exc)
        return
    except Exception as exc:                            # noqa: BLE001
        _set(job, status="error", stage="失败",
             error="%s: %s" % (type(exc).__name__, exc))
        log("导出前检查未通过：%s" % exc)
        return
    log("预计写入 %.1f MB（%s 条片段，%s 秒），磁盘剩余 %.1f MB"
        % (pre["estimated_bytes"] / 1e6, pre["n_planned"],
           pre["planned_seconds"], pre["free_bytes"] / 1e6))

    if cancelled():
        finish_cancelled("导出在落盘前已被停止。")
        return

    # ---- 3) 切分落盘 ----
    # 复用预检算出来的那份计划（`plan=pre["plan"]`）：预检报的数字与真正落盘的
    # 产物出自同一次 `_plan_export`，不再是「两次各算一遍」。
    plan = pre.get("plan")
    if plan is None:
        # 只有测试替身（或将来别的调用方）才会走到这里：生产路径的 precheck_work
        # 一定带 plan。这时落盘只能自己再算一遍，必须说出来，别静默漂开。
        log("[!] 预检没有返回计划，落盘将重新计算一份（预检数字可能不等于落盘）")
    _set(job, stage="导出中", progress=0.60, progress_note="切分片段…")
    try:
        result = te.export_work(segments, labels, src, out_root,
                                work_name=work_name, sr=sr,
                                source_kind=source_kind, drop_short=drop_short,
                                silence_finder=finder, should_stop=cancelled,
                                plan=plan, format=export_format)
    except te.ExportCancelled as exc:
        finish_cancelled("导出被中止：%s" % exc)
        return
    except Exception as exc:                            # noqa: BLE001
        _set(job, status="error", stage="失败",
             error="%s: %s" % (type(exc).__name__, exc))
        log("导出失败：%s" % exc)
        return

    _set(job, progress=0.95, stage="导出中", progress_note="整理产物…")
    log("切分完成：%d 条片段（%d 条进清单，%d 条 _短句，%d 条 _未命名，%.1f MB）"
        % (result["n_clips"], result["n_list"], result["n_short"],
           result["n_unknown"], result["total_mb"]))
    if result.get("warning"):
        log("[!] %s" % result["warning"])

    # ---- 4) 清理中间产物（在导出**成功之后**；失败时留着便于查原因）----
    freed = 0.0
    try:
        freed = ts.cleanup_intermediates(work)
    except Exception as exc:                            # noqa: BLE001 - 清理失败不算导出失败
        log("清理中间产物失败（不影响已导出的训练集）：%s" % exc)
    if freed:
        log("已清理中间产物，释放 %.1f MB" % freed)
    result["freed_mb"] = round(float(freed), 2)

    manifest = result["manifest_path"]
    files = [{"name": Path(manifest).name, "path": manifest,
              "kind": "训练集清单"}]
    _set(job, status="done", stage="完成", progress=1.0, eta=0.0,
         progress_note="", result=result, files=files,
         _export_silence_finder=None, _export_source=None)
    log("导出完成：%s" % manifest)


# 每个 work 目录一把锁。标注状态是「load → 改 → save」三步，而 HTTP 处理是多线程的：
# 两个并发的 assign/reassign 会让一个线程的 os.replace 撞上另一个线程的读，
# 在 Windows 上表现为 PermissionError: WinError 5，请求直接 500。
_LABEL_LOCKS = {}
_LABEL_LOCKS_GUARD = threading.Lock()

# /api/speakers 逐句返回原文，超长句截断（面板只用来区分「哪一句标错了」）。
SEGMENT_TEXT_LIMIT = 200


def _label_lock(work):
    # 键必须归一化：同一个目录的 "D:\work\EP01" / "d:\WORK\ep01" / 相对路径变体
    # 各拿一把锁的话，互斥当场失效。normcase 在 Windows 上会小写化并把 / 转成 \。
    #
    # 这里刻意不做「清理不再使用的锁」（F7 明确要求）：把一把正在被持有、
    # 或即将被持有的锁从字典里删掉，另一个请求就会给同一目录新建第二把锁，
    # 两个线程同时进临界区 —— 互斥失效比多占几个字典槽位严重得多。
    key = os.path.normcase(os.path.abspath(str(Path(work))))
    with _LABEL_LOCKS_GUARD:
        lock = _LABEL_LOCKS.get(key)
        if lock is None:
            # 返回 RLock：_wipe_checkpoint 持锁后会调 _wipe_annotation，同线程嵌套获取。
            lock = threading.RLock()
            _LABEL_LOCKS[key] = lock
        return lock


def _job_work_dir(job, job_id=None, work_name=None):
    """定位标注对象目录：显式 work 名 > 任务记录的 work > 按 media 推。"""
    if work_name:
        p = WORK_DIR / safe_name(work_name)
        return p if p.is_dir() else None
    if job:
        w = job.get("work")
        if w:
            return Path(w)
        media = job.get("media") or ""
        if media:
            opts = job.get("opts", {})
            if (opts.get("mode") in ("extract", "separate", "recognize", "diarize")
                    or opts.get("backend") == "qwen"):
                import stage_jobs
                p = stage_jobs.work_path(sys.modules[__name__], job)
                return p if p.is_dir() else None
            p = _legacy_work_path(Path(media))
            return p if p.is_dir() else None
    return None


def _can_label_speakers(job):
    if (job.get("kind") in ("export", "download")
            or job.get("opts", {}).get("mode") in ("extract", "separate")
            or job.get("status") not in ("done", "stopped", "error")):
        return False
    work = _job_work_dir(job)
    if work is None:
        return False
    try:
        rows = json.loads((work / "diarization.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return False
        rows = _coerce_times(rows, "说话人分割")
    except (OSError, ValueError, ApiError):
        return False
    return any(row["end"] > row["start"] and isinstance(row.get("speaker"), str)
               and row["speaker"].strip() not in ("", "?") for row in rows)


def _api_open(payload: dict) -> dict:
    """解析请求目标。失败抛 ApiError。**不修改任何状态** —— 改状态的地方必须持锁。

    这里与 api_speakers 各自实现目标定位，二者语义确实不同、不要合并：
    本函数是「要写东西」的入口，必须有句子才谈得上改判断（没识别结果就是 409）；
    api_speakers 是「要看面板」的入口，只有分割结果、还没识别结果时也允许打开。
    """
    job_id = payload.get("job") or ""
    job = None
    if job_id:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            raise ApiError("没有这个任务", 404)
    work = _job_work_dir(job, job_id, payload.get("work"))
    if work is None:
        raise ApiError("找不到该任务的中间产物目录（可能已被放弃）", 404)
    segs = _coerce_times(load_segments(work), "识别")
    if not segs:
        raise ApiError("该任务还没有识别结果", 409)
    labels = speaker_labels.load_labels(work)
    # 已知聚类 = 状态表里已有的 ∪ 分割结果里的。写入时只认这些键，
    # 否则请求里带一个拼错的键就会在面板上凭空多出一个幽灵角色。
    known = set(labels.get("clusters") or {})
    dia_path = work / "diarization.json"
    if dia_path.is_file():
        try:
            dia = json.loads(dia_path.read_text(encoding="utf-8"))
            known |= set(_dia_speakers(dia))
        except Exception:
            pass
    known.discard("?")
    st = load_state(work) or {}
    media = ((job or {}).get("media") or labels.get("media")
             or st.get("media") or "")
    if media and not labels.get("media"):
        labels["media"] = str(media)          # 让检查点自描述，下次不必再猜
    out_dir = Path((job or {}).get("subtitle_out_dir") or st.get("subtitle_out_dir")
                   or (job or {}).get("out_dir") or st.get("out_dir") or OUTPUT_ROOT)
    language = st.get('opts', {}).get('language') or 'ja'
    stem = (job or {}).get('subtitle_stem') or st.get('subtitle_stem')
    new_pair = not stem and (st.get('fingerprint')
                             or (job and job.get('work') and job.get('status') != 'done'))
    stem = stem or (Path(media).stem if media else work.name)
    return {"work": work, "segs": segs, "labels": labels, "job": job,
            "known": known, "out_dir": out_dir, "stem": stem, "language": language,
            "new_pair": new_pair, "media": media or work}


def _cluster_matches(work_dir, labels, clusters) -> dict:
    """给每个聚类算跨集「建议」：`{聚类: {work,page,score,band,auto} | None}`。

    **只读**：不写标注状态、不写库（调用方是只读接口 `/api/speakers`）。
    建议永远只是建议 —— 写名字只能走 `api_assign`（用户确认）。

    三条规则：
    * 向量只认标注状态里的 `clusters[<id>]["embedding"]`（Task 5 写入的单位向量）；
      没有 / 坏掉 → 该聚类 `None`，**不猜**。
    * **过滤「自己」**：已经叫 X 的聚类不再被建议改成 X。不过滤的话它的自我
      匹配永远是 ~1.0，面板上会显示一句「建议：X（相似度 1.00）而它本来就叫 X」
      的废话（Task 5 审查 M4 变异抓到过的同一类错误）。
    * 过滤之后**重算一对一**：被过滤掉的候选如果还占着一对一的名额，
      同集另一个聚类就永远拿不到它本该拿到的建议。

    外加一条模型身份闸门（D7）：库里的向量按**嵌入模型空间**分开存，而这批质心
    的空间由这个 work 的 `state.json` 决定。两边的空间不同、且这个作品在
    **本集这个空间里一行向量都没有**时，一个建议都不给 —— 光靠 `match_vectors`
    判不出来它只知道库那一侧的可用性，A 空间的质心会照算 B 空间的余弦。

    注意「库级模型 != 本集模型」**不等于**「无可用向量」（Task 7 审查 P1）：用户
    换回旧模型时，旧空间的行必须还能用，所以这里用
    `speaker_registry.work_rows_in_space()` 按空间问「到底有没有东西可比」，
    而不是拿两个标识直接判死。

    返回里的 `page/score/band` 是**能落地的那一个**候选：一对一配上的就是它，
    没配上（同集别的聚类分更高）就是分数最高的那个、且 `auto` 为 None —
    前端据此决定「预填」还是「只列成候选」。
    """
    empty = {c: None for c in clusters}
    work = speaker_labels.get_work_name(labels)
    if not work:
        return empty
    store = labels.get("clusters") or {}
    embeds = {}
    for c in clusters:
        rec = store.get(c)
        if not isinstance(rec, dict) or rec.get("embedding") is None:
            continue
        vec = _check_embedding(rec.get("embedding"))
        if vec is not None:
            embeds[c] = vec
    if not embeds:
        return empty
    try:
        lib_tag = (speaker_registry.load_registry().get("embedding_model") or "").strip()
    except Exception:
        lib_tag = ""
    cur_tag = _work_embedding_tag(work_dir)
    if lib_tag and cur_tag and lib_tag != cur_tag:
        try:
            rows = speaker_registry.work_rows_in_space(work=work, model=cur_tag)
        except Exception:
            rows = 0
        if not rows:
            _emit_quiet("声纹库里没有「%s」这个模型空间下的向量（库级是 %s），"
                        "本次不给建议" % (cur_tag, lib_tag))
            return empty
    try:
        raw = speaker_registry.match_vectors(
            work=work,
            clusters={c: {"embedding": v} for c, v in embeds.items()},
            model=cur_tag)
    except Exception as exc:
        _emit_quiet(f"声纹匹配失败（本次不给建议）：{exc}")
        return empty

    filtered = {}
    for c in clusters:
        current = (speaker_labels.cluster_name(labels, c) or "").strip()
        cands = [x for x in ((raw.get(c) or {}).get("candidates") or [])
                 if x.get("page") and x.get("page") != current]
        filtered[c] = cands
    # 一对一只在**还没有名字**的聚类之间做：已有名字的聚类根本不会被预填
    # （面板里名字优先于建议），让它占着名额只会把本该给未命名聚类的建议挤掉
    # —— 表现就是「另一个聚类明明 ≥ auto 阈值却没有预填」。
    # 而且必须在**过滤之后**重算：拿过滤前的 auto_page 来用，等于那个名额
    # 还被已经过滤掉的候选占着。
    pool = {}
    for c in clusters:
        named = (speaker_labels.cluster_name(labels, c) or "").strip()
        pool[c] = [] if named else (filtered.get(c) or [])
    auto_pages = speaker_registry.greedy_pair(pool)

    out = {}
    for c in clusters:
        cands = filtered.get(c) or []
        if not cands:
            out[c] = None
            continue
        want = auto_pages.get(c)
        chosen = next((x for x in cands if x.get("page") == want), None)
        if chosen is None:
            chosen = cands[0]
            want = None
        out[c] = {"work": work,
                  "page": str(chosen.get("page") or ""),
                  "score": round(float(chosen.get("score") or 0.0), 6),
                  "band": str(chosen.get("band") or "none"),
                  "auto": want}
    return out


def api_speakers(job_id=None, work_name=None):
    """列出可标注的说话人 + 3 条试听片段（片段按需生成并缓存）。

    这里与 _api_open 各自实现目标定位，二者语义确实不同、不要合并：
    本函数是只读的面板入口，只要有分割结果就能列出来（哪怕还没有识别结果）；
    _api_open 是要写状态的入口，没有句子就没有可改判的对象，直接 409。
    """
    try:
        if job_id:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                raise ApiError("没有这个任务", 404)
        else:
            job = None
        work = _job_work_dir(job, job_id, work_name)
        if work is None:
            raise ApiError("找不到该任务的中间产物目录（可能已被放弃）", 404)
        dia_path = work / "diarization.json"
        if not dia_path.is_file():
            raise ApiError("该任务没有做说话人分割，无法标注", 409)
        try:
            dia = json.loads(dia_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ApiError(f"说话人分割检查点损坏：{exc}", 409)
        if not isinstance(dia, list) or not all(isinstance(d, dict) for d in dia):
            raise ApiError("说话人分割检查点格式不对（应为对象数组）", 409)
        dia = _coerce_times(dia, "说话人分割")
        segs = _coerce_times(load_segments(work), "识别")
        labels = speaker_labels.load_labels(work)
        clusters = [c for c in _dia_speakers(dia) if c != "?"]
        speaker_labels.ensure_clusters(labels, clusters)

        # 逐句信息：i 就是 segments.jsonl 的行号，api_reassign 直接吃它。
        # 归属按当前标注状态算（单句改判 > 聚类），所以改判之后那句会出现在新聚类下面。
        # 注意分桶要用「聚类键」而不是解析后的显示名 —— 卡片的 id 是聚类键，
        # 用名字分桶会让所有已命名的聚类都拿到空列表。
        #
        # per_cluster 是**当前标注归属**口径，卡片头的 n_segments/total_s 也用它：
        # 卡片写几句就必须能展开几行，否则会出现「写 2 句、展开 0 行」。
        # 试听片段（clips）保持**原始分割口径**，那是刻意的：片段要从分割结果里
        # 挑时间段，改判一句话不该让试听内容跟着变。
        per_cluster = {c: [] for c in clusters}
        unknown = []
        others = []                 # 改判到「新建角色」等 clusters 之外的键
        for i, s in enumerate(segs):
            rec = {"i": i, "start": s["start"], "end": s["end"],
                   "text": (s.get("text") or "").strip()[:SEGMENT_TEXT_LIMIT]}
            key = (labels.get("segment_overrides") or {}).get(str(i), s.get("speaker"))
            if not key or key == "?":
                if key == "?":
                    unknown.append(rec)
                continue
            if key in per_cluster:
                per_cluster[key].append(rec)
            elif key in (labels.get("clusters") or {}):
                # 「新建角色…」建出来的键（new:<名字>）不在分割结果里，所以没有卡片；
                # 但它必须可撤销，否则用户误点一次就再也改不回来（单向门）。
                others.append(rec)
            # 既不在 clusters 里、也不是已登记的聚类键（手改坏了 speakers.json）：
            # 列表里没有对应的下拉目标，列出来只会让用户点不动，故不展示。

        src = speaker_clips.preview_source(work)
        # 跨集建议：只读地算一遍（不写状态、不写库），随聚类一起返回。
        matches = _cluster_matches(work, labels, clusters)
        items = []
        for c in clusters:
            # 注意：这里**不再**按 s["speaker"] 统计句数。句数与 total_s 一律取
            # per_cluster（当前标注归属），否则改判后会出现「卡片写 2 句、展开 0 行」。
            # 只有试听候选仍然按分割结果挑（见下面 pick_clip_candidates 的注释）。
            cands = speaker_clips.pick_clip_candidates(segs, dia, c,
                                                      limit=speaker_clips.CLIPS_PER_CLUSTER)
            if src is not None and cands:
                try:
                    cands = speaker_clips.ensure_clips(work, c, cands, src)
                except Exception as exc:
                    # 片段没生成成功就别把候选交给前端：否则面板会渲染 404 的 <audio>
                    _emit_quiet(f"试听片段生成失败（{c}）：{exc}")
                    cands = []
            rows = per_cluster[c]
            items.append({
                "id": c,
                "label": A.fmt_speaker_label(c),
                "name": speaker_labels.cluster_name(labels, c) or "",
                # 建议（可能为 None）：面板只**预填** auto 已经一对一配上的那个，
                # suggest 档只列成候选，none 留空 —— 绝不静默替换用户的名字。
                "match": matches.get(c),
                "n_segments": len(rows),          # 与 segments 同口径
                "total_s": round(sum(r["end"] - r["start"] for r in rows), 1),
                "segments": rows,
                "clips": [{"n": x["n"], "start": round(x["start"], 2),
                           "end": round(x["end"], 2), "dur": x["dur"],
                           "overlap": x["overlap"]} for x in cands],
            })
        unknown.sort(key=lambda r: r["i"])
        others.sort(key=lambda r: r["i"])
        return {"job": job_id or "", "work": work.name,
                "work_name": speaker_labels.get_work_name(labels),
                "source": src.name if src is not None else "",
                "unknown_segments": len(unknown),
                "unknown": {"n": len(unknown), "segments": unknown},
                "others": {"n": len(others), "segments": others},
                "clusters": items}, 200
    except ApiError as exc:
        return {"error": exc.msg}, exc.code


def _emit_quiet(msg):
    """调试信息只走服务端控制台，不影响接口返回。"""
    try:
        print(f"[speakers] {msg}", flush=True)
    except Exception:
        pass


def api_clip_path(cluster, n, job_id=None, work_name=None):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    job = None
    if job_id:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
    work = _job_work_dir(job, job_id, work_name)
    if work is None:
        return None
    p = speaker_clips.clip_path(work, cluster, n)
    return p if p.is_file() else None


def api_workdirs():
    """列出所有含说话人分割结果的检查点，供重启后继续标注。

    按 diarization.json 的 mtime 倒序 —— 「最近优先」要的是最近动过的检查点，
    不是目录名的逆字典序（后者会让用户想找刚标注的那一集时得自己翻）。
    mtime 相同（同一批拷进来的）用目录名兜底，保证顺序稳定。
    """
    items = []
    if WORK_DIR.is_dir():
        found = []
        for d in WORK_DIR.iterdir():
            if not d.is_dir():
                continue
            dia = d / "diarization.json"
            if not dia.is_file():
                continue
            try:
                mtime = dia.stat().st_mtime
            except OSError:
                continue
            found.append((mtime, d))
        for _, d in sorted(found, key=lambda it: (-it[0], it[1].name)):
            labels = speaker_labels.load_labels(d)
            items.append({"work": d.name,
                          "work_name": speaker_labels.get_work_name(labels),
                          "named": sum(1 for r in (labels.get("clusters") or {}).values()
                                       if isinstance(r, dict) and (r.get("name") or "").strip())})
    return {"items": items}, 200


# ----------------------------------------------------------------------------
# 角色库（全局声纹库）：作品 / 角色 / 入库集数 + 改名 + 删除
#
# 底层一律直接用 speaker_registry 的既有函数（list_registry / rename_page /
# forget_page），**不在这里重写一遍**库的语义 —— 这里只做三件事：
# 参数校验（缺参数/未知目标给 4xx，不是 500）、把库的结构整理成面板要的形状、
# 把「删了什么」如实回报给用户。
# ----------------------------------------------------------------------------

def _episode_jobs(rec) -> list:
    """一条角色记录里出现过的**集**（= 任务 id）去重排序。"""
    out = []
    for ep in (rec.get("episodes") or []):
        if not isinstance(ep, dict):
            continue
        jid = ep.get("job") or ""
        if isinstance(jid, str) and jid and jid not in out:
            out.append(jid)
    return sorted(out)


def api_registry():
    """`GET /api/registry`：作品 / 角色 / 入库集数。**只读**。"""
    try:
        reg = speaker_registry.load_registry()
        rows = speaker_registry.list_registry()
    except Exception as exc:                 # list_registry 自己容错，这里是兜底
        return {"error": f"读取角色库失败：{exc}"}, 500
    works = {}
    all_jobs = set()
    for row in rows:
        work = row["work"]
        w = works.get(work)
        if w is None:
            w = {"work": work, "model": row.get("model") or "",
                 "vectors_usable": bool(row.get("vectors_usable")),
                 "n_pages": 0, "n_episodes": 0, "n_vectors": 0,
                 "total_s": 0.0, "updated_at": "", "pages": [],
                 "_jobs": set()}
            works[work] = w
        jobs = _episode_jobs(row)
        page = {"page": row["page"],
                "n_enroll": int(row.get("n_enroll") or 0),
                "n_episodes": len(jobs), "jobs": jobs,
                "total_s": round(float(row.get("total_s") or 0.0), 3),
                "n_vectors": int(row.get("n_vectors") or 0),
                "vectors_usable": bool(row.get("vectors_usable")),
                "updated_at": row.get("updated_at") or ""}
        w["pages"].append(page)
        w["n_pages"] += 1
        w["n_vectors"] += page["n_vectors"]
        w["total_s"] += page["total_s"]
        w["_jobs"].update(jobs)
        all_jobs.update(jobs)
        # 时间戳是 ISO 文本：字典序即时间序
        w["updated_at"] = max(w["updated_at"], page["updated_at"])
    items = []
    for work in sorted(works):
        w = works[work]
        w["n_episodes"] = len(w.pop("_jobs"))
        w["total_s"] = round(w["total_s"], 1)
        w["pages"].sort(key=lambda p: p["page"])
        items.append(w)
    return {"embedding_model": reg.get("embedding_model") or "",
            "dim": int(reg.get("dim") or speaker_registry.DEFAULT_DIM),
            "thresholds": dict(reg.get("thresholds")
                               or speaker_registry.DEFAULT_THRESHOLDS),
            "n_works": len(items),
            "n_pages": sum(w["n_pages"] for w in items),
            "n_episodes": len(all_jobs),
            "works": items}, 200


def api_registry_health():
    """`GET /api/registry/health`：声纹体检（阈值可分性）。**只读**。

    原样转发 `speaker_registry.health_check()` 的结果 —— 那是纯计算，只经
    `load_registry` / `load_vectors` 两个读入口。这里**一件写的事都不做**：
    体检回答的是「现在这套阈值有没有依据」，顺手把 `suggested` 存进库就等于
    在用户没同意的情况下改掉了匹配规则（那是 `save_registry` 的语义，得由一次
    明确的操作来触发）。

    空库 / 只有一个角色时 `same_min` / `diff_max` / `gap` / `suggested` 都是
    **None 而不是 0**，这是接口契约的一部分：面板必须照这个区分渲染
    （0.000 看着像「算过了，很像」）。中间层不许补默认值，也不许把
    `ok=False`（= 阈值没依据）当成请求失败 —— HTTP 仍然是 200。

    库里数据坏到 `health_check` 自己抛（理论上不会，`load_*` 都容错）时，
    只让这一块报错，不要把整个角色库面板拖崩。
    """
    try:
        return speaker_registry.health_check(), 200
    except Exception as exc:
        return {"error": f"声纹体检失败：{exc}"}, 500


def _registry_page_exists(work, page) -> bool:
    reg = speaker_registry.load_registry()
    pages = ((reg.get("works") or {}).get(work) or {}).get("pages") or {}
    return page in pages


def _registry_str(payload, key) -> str:
    """取一个必须是非空字符串的参数；不合格抛 ApiError（4xx，不是 500）。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"缺少 {key}（必须是非空字符串）", 400)
    return value.strip()


def api_registry_rename(payload: dict):
    """`POST /api/registry/rename {"work","from","to"}`：角色改名（并入同名角色）。"""
    if not isinstance(payload, dict):
        return {"error": "请求体必须是对象"}, 400
    try:
        work = _registry_str(payload, "work")
        old = _registry_str(payload, "from")
        new = _registry_str(payload, "to")
        if old == new:
            raise ApiError("新名字与旧名字相同", 400)
        if not _registry_page_exists(work, old):
            raise ApiError(f"角色库里没有「{work} / {old}」", 404)
        merged = _registry_page_exists(work, new)
        out = speaker_registry.rename_page(work=work, old=old, new=new)
    except ApiError as exc:
        return {"error": exc.msg}, exc.code
    if not out.get("ok"):
        return {"error": "改名失败（参数不合法）"}, 400
    return {"ok": True, "work": work, "from": old, "to": new,
            "moved": int(out.get("moved") or 0),
            "merged": bool(merged)}, 200


def api_registry_forget(payload: dict):
    """`POST /api/registry/forget {"work","page"}`：忘掉一个角色（**不可逆**）。

    只删这一个 `(作品, 角色)` —— 向量与统计都是 `forget_page` 按这对键删的，
    这里绝不自己做「顺手清理」（同作品别的角色、别的作品的同名人都不许碰）。
    """
    if not isinstance(payload, dict):
        return {"error": "请求体必须是对象"}, 400
    try:
        work = _registry_str(payload, "work")
        page = _registry_str(payload, "page")
        if not _registry_page_exists(work, page):
            raise ApiError(f"角色库里没有「{work} / {page}」", 404)
        out = speaker_registry.forget_page(work=work, page=page)
    except ApiError as exc:
        return {"error": exc.msg}, exc.code
    if not out.get("ok"):
        return {"error": "删除失败（参数不合法）"}, 400
    return {"ok": True, "work": work, "page": page,
            "removed": int(out.get("removed") or 0)}, 200


def _write_back(ctx) -> tuple:
    language = ctx.get('language') or 'ja'
    try:
        out = ctx["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        with SUBTITLE_OUTPUT_LOCK:
            stem = ctx["stem"]
            state = None
            if ctx.get("new_pair"):
                current = load_state(ctx["work"]) or {}
                recorded_out = current.get("subtitle_out_dir") or current.get("out_dir")
                if (current.get("subtitle_stem") and recorded_out
                        and Path(recorded_out).resolve() == out.resolve()):
                    stem = current["subtitle_stem"]
                else:
                    stem = available_output_stem(out, stem, (f".{language}.srt", f".{language}.txt"),
                                                 ctx["media"])
                    if current and not current.get("subtitle_stem"):
                        state = current
            srt, txt = out / f"{stem}.{language}.srt", out / f"{stem}.{language}.txt"
            n = speaker_labels.rewrite_outputs(ctx["labels"], ctx["segs"], srt, txt)
            if state is not None:
                state.update(subtitle_stem=stem, subtitle_out_dir=str(out))
                save_state(ctx["work"], state)
    except Exception as exc:
        return {"error": f"重写字幕失败：{exc}"}, 500
    return {"ok": True, "rewritten": True, "lines": n,
            "srt": str(srt), "txt": str(txt)}, 200


def _reload_under_lock(ctx) -> dict:
    """持锁后重新读一遍状态，避免拿锁外读到的旧内容写回去。"""
    ctx["labels"] = speaker_labels.load_labels(ctx["work"])
    if not ctx["labels"].get("media"):
        # 检查点自描述：认识这一集的 work 目录只有 state.json 记得 media，
        # 补进去以后就不必每次都靠猜（_api_open 里的内存补写会被这里重新读到旧值）。
        st = load_state(ctx["work"]) or {}
        if st.get("media"):
            ctx["labels"]["media"] = str(st["media"])
    ctx["segs"] = _coerce_times(load_segments(ctx["work"]), "识别")
    known = set(ctx["labels"].get("clusters") or {})
    dia_path = ctx["work"] / "diarization.json"
    if dia_path.is_file():
        try:
            known |= set(_dia_speakers(
                json.loads(dia_path.read_text(encoding="utf-8"))))
        except Exception:
            pass
    known.discard("?")
    ctx["known"] = known
    return ctx


def _coerce_times(items, what: str) -> list:
    """把 start/end 强制成 float；坏值一律当检查点损坏（409），而不是让它变成 500。"""
    out = []
    for it in items:
        try:
            rec = dict(it)
            rec["start"] = float(rec["start"])
            rec["end"] = float(rec["end"])
        except (TypeError, ValueError, KeyError):
            raise ApiError(f"{what}检查点损坏：时间段不是数字", 409)
        if not (math.isfinite(rec["start"]) and math.isfinite(rec["end"])):
            raise ApiError(f"{what}检查点损坏：时间段不是有限数", 409)
        out.append(rec)
    return out


def _dia_speakers(rows) -> list:
    """从分割结果里取说话人集合；畸形条目直接丢掉（手改文件不能让请求炸掉）。"""
    try:
        items = list(rows)
    except TypeError:
        return []
    return sorted({d["speaker"] for d in items
                   if isinstance(d, dict)
                   and isinstance(d.get("speaker"), str)})


def api_assign(payload: dict):
    """写入角色名（同名即合并）并立即重写字幕。

    读-改-写全程持该 work 目录的锁：标注状态是 load → set → save 三步，
    两个并发请求会让写线程撞上读线程（Windows 上 os.replace 报 WinError 5 → 500）。
    `names` 里不认识的键直接忽略 —— 前端可能拿着过期的聚类列表，
    但绝不能因此凭空造出一个角色。
    """
    raw_names = payload.get("names")
    if raw_names is not None and not isinstance(raw_names, dict):
        return {"error": "names 必须是对象"}, 400
    if payload.get("work_name") is not None and not isinstance(payload["work_name"], str):
        return {"error": "work_name 必须是字符串"}, 400
    try:
        ctx = _api_open(payload)
    except ApiError as exc:
        return {"error": exc.msg}, exc.code
    with _label_lock(ctx["work"]):
        _reload_under_lock(ctx)
        if payload.get("work_name"):
            speaker_labels.set_work_name(ctx["labels"], payload["work_name"])
        names = {k: v for k, v in (raw_names or {}).items()
                 if k in ctx["known"] and isinstance(v, str)}
        try:
            _enroll_named_clusters(ctx, names)
        except Exception as exc:
            # 入库是「顺手做的事」，失败也必须把名字存下来 —— 名字才是主路径
            _emit_quiet(f"声纹入库失败（名字照存）：{exc}")
        speaker_labels.set_names(ctx["labels"], names)
        speaker_labels.save_labels(ctx["work"], ctx["labels"])
        return _write_back(ctx)


def _enroll_named_clusters(ctx, names) -> list:
    """用户给聚类命名 = 确认：把它的质心入声纹库，并把匹配结果写进标注状态。

    只在**这次真的被命名/改名**的聚类上做（`names` 里值非空、且与该聚类已存的
    名字不同）。理由：
    * 值没变 = 用户只是改了别的聚类，重复入库会把 70/30 滚动质心往同一批向量上
      反复加权（越滚越偏），还会把 `n_enroll` 刷成假数字；
    * 空值 = 清掉名字，不是确认。

    没有作品名就**不入库**：库按 `作品/角色` 索引，没有作品名时凭空造一个条目
    只会污染库（面板上会出现一个叫「未命名」的作品）。名字照存，行为与改造前一致。

    没有质心（这次分割没拿到 / 形状不可用）也照样登记角色：`enroll` 对
    `vector=None` 的处理是「只登记、不写向量」，该角色仍会出现在库里、可被改名。
    返回被入库的聚类键列表。
    """
    labels = ctx["labels"]
    clusters = labels.setdefault("clusters", {})
    # 先把记录**建出来**：`_api_open` 只认「状态表里已有的 ∪ 分割结果里的」聚类，
    # 并不落盘（落盘的是 api_speakers 那条只读路径）。不先 ensure 的话这里
    # `clusters.get(cluster)` 是 None，`embedding` 读不到、`match` 也没地方写
    # ——表现就是「命名了但库里没向量」。
    speaker_labels.ensure_clusters(labels, list(names))
    work = speaker_labels.get_work_name(labels)
    # 维度按 `_expected_embedding_dim()`（恒 256）取，用于拒掉「手改出来的短向量」
    # ——不能让它把库级 dim 改掉。库里那份 dim 只可能被手改/历史污染改坏，
    # 恒收 256 维正是让它自愈的那条路（详见该函数的 docstring）。
    expected_dim = _expected_embedding_dim() if work else 0
    embeds = {}
    dropped = []
    for cluster in names:
        rec = clusters.get(cluster)
        if not isinstance(rec, dict) or rec.get("embedding") is None:
            continue
        vec = _check_embedding(rec["embedding"], expected_dim)
        if vec is None:
            dropped.append(cluster)
            continue
        embeds[cluster] = vec
        # 顺手把状态里那份换成校验过的单位向量：面板读的就是它，
        # 留着原始坏值会让面板显示一个与库不一致的「声纹」。
        rec["embedding"] = [float(x) for x in vec]
    if dropped:
        _emit_quiet("聚类质心不可用（只登记角色、不写向量）：%s"
                    % "、".join(sorted(dropped)))
    if work:
        # 兜底：状态表里没有 `embedding` 或它不可用（面板比识别流程先打开、
        # 或 speakers.json 被手改过），但 sidecar 还在 —— 直接读它。
        # 声纹是锦上添花，但「明明拿到了质心却不入库」是实实在在的损失。
        missing = [c for c in names if c not in embeds]
        if missing:
            fresh = load_cluster_embeddings(ctx["work"])
            for cluster in missing:
                vec = _check_embedding(fresh.get(cluster), expected_dim)
                if vec is None:
                    continue
                rec = clusters.get(cluster)
                if isinstance(rec, dict):
                    rec["embedding"] = [float(x) for x in vec]
                embeds[cluster] = vec
    # 嵌入模型身份**从这次真正用过的分割模型推导**（D7）：写死常量时
    # 「换模型后旧向量失效」永远触发不了，而同维不同模型会被静默混比。
    model_tag = _work_embedding_tag(ctx["work"]) if work else ""
    # 匹配要在**入库之前**算：入库之后这个聚类自己就成了库里的角色，
    # 它会以 1.0 的分数匹配到自己，`match` 就变成一句废话了。
    # `model=` 传这次真正用的那个身份：库里按空间分开存，拿库级模型去比会
    # 在「换回旧模型」时算错空间（D7 的另一半）。
    matches = {}
    if embeds and work:
        try:
            matches = speaker_registry.match_vectors(
                work=work, clusters={c: {"embedding": v}
                                     for c, v in embeds.items()},
                model=model_tag)
        except Exception as exc:
            _emit_quiet(f"声纹匹配失败（只入库不预填）：{exc}")
            matches = {}

    job_id = str(labels.get("job") or "")
    enrolled = []
    for cluster, page in names.items():
        page = (page or "").strip()
        if not page or speaker_labels.cluster_name(labels, cluster) == page:
            continue
        info = matches.get(cluster) or {}
        vec = embeds.get(cluster)
        if work:
            out = speaker_registry.enroll(
                work=work, page=page,
                vector=np.asarray(vec, dtype=np.float64) if vec is not None else None,
                job=job_id, cluster=cluster, model=model_tag,
                source="manual")
            rec = clusters.get(cluster)
            if isinstance(rec, dict):
                # **入库那一刻的快照**，名字叫 `match_at_enroll` 而不是 `match`：
                # `GET /api/speakers` 返回的那个 `match` 是**实时算的当前建议**
                # （多一个 `band`），两者同名不同形，混起来读会读错一份。
                # 这份是 Task 5 计划书第 4 条要求的审计留档：没有 band、
                # 多一个 `vector_stored`，**没有任何读取方**（面板只读 API 那份）。
                rec["match_at_enroll"] = {
                    "work": work,
                    "page": (out or {}).get("page") or page,
                    "score": None if not info.get("top") else round(
                        float(info["top"]["score"]), 6),
                    "auto": info.get("auto_page"),
                    "vector_stored": bool((out or {}).get("vector_stored")),
                }
        elif isinstance(clusters.get(cluster), dict):
            # 没有作品名：不写 match_at_enroll（那会让人以为已经入库了）
            clusters[cluster].pop("match_at_enroll", None)
        enrolled.append(cluster)
    return enrolled


def _expected_embedding_dim() -> int:
    """入库时按哪个维度校验向量：**恒** `speaker_registry.DEFAULT_DIM`（256）。

    为什么不去读库里的 `dim`（旧名字 `_library_dim(work)` 就是这么写的，而且它
    把 **work 目录**当库根传给了 `load_registry`，`<work>/speakers_registry.json`
    永远不存在 ⇒ 恒返回 256）：那不是「恰好对了」，而是**必须**恒为 256 ——

    * `load_registry` 对缺失/损坏/`dim:0` 一律强制 256，所以它永远不会返回 0，
      「库里还没定过维度时不拦」那段说明是**死分支**；
    * 真去读库（被手改或历史污染成 `dim=3` 的库）会把 256 维向量判成非法
      （`vector=None`）⇒ `enroll` 再也量不出维度 ⇒ **dim 永远是 3**，
      既不能匹配也不能再入库，库彻底卡死；
    * 现状相反：照收 256 维 → `enroll` 按向量重量维度 → 库级 dim **自愈**回 256
      （Task 5 D8 的真实兜底路径）。

    所以这条路径**不能焊死**：将来真要支持非 256 维的嵌入模型，必须和
    `speaker_registry._page_vectors` 的全局 dim 过滤、以及 `pages[P]["space"]`
    的空间判定一起重新设计，不能只把这里的常量换成读库。
    """
    return int(speaker_registry.DEFAULT_DIM)


def _check_embedding(value, expected_dim=0):
    """该不该拿这个向量去入库：能用就返回**单位**向量，否则 None（绝不抛）。

    `speakers.json` 是给人手改的，而 `enroll` 会**从向量量出维度并写进库级 dim**
    （`speaker_registry.enroll` 的 `if not dim and unit is not None`）——
    一个手改出来的 `[0.1, 0.2, 0.3]` 足以把库级维度改成 3，于是**全部** 256 维
    向量当场不再参与匹配（`_page_vectors` 按全局 dim 过滤），而没有任何报错。

    所以入库前必须校验：一维、可转 float、全有限、长度等于
    `_expected_embedding_dim()`（调用方传进来的就是它，**恒 256**；不是「库里
    还没定过维度就不拦」—— 那个分支不存在，见该函数的 docstring）。
    不合格就当「没有质心」处理：只登记角色、不写向量 —— 那正是 `enroll` 对
    `vector=None` 的行为，也是 `_enroll_named_clusters` docstring 承诺的行为。
    """
    if isinstance(value, (str, bytes, dict)):
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not bool(np.all(np.isfinite(arr))):
        return None
    if expected_dim and int(arr.shape[0]) != int(expected_dim):
        return None
    return speaker_registry.normalise_vector(arr)


def api_reassign(payload: dict):
    """把某一句改判到别的说话人（拆分纠错）。"""
    try:
        index = int(payload.get("index"))
    except (TypeError, ValueError):
        return {"error": "index 必须是整数"}, 400
    raw_to = payload.get("to")
    if raw_to is not None and not isinstance(raw_to, str):
        return {"error": "to 必须是字符串"}, 400
    if payload.get("work_name") is not None and not isinstance(payload["work_name"], str):
        return {"error": "work_name 必须是字符串"}, 400
    to = (raw_to or "").strip()
    if not to:
        return {"error": "缺少 to"}, 400
    try:
        ctx = _api_open(payload)
    except ApiError as exc:
        return {"error": exc.msg}, exc.code
    with _label_lock(ctx["work"]):
        _reload_under_lock(ctx)
        if not (0 <= index < len(ctx["segs"])):
            return {"error": "index 超出句子范围"}, 400
        if payload.get("work_name"):
            speaker_labels.set_work_name(ctx["labels"], payload["work_name"])
        if to.startswith("new:"):
            to = to[4:].strip() or "?"
            if to != "?":
                # 新角色：聚类键就是它的名字，这样阶段 2 导出时目录名也是对的
                speaker_labels.ensure_clusters(ctx["labels"], [to])
                speaker_labels.set_names(ctx["labels"], {to: to})
        elif to != "?" and to not in (ctx["labels"].get("clusters") or {}):
            speaker_labels.ensure_clusters(ctx["labels"], [to])
        speaker_labels.set_override(ctx["labels"], index, to)
        speaker_labels.save_labels(ctx["work"], ctx["labels"])
        obj, code = _write_back(ctx)
        if code == 200:
            obj["index"] = index
            obj["to"] = to
        return obj, code


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, s, ctype="text/plain; charset=utf-8"):
        body = s.encode("utf-8") if isinstance(s, str) else s
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _local_host(self):
        host = self.headers.get("Host", "").lower()
        port = self.server.server_address[1]
        return host if host in (f"127.0.0.1:{port}", f"localhost:{port}") else None

    def _local_action(self):
        host = self._local_host()
        origin = self.headers.get("Origin")
        referrer = self.headers.get("Referer")
        return (host is not None
                and self.headers.get("Sec-Fetch-Site") in (None, "same-origin")
                and (origin is None or origin == f"http://{host}")
                and (referrer is None or referrer.startswith(f"http://{host}/")))

    # ---- GET ----
    def do_GET(self):
        if self._local_host() is None:
            return self._json({"error": "请通过本机地址访问集声"}, 403)
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            if not UI_FILE.exists():
                return self._text("ui.html 缺失")
            return self._text(UI_FILE.read_text(encoding="utf-8"), "text/html; charset=utf-8")

        background_prefix = "/assets/backgrounds/"
        if u.path.startswith(background_prefix):
            name = u.path.removeprefix(background_prefix)
            if name not in BACKGROUND_FILES:
                return self.send_error(404)
            try:
                image = (BACKGROUND_DIR / name).read_bytes()
            except FileNotFoundError:
                return self.send_error(404)
            return self._text(image, "image/webp")

        if u.path == "/api/models":
            import stage_jobs
            import asr_languages
            import download_jobs
            qwen_ready, qwen_note = stage_jobs.qwen_status()
            download_ready, download_note = download_jobs.status()
            asr = []
            if A.resolve_model(None) == str(A.LOCAL_MODEL_DIR):
                asr.append({"value": "", "label": "anime-whisper（本地，已就绪）"})
            else:
                asr.append({"value": "", "label": "anime-whisper（在线仓库，未下载本地）"})
            asr.append({"value": "quantumcookie/anime-whisper-ct2-fp16",
                        "label": "anime-whisper CT2 FP16（在线）"})
            asr.append({"value": "quantumcookie/anime-whisper-ct2",
                        "label": "anime-whisper CT2 INT8（更省显存）"})
            asr.append({"value": "kotoba-tech/kotoba-whisper-v2.2",
                        "label": "kotoba-whisper v2.2（通用日语）"})
            return self._json({
                "asr": asr,
                "qwen_ready": qwen_ready,
                "qwen_note": qwen_note,
                "qwen_languages": asr_languages.options(),
                "download_ready": download_ready,
                "download_note": download_note,
                "background_presets": [Path(name).stem for name in sorted(BACKGROUND_FILES)
                                       if (BACKGROUND_DIR / name).is_file()],
                "separation": SEP_MODEL_CHOICES,
                "separation_ready": SEP_READY,
                "separation_note": SEP_NOTE,
                "diarize": DIARIZE_MODEL_CHOICES,
                "diarize_ready": DIARIZE_READY,
                "diarize_note": DIARIZE_NOTE,
                "default_output": str(OUTPUT_ROOT),
                "sep_output": str(SEP_OUT_ROOT),
                "hint": (f"GPU 可用 · 默认输出 {OUTPUT_ROOT}" if _cuda_ok()
                         else "⚠ CUDA 不可用，将使用 CPU（很慢）"),
            })

        if u.path == "/api/status" or u.path == "/api/jobs":
            if u.path == "/api/jobs":
                with JOBS_LOCK:
                    return self._json({"jobs": [_job_view(x) for x in JOBS.values()]})
            jid = (q.get("id") or [""])[0]
            with JOBS_LOCK:
                j = JOBS.get(jid)
            if not j:
                return self._json({"error": "no such job"}, 404)
            return self._json(_job_view(j))

        if u.path == "/api/speakers":
            obj, code = api_speakers((q.get("job") or [None])[0],
                                     (q.get("work") or [None])[0])
            return self._json(obj, code)

        if u.path == "/api/workdirs":
            obj, code = api_workdirs()
            return self._json(obj, code)

        if u.path == "/api/registry/health":
            obj, code = api_registry_health()
            return self._json(obj, code)

        if u.path == "/api/registry":
            obj, code = api_registry()
            return self._json(obj, code)

        if u.path == "/api/clip":
            p = api_clip_path((q.get("spk") or [""])[0], (q.get("n") or ["0"])[0],
                              (q.get("job") or [None])[0],
                              (q.get("work") or [None])[0])
            if p is None:
                return self._json({"error": "片段不存在"}, 404)
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "none")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        if u.path == "/api/audio":
            # 供 <audio> 标签试听分离结果；只允许读 output 目录下的文件
            raw = (q.get("path") or [""])[0]
            p = Path(raw)
            try:
                p.resolve().relative_to(OUTPUT_ROOT.resolve())
            except (ValueError, OSError):
                with JOBS_LOCK:
                    allowed = any(p.resolve() == Path(f['path']).resolve()
                                  for job in JOBS.values() for f in job.get('files', []))
                if not allowed:
                    return self._json({"error": "路径不允许"}, 403)
            if not p.is_file():
                return self._json({"error": "文件不存在"}, 404)
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            self.send_header("Accept-Ranges", "none")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        if u.path == "/api/openfolder":
            if not self._local_action():
                return self._json({"error": "请从本机工作台打开目录"}, 403)
            raw = (q.get("path") or [""])[0]
            p = Path(raw) if raw else OUTPUT_ROOT
            try:
                if raw and not p.exists():
                    p = OUTPUT_ROOT
                if not p.is_dir():
                    return self._json({"error": "请选择一个目录"}, 400)
                p = p.resolve()
                os.startfile(str(p), "explore")  # noqa: S606
                return self._json({"ok": True, "path": str(p)})
            except (AttributeError, OSError, ValueError) as exc:
                return self._json({"error": str(exc)}, 500)

        if u.path == "/api/resumable":
            return self._json({"items": list_resumable()})

        if u.path in ("/api/result", "/api/download"):
            jid = (q.get("id") or [""])[0]
            name = (q.get("name") or [""])[0]
            with JOBS_LOCK:
                j = JOBS.get(jid)
            if not j:
                return self._json({"error": "no such job"}, 404)
            target = next((f for f in j.get("files", []) if f["name"] == name), None)
            if not target or not Path(target["path"]).exists():
                return self._json({"error": "no such file"}, 404)
            p = Path(target["path"])
            ctype = ("application/octet-stream" if u.path == "/api/download"
                     else (mimetypes.guess_type(p.name)[0] or "text/plain"))
            from urllib.parse import quote
            disp = (f"attachment; filename=\"download{p.suffix}\"; filename*=UTF-8''{quote(p.name, safe='')}"
                    if u.path == "/api/download" else "inline")
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Disposition", disp)
            self.send_header("Content-Length", str(p.stat().st_size))
            self.end_headers()
            with p.open('rb') as source:
                shutil.copyfileobj(source, self.wfile, length=1024 * 1024)
            return

        self._json({"error": "not found", "path": u.path}, 404)

    # ---- POST ----
    def do_POST(self):
        if not self._local_action():
            self.close_connection = True
            return self._json({"error": "请从本机工作台提交操作"}, 403)
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == '/api/download':
            try:
                payload = json.loads(self._body().decode('utf-8') or '{}')
            except (ValueError, UnicodeError):
                return self._json({'error': '请求体不是合法 JSON'}, 400)
            import download_jobs
            obj, code = download_jobs.enqueue(sys.modules[__name__], payload)
            return self._json(obj, code)

        if u.path == '/api/choose-folder':
            try:
                payload = json.loads(self._body().decode('utf-8') or '{}')
                if not isinstance(payload, dict):
                    raise ValueError('请求体必须是对象')
                import folder_picker
                path = folder_picker.choose_folder(str(payload.get('initial') or ''))
                return self._json({'path': path, 'cancelled': not bool(path)})
            except (ValueError, TypeError) as exc:
                return self._json({'error': str(exc)}, 400)
            except Exception as exc:
                return self._json({'error': '无法打开目录选择器：' + str(exc)}, 500)

        if u.path == "/api/upload":
            name = Path((q.get("name") or ["upload.bin"])[0]).name
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json({"error": "文件长度无效"}, 400)
            if n <= 0:
                return self._json({"error": "空文件"}, 400)
            if name in ("", ".", "..") or ":" in name:
                return self._json({"error": "文件名无效"}, 400)
            folder = None
            try:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                folder = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
                dest = folder / name
                with tempfile.NamedTemporaryFile(mode="wb", dir=folder, delete=False) as fh:
                    temporary = Path(fh.name)
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            raise OSError("上传未完成")
                        fh.write(chunk)
                        left -= len(chunk)
                os.replace(temporary, dest)
            except Exception as exc:
                if folder is not None:
                    shutil.rmtree(folder, ignore_errors=True)
                return self._json({"error": f"写入失败：{exc}"}, 500)
            return self._json({"path": str(dest)})

        if u.path == "/api/stop":
            try:
                req = json.loads(self._body().decode("utf-8") or "{}")
            except Exception:
                return self._json({"error": "请求格式错误"}, 400)
            ids = req.get("ids") or ([req["id"]] if req.get("id") else [])
            n = 0
            with JOBS_LOCK:
                targets = [JOBS.get(jid) for jid in ids]
                for j in targets:
                    if j and j["status"] in ("wait", "run"):
                        j["stop"] = True
                        n += 1
                        if j["status"] == "wait":
                            j.update(status="stopped", stage="已停止")
            # 正在跑子进程（人声分离）的，直接结束它
            for j in targets:
                if not j:
                    continue
                p = j.get("_proc")
                if p is not None and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
            return self._json({"stopped": n})

        if u.path in ("/api/speakers/assign", "/api/segments/reassign"):
            try:
                payload = json.loads(self._body().decode("utf-8") or "{}")
            except Exception:
                return self._json({"error": "请求体不是合法 JSON"}, 400)
            if not isinstance(payload, dict):
                return self._json({"error": "请求体必须是对象"}, 400)
            fn = api_assign if u.path == "/api/speakers/assign" else api_reassign
            obj, code = fn(payload)
            return self._json(obj, code)

        if u.path in ("/api/registry/rename", "/api/registry/forget"):
            try:
                payload = json.loads(self._body().decode("utf-8") or "{}")
            except Exception:
                return self._json({"error": "请求体不是合法 JSON"}, 400)
            if not isinstance(payload, dict):
                return self._json({"error": "请求体必须是对象"}, 400)
            fn = (api_registry_rename if u.path == "/api/registry/rename"
                  else api_registry_forget)
            obj, code = fn(payload)
            return self._json(obj, code)

        if u.path == "/api/export":
            try:
                payload = json.loads(self._body().decode("utf-8") or "{}")
            except Exception:
                return self._json({"error": "请求体不是合法 JSON"}, 400)
            obj, code = api_export(payload)
            return self._json(obj, code)

        if u.path == "/api/abandon":
            try:
                req = json.loads(self._body().decode("utf-8") or "{}")
            except Exception:
                return self._json({"error": "请求格式错误"}, 400)
            ids = req.get("ids") or ([req["id"]] if req.get("id") else [])
            n = 0
            with JOBS_LOCK:
                targets = [JOBS.get(jid) for jid in ids]
            for j in targets:
                if not j:
                    continue
                j["abandon"] = True
                j["stop"] = True
                n += 1
                # 正在跑的（分离/分割/ffmpeg）先把子进程掐掉，
                # 否则它可能一直挂在那里，worker 也回不来
                p = j.get("_proc")
                if p is not None and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                # 没在跑的：请求线程直接删检查点；
                # 正在跑的交给 worker 删——worker 是单线程，
                # 等它收尾后才会开始下一个任务，不会和新任务抢目录。
                if j.get("status") != "run":
                    _discard_job(j)
            # 立刻从任务列表移除，界面上的卡片随即消失
            with JOBS_LOCK:
                for j in targets:
                    if j:
                        JOBS.pop(j["id"], None)
            return self._json({"abandoned": n})

        if u.path == "/api/run":
            try:
                req = json.loads(self._body().decode("utf-8") or "{}")
            except Exception:
                return self._json({"error": "请求格式错误"}, 400)

            try:
                import stage_jobs
                if not isinstance(req, dict):
                    raise ValueError('请求体必须是对象')
                stage_jobs.validate(req)
            except ValueError as exc:
                return self._json({'error': str(exc)}, 400)

            raw = (req.get("path") or "").strip()
            if not raw:
                return self._json({"error": "缺少输入路径"}, 400)
            p = Path(raw)
            if not p.exists():
                return self._json({"error": f"路径不存在：{raw}"}, 400)
            if not p.is_dir() and not p.is_file():
                return self._json({"error": f"不是文件或目录：{raw}"}, 400)

            out = (req.get("output_dir") or "").strip()
            if out:
                try:
                    Path(out).mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    return self._json({"error": f"输出目录不可用：{exc}"}, 400)

            if p.is_dir():
                files = sorted((x for x in p.iterdir()
                                if x.is_file() and x.suffix.lower() in A.MEDIA_EXT),
                               key=A.natural_key)
                if not files:
                    return self._json({"error": "目录里没有媒体文件"}, 400)
                ids = [_enqueue(f, req, True) for f in files]
                return self._json({"ids": ids, "count": len(ids)})

            return self._json({"id": _enqueue(p, req, False)})

        self._json({"error": "not found"}, 404)


def _enqueue(media: Path, req: dict, is_dir_input: bool) -> str:
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid, "media": str(media), "name": media.name,
        "is_dir_input": is_dir_input, "status": "wait", "stage": "抽取音频",
        "progress": 0.0, "eta": None, "duration": None, "processed_sec": None,
        "logs": [f"已入队：{media.name}"], "files": [], "error": "", "stop": False,
        "out_dir": (req.get("output_dir") or "").strip() or
                   str(OUTPUT_ROOT / media.parent.name if is_dir_input else OUTPUT_ROOT),
        "opts": {
            "backend": req.get('backend') or 'whisper',
            "language": req.get('language') or ('zh' if req.get('backend') == 'qwen' else 'ja'),
            "asr_model": req.get("asr_model") or None,
            "separate": bool(req.get("separate")),
            "sep_model": req.get("sep_model") or "",
            "vad": bool(req.get("vad", True)),
            "beam": req.get("beam") or 5,
            "glossary": req.get("glossary") or "",
            "diarize": bool(req.get("diarize")),
            "diarize_model": (req.get("diarize_model")
                              or A.DEFAULT_DIARIZE_MODEL),
            "num_speakers": req.get("num_speakers") or None,
            "diarize_min_overlap": req.get("diarize_min_overlap") or 0.3,
            # asr = 分离(可选) + 识别    separate = 只做分离并导出试听
            "mode": req.get("mode") or "asr",
        },
    }
    with JOBS_LOCK:
        JOBS[jid] = job
    TASK_QUEUE.put(job)
    return jid


def main():
    # 输出重定向到文件时按区域编码写，日文里 GBK 编不了的字符不能把服务打断
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    A.register_cuda_dlls()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # 两个探测都是「起子进程等它退出」的阻塞操作（实测分离 2.5 秒 + 分割 10.3 秒），
    # 串行跑等于白等两者之和，所以并发跑。
    _detect_environments()
    _cuda_ok()

    threading.Thread(target=worker_loop, daemon=True).start()

    pend = list_resumable()
    # Windows 上 SO_REUSEADDR 允许两个进程绑同一端口，会导致请求被旧实例接走，
    # 出现「改了代码却不生效」的假象。这里关掉，重复启动就直接报错。
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print(f"\n[X] 端口 {PORT} 已被占用，无法启动：{exc}")
        print("    多半是已经有一个工作台在运行。")
        print("    可以直接用浏览器打开 http://%s:%d ，" % (HOST, PORT))
        print("    或先关掉旧实例（任务管理器里结束 python.exe / python3.exe）。")
        return 1
    print("=" * 60)
    print("  集声")
    print("=" * 60)
    print(f"  界面地址 : http://{HOST}:{PORT}")
    print(f"  输出目录 : {OUTPUT_ROOT}")
    print(f"  检查点   : {WORK_DIR}")
    print(f"  ASR 模型 : {'本地 anime-whisper 已就绪' if A.resolve_model(None) == str(A.LOCAL_MODEL_DIR) else '将走在线下载'}")
    print(f"  人声分离 : {'可用 · ' + SEP_NOTE if SEP_READY else '未就绪 —— ' + SEP_NOTE}")
    print(f"  说话人分割: {'可用 · ' + DIARIZE_NOTE if DIARIZE_READY else '未就绪 —— ' + DIARIZE_NOTE}")
    print(f"  GPU      : {'可用' if _cuda_ok() else '不可用（将用 CPU）'}")
    if pend:
        print(f"  ⏸ 可继续 : {len(pend)} 个未完成任务（界面上点「继续上次进度」）")
    print("=" * 60)
    print("  按 Ctrl+C 退出")
    if "--no-browser" not in sys.argv:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    sys.exit(main())
