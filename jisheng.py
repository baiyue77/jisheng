#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
jisheng.py —— 集声日语识别命令行

单集用，整季也用。使用 litagin/anime-whisper（5300 小时动漫语音微调）
的 CTranslate2 量化版，本地 GPU 加速。

用法
----
    # 环境自检
    jisheng.cmd --check

    # 单集
    jisheng.cmd "D:/番剧/S01/EP01.mkv"

    # 整季（直接传目录即可）
    jisheng.cmd "D:/番剧/S01"

    # 整季 + 术语表统一专有名词
    jisheng.cmd "D:/番剧/S01" --glossary "D:/番剧/术语表.txt"

    # 区分说话人（每行字幕前加 [说话人N]）
    jisheng.cmd "D:/番剧/S01/EP01.mkv" --diarize

    # 已知人数时指定，准确率更高
    jisheng.cmd "D:/番剧/S01/EP01.mkv" --diarize --num-speakers 2

    # 强制重跑（忽略已有结果）
    jisheng.cmd "D:/番剧/S01" --force

输出
----
    默认写到项目的 output 目录：

        output\<集名>.ja.srt   —— 可直接挂载的字幕
        output\<集名>.ja.txt   —— 纯文本

    整季输入目录时，会先建一个同名子目录，避免不同季互相覆盖：

        output\<季名>\<集名>.ja.srt

    用 -o/--output-dir 可以改到别处。

模型缓存
----
    模型名：quantumcookie/anime-whisper-ct2-fp16
            （litagin/anime-whisper 的 CTranslate2 FP16 版，约 1.4 GB）

    下载方式：

        jisheng.cmd --download
            从 Hugging Face 下载公开模型到本地模型目录。
        jisheng.cmd --download --endpoint <可信公开模型镜像>
            使用指定端点匿名下载。

    手动下载（命令行一直失败时最可靠）：

        1. 浏览器打开
           https://huggingface.co/quantumcookie/anime-whisper-ct2-fp16/tree/main
        2. 下载页面里全部 12 个文件
           （model.bin 必须下，约 1.4 GB；其余 11 个都是小文件）
        3. 全部放进项目的 models\anime-whisper-ct2-fp16\
        4. 直接运行识别，脚本会自动发现并使用这个目录

    脚本以此判断本地模型是否完整（缺一个就回退到在线下载）：

        model.bin  config.json  tokenizer.json
        vocabulary.json  preprocessor_config.json

    缓存位置由 HF_HOME 控制，默认为项目的 models 目录；
    若外部已设置 HF_HOME，脚本会尊重外部设置。

说话人分割（--diarize）
----
    anime-whisper 只负责「听写」，本身不区分谁在说话。加 --diarize 后，
    脚本会额外调用 pyannote.audio 做说话人分割（speaker diarization），
    再按时间重叠给每行字幕加上 [说话人N] 标签：

        1
        00:00:06,584 --> 00:00:08,120
        [说话人1] 大丈夫ですか？

    首次使用：

        1. 运行 setup_diarize.cmd 创建 venv-diarize 环境（pyannote + torch）。

        2. 模型来源按这个顺序决定：
             a) --diarize-model 传了非默认值 -> 用它；
             b) models\pyannote\ 下放了手动下载的模型 -> 用本地（推荐，免 token）；
             c) 都没有 -> 走 HuggingFace，需要先接受
                pyannote/speaker-diarization-3.1 与 pyannote/segmentation-3.0
                的用户协议，并 set HF_TOKEN=hf_xxxxxxxx

           国内 huggingface.co 常不可达，此时用 (b)：三个仓库（都约 31 MB）
           可从 ModelScope 取回，放到
               models\pyannote\speaker-diarization-3.1\config.yaml
               models\pyannote\segmentation-3.0\{config.yaml,pytorch_model.bin}
               models\pyannote\wespeaker-voxceleb-resnet34-LM\{config.yaml,pytorch_model.bin}
           脚本会据此生成 config，把子模型指向本地绝对路径。详见
           「说话人分割说明.md」。

    可用参数：

        --diarize                 启用说话人区分
        --diarize-model MODEL     模型 ID 或本地路径
        --num-speakers N          强制人数；留空自动估计
        --diarize-min-overlap SEC 与分割片段的最小重叠秒数，低于则标 [?]（默认 0.3）

    重要限制：

        * 标签是按「声音」聚类出来的，不是角色名。同一声优配多个角色时，
          会被归为同一个说话人；不同声优配同一角色时可能被拆成两人。
        * 动漫配音音色夸张、BGM 重时准确率会下降，建议先做 --glossary 无关的
          人声分离（WebUI 里勾选「先做人声分离」）再分割。
        * venv-diarize 缺失、模型未授权或显存不足时，脚本会打印警告并
          自动退化为「无标签输出」，不会中断识别任务。

整季设计
----
    * 自然排序：EP01 / EP02 / EP10 正确排序，不会把 EP10 排到 EP02 前面
    * 断点续跑：已生成 .ja.srt 的剧集自动跳过，中断后直接重跑即可
    * 术语表：把「听错的写法 -> 正确写法」写成文本，全季统一替换
    * 末尾汇总：成功 / 跳过 / 失败数量与总耗时

术语表格式（每行一条，# 开头为注释，四种分隔符都支持）
----
    第一文研究所 -> 第一研究所
    样本 甲 => 样本甲

环境
----
    项目目录：

        jisheng.cmd        <- 启动器，把视频或文件夹拖到它上面即可
        jisheng.py   <- 本脚本
        models\        <- 模型缓存（HF_HOME）
        output\        <- 字幕输出

    按 README.md 创建 venv。CPU 识别用 --device cpu；已有兼容的 CUDA
    运行库时，可通过 JISHENG_CUDA_LIB 指定 DLL 所在目录。

    首次运行会自动从 Hugging Face 下载模型（约 1.5 GB）到 models\。

关键约束
----
    * 官方明确要求 anime-whisper 必须「不带 initial_prompt」运行，
      任何提示词都会导致明显劣化和内容幻觉。本脚本不提供该选项。
      想统一专有名词，请用 --glossary 做后处理替换。
    * 默认开启 VAD，剪掉 BGM / 静默段，可显著降低幻觉。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def make_streams_robust():
    """让 stdout/stderr 在编码不了时替换字符，而不是抛异常。

    日文里有些字符（例如片假名中点 '・' U+30FB）用 GBK 编不出来。
    把输出重定向到文件时 Python 会按区域编码写，一旦遇到这些字符就
    UnicodeEncodeError，会把整集识别任务打断——纯粹是输出编码问题，
    不该毁掉已经跑完的识别。控制台（chcp 65001）下不受影响。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


# 本模块会往标准输出打日文（识别结果、字幕预览），所以导入时就放宽编码，
# 这样无论从 CLI、WebUI 还是被直接调用，都不会因为一个字符崩掉整个任务。
make_streams_robust()

# 默认模型：anime-whisper 的 CTranslate2 FP16 量化版
DEFAULT_MODEL = "quantumcookie/anime-whisper-ct2-fp16"

# 模型缓存和字幕输出相对于脚本所在目录。
DEFAULT_HOME = Path(__file__).resolve().parent
MODEL_CACHE_DIR = DEFAULT_HOME / "models"
DEFAULT_OUTPUT_ROOT = DEFAULT_HOME / "output"

# 说话人分割子进程环境（与 venv-sep 对齐，避免污染 ASR 环境）
DIARIZE_ENV = DEFAULT_HOME / "venv-diarize"
DIARIZE_PYTHON = DIARIZE_ENV / "Scripts" / "python.exe"
DIARIZE_WORKER = DEFAULT_HOME / "diarize_worker.py"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"

# 手动下载的模型放这里，脚本会自动识别并优先使用它
LOCAL_MODEL_DIR = MODEL_CACHE_DIR / "anime-whisper-ct2-fp16"

# 手动下载的 pyannote 说话人分割模型放这里（国内可从 ModelScope 获取）
LOCAL_PYANNOTE_DIR = MODEL_CACHE_DIR / "pyannote"

# CTranslate2 模型必需的几个文件（手动下载时对照检查）
REQUIRED_MODEL_FILES = (
    "model.bin", "config.json", "tokenizer.json",
    "vocabulary.json", "preprocessor_config.json",
)

MEDIA_EXT = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".ts", ".m2ts", ".wmv",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}


def _configure_model_cache():
    """把 HuggingFace 模型缓存指到项目的 models 目录。

    必须在 import huggingface_hub / faster_whisper 之前执行。
    若外部已显式设置 HF_HOME，则尊重外部设置，不覆盖。
    """
    existing = os.environ.get("HF_HOME")
    if existing:
        return Path(existing)
    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[!] 无法创建模型缓存目录 {MODEL_CACHE_DIR}：{exc}", file=sys.stderr)
        return None
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
    return MODEL_CACHE_DIR


CACHE_DIR = _configure_model_cache()


def resolve_model(model_arg):
    """决定用哪个模型：显式指定 > 本地手动下载的目录 > HuggingFace 在线仓库。"""
    if model_arg:
        return model_arg
    if all((LOCAL_MODEL_DIR / f).exists() for f in REQUIRED_MODEL_FILES):
        return str(LOCAL_MODEL_DIR)
    return DEFAULT_MODEL


def download_model(endpoint=None) -> int:
    """把模型下载到项目的 models/anime-whisper-ct2-fp16 目录。"""
    source = endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    from huggingface_hub import snapshot_download

    print(f"仓库     : {DEFAULT_MODEL}")
    print(f"下载源   : {source}")
    print(f"下载到   : {LOCAL_MODEL_DIR}")
    print("大小约 1.4 GB（只有 model.bin 是大文件）")
    print()

    try:
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=DEFAULT_MODEL,
                          local_dir=str(LOCAL_MODEL_DIR),
                          endpoint=source, token=False,
                          max_workers=4)
    except Exception as exc:
        print(f"\n[X] 下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("\n改用浏览器手动下载：", file=sys.stderr)
        print(f"  1. 打开 https://huggingface.co/{DEFAULT_MODEL}/tree/main",
              file=sys.stderr)
        print("  2. 把页面里的文件全部下载（model.bin 必须下）", file=sys.stderr)
        print(f"  3. 放进 {LOCAL_MODEL_DIR}", file=sys.stderr)
        print("  4. 重新运行识别，脚本会自动使用这个目录", file=sys.stderr)
        return 4

    missing = [f for f in REQUIRED_MODEL_FILES
               if not (LOCAL_MODEL_DIR / f).exists()]
    if missing:
        print(f"\n[!] 缺少文件：{', '.join(missing)}", file=sys.stderr)
        print("    请补下这些文件后再运行识别。", file=sys.stderr)
        return 4

    print("\n下载完成。之后直接运行识别即可，脚本会自动使用这个本地目录。")
    return 0


# --------------------------------------------------------------------------
# 运行环境
# --------------------------------------------------------------------------

# 必须持有 add_dll_directory 返回的句柄，否则目录注册会被回收掉
_DLL_DIR_HANDLES = []


def _candidate_lib_dirs() -> list:
    """列出所有可能藏着 cuBLAS / cuDNN 的目录。"""
    import glob

    dirs = []

    # 1) pip 安装的 nvidia CUDA 轮子
    for entry in sys.path:
        if entry and os.path.isdir(entry):
            dirs.extend(glob.glob(os.path.join(entry, "nvidia", "*", "bin")))

    # 2) ctranslate2 自带打包的 DLL（通常含 cudnn）
    for sub in ("Lib/site-packages/ctranslate2", "lib/site-packages/ctranslate2"):
        for root in (sys.prefix, sys.base_prefix):
            dirs.extend(glob.glob(os.path.join(root, sub)))

    # 3) 环境变量手动指定（多路径用分号分隔）
    extra = os.environ.get("JISHENG_CUDA_LIB") or os.environ.get("ANIME_ASR_CUDA_LIB", "")
    dirs.extend(p for p in extra.split(os.pathsep) if p)

    seen, unique = set(), []
    for d in dirs:
        d = os.path.normpath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique.append(d)
    return unique


def register_cuda_dlls(verbose: bool = False) -> list:
    """让 CTranslate2 在 Windows 上能找到 cuBLAS / cuDNN DLL。

    这里要同时做两件事，缺一不可：

    1) os.add_dll_directory() —— 让 Python 层的 ctypes 能找到。
       注意它返回的是句柄，必须一直持有；一旦被垃圾回收，
       目录就会从搜索路径里移除。

    2) 把目录塞进 PATH —— CTranslate2 的 CUDA 后端是 C++ 代码，
       用 LoadLibrary 按「默认搜索顺序」加载 DLL，这个顺序会读 PATH，
       但**不读** add_dll_directory 注册的目录。只做第 1 步的话，
       模型能加载、一推理就报 "Library cublas64_12.dll is not found"。
    """
    found = _candidate_lib_dirs()

    if hasattr(os, "add_dll_directory"):
        for directory in found:
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(directory))
            except OSError:
                pass

    if found:
        current = os.environ.get("PATH", "")
        parts = current.split(os.pathsep) if current else []
        prepend = [d for d in found if d not in parts]
        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend + parts)

    if verbose:
        for d in found:
            print(f"    + {d}")
    return found


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------

def fmt_ts(seconds: float) -> str:
    """秒 -> SRT 时间戳 00:00:00,000"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_hms(seconds: float) -> str:
    """秒 -> 人读的 1h23m45s"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def natural_key(path: Path):
    """自然排序：EP02 排在 EP10 前面。"""
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def extract_audio(src: Path, dst: Path) -> float:
    """用 ffmpeg 抽取 16kHz 单声道 WAV，返回时长（秒）。

    带 -nostdin：ffmpeg 默认会去读标准输入，被当成后台进程跑时可能
    因为等 stdin 而永远挂住（WebUI 里会连带卡死整个 worker）。
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽音频失败:\n{result.stderr.strip()[:800]}")

    duration = 0.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(dst)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            duration = 0.0
    return duration


def load_glossary(path: Path) -> list:
    """读取术语表，返回 [(错, 对), ...]，按关键词长度降序（避免短词先命中）。"""
    pairs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("\t", "->", "=>", "→", "="):
            if sep in line:
                old, new = line.split(sep, 1)
                old, new = old.strip(), new.strip()
                if old:
                    pairs.append((old, new))
                break
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def apply_glossary(text: str, pairs: list) -> str:
    for old, new in pairs:
        if old in text:
            text = text.replace(old, new)
    return text


def read_wav_float32(path: Path):
    """把 16kHz 单声道 WAV 读成 float32 数组。

    为什么不用 faster-whisper 自带的解码：它内部走 PyAV，
    而 faster-whisper 1.2.1 调用的 av.open(metadata_errors=...) 在 av 19 里已被移除，
    会直接报 "unexpected keyword argument"。我们这里已经用 ffmpeg 解好了音频，
    自己读进来交给模型即可，既绕开这个坑，也少一层不可控依赖。
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if channels != 1 or rate != 16000 or width != 2:
        raise RuntimeError(
            f"音频格式不符合预期：{channels}ch / {rate}Hz / {width*8}bit"
        )
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def audio_duration(path: Path) -> float:
    """用 ffprobe 读音频时长（秒）。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        try:
            return float(probe.stdout.strip())
        except ValueError:
            pass
    return 0.0


def _local_pyannote_pipeline() -> Path | None:
    """本地放了手动下载的 pyannote 模型时，生成一份指向本地权重的 config。

    目录布局（从 ModelScope 下载后直接放成这样）：

        models/pyannote/speaker-diarization-3.1/config.yaml
        models/pyannote/segmentation-3.0/{config.yaml,pytorch_model.bin}
        models/pyannote/wespeaker-voxceleb-resnet34-LM/{config.yaml,pytorch_model.bin}

    原始 config.yaml 里写的是 HuggingFace 仓库 ID，离线时会 401 / 超时。

    注意两个反直觉的点（都踩过）：
      * pyannote 的 ``from_pretrained`` 只判断 ``os.path.isfile``，
        所以必须传 **config.yaml 文件路径**，传目录会被当成 HF repo id；
      * 子模型的 ``pytorch_model.bin`` 本身就是 pytorch-lightning checkpoint
        （含 architecture 元信息），因此这里指向 **权重文件**，同样不能给目录。

    返回生成的 config.yaml 路径；本地模型不完整时返回 None。
    """
    base = LOCAL_PYANNOTE_DIR
    seg = base / "segmentation-3.0" / "pytorch_model.bin"
    emb = base / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin"
    src = base / "speaker-diarization-3.1" / "config.yaml"
    if not (src.is_file() and seg.is_file() and emb.is_file()):
        return None

    out_cfg = base / "local-pipeline" / "config.yaml"
    try:
        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        text = text.replace("pyannote/segmentation-3.0", seg.as_posix())
        text = text.replace("pyannote/wespeaker-voxceleb-resnet34-LM",
                            emb.as_posix())
        # 内容没变就别重写。这个函数在**只读**路径上也会被调用（面板探测、
        # 推导嵌入模型标识），每次都重写会让文件 mtime 一直变 —— 拿它做缓存
        # 失效判定的调用方（webui.embedding_model_tag）就永远命中不了缓存。
        try:
            if out_cfg.read_text(encoding="utf-8") == text:
                return out_cfg
        except OSError:
            pass
        out_cfg.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[!] 生成本地 pyannote config 失败：{exc}", file=sys.stderr)
        return None
    return out_cfg


def resolve_diarize_model(model_arg):
    """决定用哪个 diarization 模型。

    * 显式传了非默认值 -> 用它（强制走 HuggingFace 或自定义路径）
    * 否则本地有手动下载的 pyannote -> 用本地生成的 pipeline 目录
    * 都没有 -> 用 HuggingFace 上的默认仓库
    """
    if model_arg and model_arg != DEFAULT_DIARIZE_MODEL:
        return model_arg
    local = _local_pyannote_pipeline()
    if local is not None:
        return str(local)
    return model_arg or DEFAULT_DIARIZE_MODEL


def _drain(pipe, sink):
    """把子进程输出读进列表，避免管道缓冲区写满导致子进程卡死。"""
    try:
        for line in pipe:
            sink.append(line)
    except Exception:
        pass


def _terminate(proc, grace: float = 5.0):
    """先 terminate，宽限期后强杀。"""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass


def run_diarization(wav: Path, args, out_json: Path = None,
                    on_proc=None, should_stop=None,
                    timeout: float = 1800,
                    out_embeddings: Path = None) -> list | None:
    """调用独立子进程跑 pyannote 说话人分割，返回 [{start,end,speaker},...]。

    参数
    ----
    on_proc : 回调，拿到子进程句柄（结束后回调 None）。
              WebUI 靠它在「停止 / 放弃」时直接杀掉这个子进程。
    should_stop : 返回 True 时中止，抛 RuntimeError("__STOPPED__")。
                  没有它的话，子进程一旦挂住，调用方就永远回不来。
    timeout : 兜底超时（秒），0 表示不限。
    out_embeddings : 聚类质心 sidecar（npz）路径。**可选**：不传就不产生这个文件，
                     返回值与行为与改造前完全一致（老调用点/老测试不受影响）。
                     传了也只是让子进程多写一个文件，**返回值仍然是 list | None**
                     —— 调用方自己按这个路径去读质心（用
                     `speaker_registry.extract_centroids`）。
                     worker 那边拿不到质心时不会写这个文件，这里也不当失败。
                     三种退化的实际归宿（详见 `diarize_worker.py` 模块 docstring）：
                     `centroids is None`（OracleClustering）→ 不写、不打印
                     `EMBEDDINGS=`；行数 != 标签数（pyannote 3.4 下到不了，留作防御）
                     → 不写；部分行全零/非有限 → **照写**，由读端逐行丢弃。
    """
    if not DIARIZE_PYTHON.exists():
        print(f"[!] 说话人分割环境未安装：{DIARIZE_PYTHON}", file=sys.stderr)
        print("    请先运行 setup_diarize.cmd 创建 venv-diarize。", file=sys.stderr)
        return None
    if not DIARIZE_WORKER.exists():
        print(f"[!] 找不到 {DIARIZE_WORKER}", file=sys.stderr)
        return None

    out_json = wav.with_suffix(".diarize.json") if out_json is None else out_json
    model_ref = resolve_diarize_model(args.diarize_model)
    cmd = [
        str(DIARIZE_PYTHON), str(DIARIZE_WORKER),
        str(wav), str(out_json),
        "--model", model_ref,
        "--device", args.device,
    ]
    if args.num_speakers is not None:
        cmd += ["--num-speakers", str(args.num_speakers)]
    if out_embeddings is not None:
        cmd += ["--embeddings-out", str(out_embeddings)]

    local = LOCAL_PYANNOTE_DIR in Path(model_ref).parents
    print(f"        模型：{model_ref}"
          f"{'  [本地目录]' if local else '  [HuggingFace 在线仓库]'}")

    print("        调用 diarize_worker.py 进行说话人分割...")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if on_proc:
        on_proc(proc)

    out_lines, err_lines = [], []
    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_lines),
                             daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_lines),
                             daemon=True)
    t_out.start()
    t_err.start()

    started = time.time()
    try:
        while proc.poll() is None:
            if should_stop and should_stop():
                _terminate(proc)
                raise RuntimeError("__STOPPED__")
            if timeout and (time.time() - started) > timeout:
                _terminate(proc)
                print(f"[!] 说话人分割超时（>{int(timeout)}s），已中止。",
                      file=sys.stderr)
                return None
            time.sleep(0.5)
    finally:
        t_out.join(timeout=3)
        t_err.join(timeout=3)
        # 关掉管道句柄：长跑的服务器会放弃很多任务，泄漏句柄迟早出问题
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass
        if on_proc:
            on_proc(None)

    if proc.returncode != 0:
        tail = "".join(err_lines).strip().splitlines()[-5:]
        print("[!] 说话人分割失败（已跳过，输出将不带标签）：", file=sys.stderr)
        for line in tail:
            print(f"    {line}", file=sys.stderr)
        return None

    try:
        return json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[!] 读取 diarization 结果失败：{exc}", file=sys.stderr)
        return None


def assign_speakers(asr_segments: list, dia_segments: list, min_overlap: float) -> list:
    """按时间重叠把每个 ASR 片段归属到主要说话人。"""
    for seg in asr_segments:
        best, best_total = None, min_overlap
        counts = {}
        for d in dia_segments:
            ov = max(0.0, min(seg["end"], d["end"]) - max(seg["start"], d["start"]))
            if ov > 0:
                counts[d["speaker"]] = counts.get(d["speaker"], 0.0) + ov
        for speaker, total in counts.items():
            if total > best_total:
                best_total = total
                best = speaker
        seg["speaker"] = best if best else "?"
    return asr_segments


def fmt_speaker_label(speaker) -> str:
    """把 pyannote 的 SPEAKER_00 转成 [说话人1]，未知返回 [?]。"""
    if not speaker or speaker == "?":
        return "[?]"
    if isinstance(speaker, str) and speaker.upper().startswith("SPEAKER_"):
        try:
            idx = int(speaker.split("_", 1)[1]) + 1
            return f"[说话人{idx}]"
        except ValueError:
            pass
    return f"[{speaker}]"


def prefix_with_speaker(text: str, speaker) -> str:
    """在文本前加说话人标签。

    * speaker 为 None / 空  -> 原样返回（根本没启用 diarization）
    * speaker 为 "?"        -> 加 [?]（diarization 结果不可靠）
    * SPEAKER_xx            -> 加 [说话人N]
    """
    if not speaker:
        return text
    label = fmt_speaker_label(speaker)
    if text.startswith(label):
        return text
    return f"{label} {text}"


def vad_chunks(audio, vad_min_silence: int = 500):
    """用 Silero VAD 把音频切成语音段。

    这是时间轴的来源。anime-whisper 基于 kotoba-whisper v2.0，
    该系列没有训练可靠的 timestamp token：直接整段识别会得到
    「每 30 秒切一段、每段时长 0.0x 秒」的错乱时间轴
    （30 秒正是 Whisper 的滑窗长度）。改用 VAD 边界作为时间轴后，
    时间与文本无关，因此始终准确。
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=vad_min_silence, speech_pad_ms=200)
    return get_speech_timestamps(audio, opts, sampling_rate=16000)


def transcribe_chunk(model, audio, chunk, beam: int, glossary: list,
                     language: str = "ja", prompt: str = None):
    """识别单个 VAD 语音段，返回 {start, end, text} 或 None。"""
    piece = audio[int(chunk["start"]):int(chunk["end"])]
    if piece.size < 1600:                       # 短于 0.1 秒
        return None
    segs, _ = model.transcribe(
        piece,
        language=language,
        task="transcribe",
        beam_size=beam,
        initial_prompt=prompt,                  # anime-whisper 必须为 None；中文可传热词
        vad_filter=False,                       # 已经切好了
        condition_on_previous_text=False,
        without_timestamps=True,                # 时间轴用 VAD 的
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    text = apply_glossary("".join(s.text for s in segs).strip(), glossary)
    if not text:
        return None
    return {
        "start": chunk["start"] / 16000.0,
        "end": chunk["end"] / 16000.0,
        "text": text,
    }


def vad_chunked_transcribe(model, audio, args, glossary, on_progress=None):
    """切段 + 逐段识别（命令行用；WebUI 走分块检查点版本）。"""
    chunks = ([{"start": 0, "end": len(audio)}] if getattr(args, "no_vad", False)
              else vad_chunks(audio, args.vad_min_silence))
    total = len(chunks)
    results = []
    for i, ch in enumerate(chunks, 1):
        seg = transcribe_chunk(model, audio, ch, args.beam, glossary,
                               getattr(args, "language", "ja"),
                               getattr(args, "prompt", None))
        if seg:
            results.append(seg)
        if on_progress:
            on_progress(i / total if total else 1.0)
    return results


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """原子写文本：先写同目录临时文件，再 os.replace 到目标。

    字幕文件是用户的成品：直接 write_text 写到一半失败（磁盘满 / 文件被播放器
    或编辑器占用）会把用户原来那份好的 SRT 毁成半截。临时名带 pid 与线程 id，
    多线程并发也不会互相覆盖；replace 失败时把临时文件删掉，不留下垃圾。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_srt(segments, path: Path) -> int:
    """写出 SRT，返回实际写入的字幕条数。"""
    lines = []
    index = 0
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if "speaker" in seg:
            text = prefix_with_speaker(text, seg["speaker"])
        index += 1
        lines.append(str(index))
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        lines.append(text)
        lines.append("")
    atomic_write_text(path, "\n".join(lines))
    return index


# --------------------------------------------------------------------------
# 核心流程
# --------------------------------------------------------------------------

def transcribe_one(model, media: Path, out_dir: Path, args, glossary: list) -> dict:
    """处理单个文件，返回统计信息。"""
    started = time.time()
    stats = {"media": media.name, "duration": 0.0, "segments": 0,
             "elapsed": 0.0, "status": "ok", "note": ""}

    print(f"\n=== {media.name} ===")

    with tempfile.TemporaryDirectory(prefix="jisheng_") as tmp:
        wav = Path(tmp) / "audio16k.wav"
        print("  [1/3] 抽取 16kHz 单声道音频 ...")
        duration = extract_audio(media, wav)
        stats["duration"] = duration

        diarize_results = None
        if args.diarize:
            print("  [1.5/3] 说话人分割中（pyannote）...")
            diarize_results = run_diarization(wav, args)

        print("  [2/3] 识别中（anime-whisper）...")
        audio = read_wav_float32(wav)

        if args.timing == "vad":
            def _p(frac):
                if duration > 0:
                    print(f"        VAD 分段进度 {frac * 100:.0f}%", end="\r", flush=True)

            results = vad_chunked_transcribe(model, audio, args, glossary, _p)
            print(" " * 40, end="\r")
            for r in results:
                print(f"        [{fmt_ts(r['start'])}] {r['text']}", flush=True)
        else:
            segments, _info = model.transcribe(
                audio,
                language=args.language,
                task="transcribe",
                beam_size=args.beam,
                # anime-whisper 必须传 None，否则严重幻觉；中文模型可传热词
                initial_prompt=args.prompt,
                vad_filter=not args.no_vad,
                vad_parameters=dict(
                    min_silence_duration_ms=args.vad_min_silence,
                    speech_pad_ms=200,
                ),
                condition_on_previous_text=args.condition_on_previous,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=False,
            )
            results = []
            for seg in segments:
                text = apply_glossary(seg.text.strip(), glossary)
                if not text:
                    continue
                results.append({"start": seg.start, "end": seg.end, "text": text})
                pct = ""
                if duration > 0:
                    pct = f" ({min(seg.end / duration * 100, 100):.0f}%)"
                print(f"        [{fmt_ts(seg.start)}]{pct} {text}", flush=True)

    if not results:
        stats["status"] = "empty"
        stats["elapsed"] = time.time() - started
        print("  [!] 没有识别到任何语音内容，请检查音轨或改用 --no-vad 重试。")
        return stats

    if diarize_results:
        results = assign_speakers(results, diarize_results, args.diarize_min_overlap)
        for r in results:
            label = fmt_speaker_label(r.get("speaker"))
            print(f"        [{fmt_ts(r['start'])}] {label} {r['text']}", flush=True)

    print("  [3/3] 写出字幕 ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 注意：不能用 Path.with_suffix()，它会把 ".ja" 当后缀替换掉，
    # 结果变成 xxx.srt —— 那样断点续跑检查的 xxx.ja.srt 永远匹配不上。
    srt_path = out_dir / f"{media.stem}.ja.srt"
    txt_path = out_dir / f"{media.stem}.ja.txt"
    count = write_srt(results, srt_path)
    txt_path.write_text("\n".join(prefix_with_speaker(r["text"], r.get("speaker")) for r in results), encoding="utf-8")

    stats["segments"] = count
    stats["elapsed"] = time.time() - started
    speed = f"{duration / stats['elapsed']:.1f}x 实时" if duration and stats["elapsed"] else "n/a"
    print(f"  完成：{count} 条字幕，耗时 {fmt_hms(stats['elapsed'])}（{speed}）")
    print(f"        {srt_path}")
    return stats


def resolve_output_dir(target: Path, output_arg) -> Path:
    """决定字幕写到哪。

    * 显式给了 -o/--output-dir -> 用它
    * 输入是目录（整季）        -> 项目下的 output/<目录名>/，避免不同季互相覆盖
    * 输入是单个文件            -> 项目下的 output/
    """
    if output_arg:
        return Path(output_arg).expanduser()
    if target.is_dir():
        return DEFAULT_OUTPUT_ROOT / target.name
    return DEFAULT_OUTPUT_ROOT


def collect_inputs(target: Path, season: bool) -> list:
    if target.is_file():
        return [target]
    if target.is_dir() and season:
        files = sorted(
            (p for p in target.iterdir()
             if p.is_file() and p.suffix.lower() in MEDIA_EXT),
            key=natural_key,
        )
        if not files:
            raise SystemExit(f"目录里没有找到媒体文件：{target}")
        return files
    raise SystemExit(f"输入无效：{target}（目录请加 --season）")


def print_summary(all_stats: list, total_elapsed: float) -> None:
    ok = [s for s in all_stats if s["status"] == "ok"]
    empty = [s for s in all_stats if s["status"] == "empty"]
    failed = [s for s in all_stats if s["status"] == "failed"]
    skipped = [s for s in all_stats if s["status"] == "skipped"]

    total_audio = sum(s["duration"] for s in all_stats)
    total_segments = sum(s["segments"] for s in ok)

    print("\n" + "=" * 62)
    print("汇总")
    print("=" * 62)
    print(f"  识别成功 : {len(ok)} 集")
    if skipped:
        print(f"  已跳过   : {len(skipped)} 集（已有字幕，加 --force 可重跑）")
    if empty:
        print(f"  无语音   : {len(empty)} 集  -> " +
              ", ".join(s["media"] for s in empty))
    if failed:
        print(f"  失败     : {len(failed)} 集  -> " +
              ", ".join(s["media"] for s in failed))
        for s in failed:
            print(f"             {s['media']}: {s['note']}")
    if total_audio:
        print(f"  音频总时长: {fmt_hms(total_audio)}")
    if total_segments:
        print(f"  字幕总条数: {total_segments}")
    print(f"  总耗时   : {fmt_hms(total_elapsed)}")
    if total_audio and total_elapsed:
        print(f"  综合速度 : {total_audio / total_elapsed:.1f}x 实时")
    if len(ok) > 1:
        worst = max(ok, key=lambda s: s["elapsed"])
        print(f"  最慢一集 : {worst['media']}（{fmt_hms(worst['elapsed'])}）")
    print("=" * 62)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="动漫日语人声识别（anime-whisper）— 支持单集与整季",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="媒体文件，或加 --season 时传入目录")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（默认与输入文件同目录）")
    parser.add_argument("--season", "--batch", dest="season", action="store_true",
                        help="把 input 当作目录，批量处理整季（自然排序 + 断点续跑）")
    parser.add_argument("--glossary", default=None,
                        help="术语表文件，用于统一专有名词（后处理替换）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有字幕，强制重新识别")
    parser.add_argument("--model", default=None,
                        help="模型 ID 或本地目录；留空时自动选用本地已下载的模型")
    parser.add_argument("--language", default="ja",
                        help="识别语言，默认 ja；中文用 zh（anime-whisper 只支持 ja）")
    parser.add_argument("--prompt", default=None,
                        help="热词提示（仅中文等通用 Whisper 模型可用；"
                             "anime-whisper 必须留空，否则严重幻觉）")
    parser.add_argument("--download", action="store_true",
                        help="从 Hugging Face 匿名下载公开模型到本地后退出")
    parser.add_argument("--endpoint", default=None,
                        help="配合 --download 使用的可信公开模型镜像地址")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default=None,
                        help="留空时自动选（cuda->float16 / cpu->int8）")
    parser.add_argument("--beam", type=int, default=5, help="beam search 宽度")
    parser.add_argument("--no-vad", action="store_true",
                        help="关闭 VAD 过滤（BGM 重的音频不建议关）")
    parser.add_argument("--vad-min-silence", type=int, default=500,
                        help="VAD 判定静音的最短毫秒数")
    parser.add_argument("--condition-on-previous", action="store_true",
                        help="开启上下文依赖；动漫音频可能出现重复幻觉，默认关闭")
    parser.add_argument("--timing", default="vad", choices=["vad", "model"],
                        help="时间轴来源：vad=按语音段切分（推荐，时间轴准确）；"
                             "model=用模型自带时间戳（anime-whisper 不可靠）")
    parser.add_argument("--diarize", action="store_true",
                        help="启用说话人分割，在字幕前加 [说话人N] 标签（需先安装 venv-diarize）")
    parser.add_argument("--diarize-model", default=None,
                        help="说话人分割模型 ID 或本地路径（默认 pyannote/speaker-diarization-3.1）")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="强制说话人数（留空由模型自动估计）")
    parser.add_argument("--diarize-min-overlap", type=float, default=0.3,
                        help="ASR 片段与 diarization 片段的最小重叠秒数，低于此值标 [?]")
    parser.add_argument("--check", action="store_true",
                        help="只做环境自检（CUDA / ffmpeg / 依赖），不做识别")
    args = parser.parse_args()

    if args.download:
        return download_model(args.endpoint)

    dll_dirs = register_cuda_dlls(verbose=args.check)

    import ctranslate2

    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_count = 0

    media_tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    missing_tools = [name for name, path in media_tools.items() if not path]
    if args.check:
        writable = False
        try:
            DEFAULT_HOME.mkdir(parents=True, exist_ok=True)
            probe = DEFAULT_HOME / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            writable = True
        except OSError:
            writable = False

        cached = sorted(CACHE_DIR.rglob("models--*")) if CACHE_DIR and CACHE_DIR.exists() else []
        local_ok = all((LOCAL_MODEL_DIR / f).exists() for f in REQUIRED_MODEL_FILES)

        print("\n--- 环境自检 ---")
        print(f"Python        : {sys.version.split()[0]}")
        print(f"faster-whisper: {__import__('faster_whisper').__version__}")
        print(f"ctranslate2   : {ctranslate2.__version__}")
        print(f"CUDA 设备数   : {cuda_count}")
        for name, path in media_tools.items():
            print(f"{name:<14}: {path or '未找到'}")
        print(f"CUDA 库目录   : 找到 {len(dll_dirs)} 个")
        print(f"项目根目录    : {DEFAULT_HOME}  ({'可写' if writable else '不可写！'})")
        print(f"模型缓存      : {CACHE_DIR or '(默认位置)'}")
        print(f"本地模型      : {LOCAL_MODEL_DIR}")
        if local_ok:
            print("              -> 已就绪，识别时会直接用这个目录")
        else:
            missing = [f for f in REQUIRED_MODEL_FILES
                       if not (LOCAL_MODEL_DIR / f).exists()]
            print(f"              -> 未就绪（缺 {', '.join(missing)}）")
            print("                 运行 jisheng.cmd --download 下载，或参考 README 手动放置")
        print(f"在线缓存      : {len(cached)} 个 HuggingFace 缓存条目")
        print(f"输出目录      : {DEFAULT_OUTPUT_ROOT}")
        dia_ready = DIARIZE_PYTHON.exists() and DIARIZE_WORKER.exists()
        print(f"说话人分割    : {'venv-diarize 已就绪' if dia_ready else '未就绪（运行 setup_diarize.cmd 安装）'}")
        print("说话人标注    : 在 WebUI 里做（识别完成后点任务卡片上的「说话人标注」）")
        dia_model = resolve_diarize_model(None)
        dia_local = LOCAL_PYANNOTE_DIR in Path(dia_model).parents
        print(f"                模型：{dia_model}")
        if dia_local:
            print("                      -> 本地目录，不需要 HF_TOKEN")
        else:
            print("                      -> 需要 HuggingFace 授权 + HF_TOKEN")
        print("\n结论：" + ("可以 GPU 加速运行。" if cuda_count
                          else "CUDA 不可用，将退回 CPU（速度慢许多）。"))
        if not writable:
            print("[!] 请确认项目目录可写。")
        if missing_tools:
            print("[!] 请在 PATH 或 FFMPEG_DIR 中配置 FFmpeg 和 ffprobe。")
        return 2 if missing_tools or not writable else 0

    if not args.input:
        parser.error("请提供媒体文件路径，或使用 --check 做环境自检")

    if missing_tools:
        print(f"[X] 找不到 {'、'.join(missing_tools)}，请先安装 FFmpeg 并加入 PATH。", file=sys.stderr)
        return 2

    target = Path(args.input).expanduser()
    if not target.exists():
        print(f"[X] 路径不存在：{target}", file=sys.stderr)
        return 2

    # 传目录就自动按整季处理，省得记 --season 这个参数
    if target.is_dir() and not args.season:
        args.season = True
        print("检测到输入是目录，自动启用整季模式。")

    glossary = []
    if args.glossary:
        gpath = Path(args.glossary).expanduser()
        if not gpath.is_file():
            print(f"[X] 术语表不存在：{gpath}", file=sys.stderr)
            return 2
        glossary = load_glossary(gpath)
        print(f"术语表：{gpath.name} 载入 {len(glossary)} 条替换规则")

    inputs = collect_inputs(target, args.season)
    out_dir = resolve_output_dir(target, args.output_dir)

    print(f"模型缓存：{CACHE_DIR or '(使用默认位置)'}")
    print(f"输出目录：{out_dir}")
    print(f"待处理：{len(inputs)} 个文件")

    # 断点续跑：先算清楚哪些要跳过，避免白加载模型
    pending, skipped = [], []
    for media in inputs:
        dest = out_dir / f"{media.stem}.ja.srt"
        if dest.exists() and not args.force:
            skipped.append(media)
        else:
            pending.append(media)

    if skipped:
        print(f"已完成 {len(skipped)} 个，自动跳过（加 --force 可重跑）")

    all_stats = [
        {"media": m.name, "duration": 0.0, "segments": 0,
         "elapsed": 0.0, "status": "skipped", "note": ""}
        for m in skipped
    ]

    if not pending:
        print_summary(all_stats, 0.0)
        return 0

    from faster_whisper import WhisperModel

    if args.device == "cuda" and cuda_count == 0:
        print("[!] CUDA 不可用，自动退回 CPU 模式（速度会慢很多）。", file=sys.stderr)
        args.device = "cpu"

    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if args.device == "cuda" else "int8"

    model_ref = resolve_model(args.model)
    if model_ref == str(LOCAL_MODEL_DIR):
        print(f"加载模型：{model_ref}   [本地目录]")
    else:
        print(f"加载模型：{model_ref}   [HuggingFace 在线仓库]")
        print("  [提示] 若下载一直失败，改用：jisheng.cmd --download")
    print(f"设备：{args.device} / {compute_type}")
    try:
        model = WhisperModel(model_ref, device=args.device, compute_type=compute_type)
    except Exception as exc:
        print(f"\n[X] 模型加载失败：{exc}", file=sys.stderr)
        print("  下载公开模型：jisheng.cmd --download", file=sys.stderr)
        print(f"  手动：把模型文件放进 {LOCAL_MODEL_DIR}", file=sys.stderr)
        print("  显存不足时，可关闭占用 GPU 的应用或使用 --device cpu。", file=sys.stderr)
        return 3

    season_started = time.time()
    for i, media in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] 剩余 {len(pending) - i} 个")
        try:
            all_stats.append(transcribe_one(model, media, out_dir, args, glossary))
        except KeyboardInterrupt:
            print("\n[!] 收到中断，已完成的字幕均已落盘，重跑即可续上。",
                  file=sys.stderr)
            break
        except Exception as exc:
            print(f"  [X] 处理失败：{exc}", file=sys.stderr)
            all_stats.append({"media": media.name, "duration": 0.0, "segments": 0,
                              "elapsed": 0.0, "status": "failed",
                              "note": str(exc)[:120]})

    print_summary(all_stats, time.time() - season_started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
