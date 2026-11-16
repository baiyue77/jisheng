#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""pyannote.audio 3.x 与新版依赖的兼容处理。

pyannote.audio 3.x 假设自己跑在「torch 2.5 及更早 + 老版 torchaudio」上，
但在部分新版 PyTorch / torchaudio 组合上有两处不兼容：

1. torchaudio >= 2.9 移除了 I/O 后端的旧 API
   （``AudioMetaData`` / ``list_audio_backends`` / ``info``），
   而 pyannote 在导入期和 preprocessor 构造期就会引用它们。

2. torch >= 2.6 把 ``torch.load`` 的 ``weights_only`` 默认改成 True，
   pyannote 的 pytorch-lightning checkpoint 会被安全加载器拒绝。

两处都用「最小、精确」的方式绕开，而不是整体关掉安全校验：

* torchaudio：只补导入/构造期需要的两个符号；真正的音频解码由调用方
  用标准库完成，pyannote 走 ``file["waveform"]`` 分支，不会碰 torchaudio 的 I/O。
* weights_only：只放行 pickle 里实际引用的 4 个类
  （用 pickletools 扫描 ``data.pkl`` 确认过）。

由 diarize_worker.py 和 setup_diarize.py 的自检共用。
"""


def patch_torchaudio():
    """补齐 pyannote 需要、但新版 torchaudio 已移除的最小接口。"""
    try:
        import torchaudio
    except ImportError:
        return False

    if not hasattr(torchaudio, "AudioMetaData"):
        from collections import namedtuple
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels",
             "bits_per_sample", "encoding"],
        )
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    return True


def allow_checkpoint_globals():
    """把 pyannote checkpoint 里出现的非张量全局对象加入 torch 安全名单。

    扫描 pytorch_model.bin 的 pickle 后确认只有这 4 个（张量类默认已放行）：
        torch.torch_version.TorchVersion
        pyannote.audio.core.task.{Problem, Resolution, Specifications}
    """
    try:
        import torch
        from torch.torch_version import TorchVersion
        from pyannote.audio.core.task import (Problem, Resolution,
                                              Specifications)
    except Exception:
        return False

    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is not None:
        add([TorchVersion, Problem, Resolution, Specifications])
    return True


def apply_all():
    """按顺序应用全部兼容处理（torchaudio 必须在 import pyannote 之前）。"""
    ok = patch_torchaudio()
    allow_checkpoint_globals()
    return ok


if __name__ == "__main__":
    # 自检：能 import 成功就说明兼容层有效
    import importlib.metadata as md

    apply_all()
    import pyannote.audio  # noqa: F401
    import torch

    print("pyannote", md.version("pyannote.audio"))
    print("torch   ", torch.__version__, "cuda", torch.cuda.is_available())
