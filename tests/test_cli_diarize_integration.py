#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""端到端集成测试：真实 ASR + 打桩的 diarization 结果。

覆盖链路：
    extract_audio -> run_diarization(打桩) -> VAD -> ASR -> assign_speakers
    -> write_srt(带 [说话人N] 前缀)

不覆盖 pyannote 本体（需要联网授权模型与 HF_TOKEN）。

默认跳过（耗时约 1-2 分钟），显式开启：
    set JISHENG_RUN_INTEGRATION=1
    venv\Scripts\python.exe -m unittest tests.test_cli_diarize_integration -v
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jisheng as A  # noqa: E402

MEDIA = HERE / "compare" / "片段_16m00s_19m00s.mp4"
MODEL_DIR = A.LOCAL_MODEL_DIR


def _enabled() -> bool:
    return os.environ.get("JISHENG_RUN_INTEGRATION") == "1"


def _model_ready() -> bool:
    return all((MODEL_DIR / f).exists() for f in A.REQUIRED_MODEL_FILES)


def _device() -> tuple:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _base_args(diarize: bool):
    return types.SimpleNamespace(
        diarize=diarize,
        diarize_model=None,
        num_speakers=None,
        diarize_min_overlap=0.3,
        timing="vad",
        beam=5,
        vad_min_silence=500,
        no_vad=False,
        condition_on_previous=False,
        device=_device()[0],
        model=None,
    )


def _fake_turns_from_chunks(media: Path) -> list:
    """按 VAD 段落奇偶交替分配说话人，保证归属确定可断言。"""
    with tempfile.TemporaryDirectory(prefix="jisheng_it_") as tmp:
        wav = Path(tmp) / "a.wav"
        A.extract_audio(media, wav)
        audio = A.read_wav_float32(wav)
        chunks = A.vad_chunks(audio, vad_min_silence=500)
    turns = []
    for i, ch in enumerate(chunks):
        turns.append({
            "start": ch["start"] / 16000.0,
            "end": ch["end"] / 16000.0,
            "speaker": "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
        })
    return turns


@unittest.skipUnless(_enabled(), "需要 JISHENG_RUN_INTEGRATION=1 才运行")
@unittest.skipUnless(MEDIA.exists(), f"缺少测试素材 {MEDIA}")
@unittest.skipUnless(_model_ready(), f"缺少本地模型 {MODEL_DIR}")
class TestDiarizeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 真实 CLI 在 main() 里会先注册 CUDA DLL；测试必须照做，
        # 否则 ctranslate2 会报 "cublas64_12.dll is not found"。
        A.register_cuda_dlls()
        from faster_whisper import WhisperModel
        device, compute = _device()
        cls.model = WhisperModel(str(MODEL_DIR), device=device,
                                 compute_type=compute)
        cls.turns = _fake_turns_from_chunks(MEDIA)

    def test_diarize_labels_appear_in_srt(self):
        original = A.run_diarization
        A.run_diarization = lambda wav, args, out_json=None: self.turns
        try:
            with tempfile.TemporaryDirectory(prefix="jisheng_out_") as tmp:
                out_dir = Path(tmp)
                stats = A.transcribe_one(self.model, MEDIA, out_dir,
                                         _base_args(True), [])
                srt = out_dir / f"{MEDIA.stem}.ja.srt"
                txt = out_dir / f"{MEDIA.stem}.ja.txt"
                self.assertEqual(stats["status"], "ok")
                self.assertGreater(stats["segments"], 0)
                body = srt.read_text(encoding="utf-8")
                txt_body = txt.read_text(encoding="utf-8")
        finally:
            A.run_diarization = original

        self.assertIn("[说话人1]", body)
        self.assertIn("[说话人2]", body)
        self.assertIn("[说话人", txt_body)

        # 每个台词行都应带标签（打桩结果完整覆盖所有 VAD 段）
        text_lines = [
            ln for ln in body.splitlines()
            if ln.strip() and not ln.strip().isdigit() and "-->" not in ln
        ]
        self.assertTrue(text_lines)
        for ln in text_lines:
            self.assertTrue(ln.startswith("[说话人"),
                            f"未加标签的台词行：{ln!r}")

    def test_without_diarize_output_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="jisheng_out_") as tmp:
            out_dir = Path(tmp)
            stats = A.transcribe_one(self.model, MEDIA, out_dir,
                                     _base_args(False), [])
            srt = out_dir / f"{MEDIA.stem}.ja.srt"
            self.assertEqual(stats["status"], "ok")
            body = srt.read_text(encoding="utf-8")
        self.assertNotIn("[说话人", body)
        self.assertNotIn("[?]", body)


@unittest.skipUnless(_enabled(), "需要 JISHENG_RUN_INTEGRATION=1 才运行")
@unittest.skipUnless(MEDIA.exists(), f"缺少测试素材 {MEDIA}")
@unittest.skipUnless(A.DIARIZE_PYTHON.exists(), f"缺少 {A.DIARIZE_PYTHON}")
@unittest.skipUnless(A.resolve_diarize_model(None) != A.DEFAULT_DIARIZE_MODEL,
                     "缺少本地 pyannote 模型（models\\pyannote）")
class TestRealDiarizationWorker(unittest.TestCase):
    """真实调用 pyannote 子进程（本地模型，不走网络）。

    覆盖 diarize_worker.py 的整条链路：torchaudio 兼容层、
    checkpoint 安全白名单、本地 config 加载、波形直喂。
    """

    def test_worker_returns_speaker_turns(self):
        with tempfile.TemporaryDirectory(prefix="jisheng_dia_") as tmp:
            wav = Path(tmp) / "a.wav"
            A.extract_audio(MEDIA, wav)
            out = Path(tmp) / "dia.json"
            args = types.SimpleNamespace(
                diarize_model=None, device="cuda", num_speakers=None)
            turns = A.run_diarization(wav, args, out_json=out)

        self.assertIsNotNone(turns, "真实分割返回 None，见上面的 stderr")
        self.assertGreater(len(turns), 0)
        for t in turns:
            self.assertLessEqual({"start", "end", "speaker"}, set(t))
            self.assertLess(t["start"], t["end"])
        speakers = {t["speaker"] for t in turns}
        self.assertGreaterEqual(len(speakers), 1)
        # 3 分钟动漫片段里明显不止一个人说话
        self.assertGreaterEqual(len(speakers), 2, f"只识别到 {speakers}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
