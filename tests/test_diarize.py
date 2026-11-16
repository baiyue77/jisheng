#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""说话人分割相关逻辑的单元测试。

运行：
    venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jisheng as A  # noqa: E402


class TestAssignSpeakers(unittest.TestCase):
    def test_fully_inside_one_speaker(self):
        asr = [{"start": 1.0, "end": 2.0, "text": "a"}]
        dia = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        out = A.assign_speakers(asr, dia, 0.3)
        self.assertEqual(out[0]["speaker"], "SPEAKER_00")

    def test_picks_largest_overlap(self):
        asr = [{"start": 1.0, "end": 3.0, "text": "a"}]
        dia = [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},   # 0.5s
            {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},   # 1.5s
        ]
        out = A.assign_speakers(asr, dia, 0.3)
        self.assertEqual(out[0]["speaker"], "SPEAKER_01")

    def test_below_min_overlap_marks_unknown(self):
        asr = [{"start": 1.0, "end": 2.0, "text": "a"}]
        dia = [{"start": 1.9, "end": 2.5, "speaker": "SPEAKER_00"}]  # 仅 0.1s
        out = A.assign_speakers(asr, dia, 0.3)
        self.assertEqual(out[0]["speaker"], "?")

    def test_no_overlap_marks_unknown(self):
        asr = [{"start": 10.0, "end": 11.0, "text": "a"}]
        dia = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        out = A.assign_speakers(asr, dia, 0.3)
        self.assertEqual(out[0]["speaker"], "?")

    def test_touching_boundaries_do_not_count(self):
        asr = [{"start": 1.0, "end": 2.0, "text": "a"}]
        dia = [{"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"}]
        out = A.assign_speakers(asr, dia, 0.3)
        self.assertEqual(out[0]["speaker"], "?")

    def test_empty_diarization(self):
        asr = [{"start": 1.0, "end": 2.0, "text": "a"}]
        out = A.assign_speakers(asr, [], 0.3)
        self.assertEqual(out[0]["speaker"], "?")

    def test_multiple_segments_mutated_in_place(self):
        asr = [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
        ]
        dia = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ]
        out = A.assign_speakers(asr, dia, 0.3)
        self.assertIs(out, asr)
        self.assertEqual([s["speaker"] for s in out],
                         ["SPEAKER_00", "SPEAKER_01"])


class TestSpeakerFormatting(unittest.TestCase):
    def test_label_numbering_starts_at_one(self):
        self.assertEqual(A.fmt_speaker_label("SPEAKER_00"), "[说话人1]")
        self.assertEqual(A.fmt_speaker_label("SPEAKER_01"), "[说话人2]")
        self.assertEqual(A.fmt_speaker_label("SPEAKER_11"), "[说话人12]")

    def test_unknown_label(self):
        self.assertEqual(A.fmt_speaker_label("?"), "[?]")
        self.assertEqual(A.fmt_speaker_label(None), "[?]")
        self.assertEqual(A.fmt_speaker_label(""), "[?]")

    def test_custom_label_passthrough(self):
        self.assertEqual(A.fmt_speaker_label("CUSTOM_A"), "[CUSTOM_A]")

    def test_prefix_adds_label(self):
        self.assertEqual(A.prefix_with_speaker("こんにちは", "SPEAKER_00"),
                         "[说话人1] こんにちは")

    def test_prefix_does_not_duplicate(self):
        text = "[说话人1] こんにちは"
        self.assertEqual(A.prefix_with_speaker(text, "SPEAKER_00"), text)

    def test_prefix_unknown_speaker_gets_question_mark(self):
        self.assertEqual(A.prefix_with_speaker("こんにちは", "?"), "[?] こんにちは")

    def test_prefix_without_diarization_is_plain(self):
        self.assertEqual(A.prefix_with_speaker("こんにちは", None), "こんにちは")
        self.assertEqual(A.prefix_with_speaker("こんにちは", ""), "こんにちは")

    def test_prefix_question_mark_not_duplicated(self):
        self.assertEqual(A.prefix_with_speaker("[?] こんにちは", "?"),
                         "[?] こんにちは")


class TestWriteSrt(unittest.TestCase):
    def test_speaker_prefix_in_srt(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "おはよう",
             "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "おはようございます",
             "speaker": "SPEAKER_01"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.ja.srt"
            n = A.write_srt(segs, p)
            body = p.read_text(encoding="utf-8")
        self.assertEqual(n, 2)
        self.assertIn("[说话人1] おはよう", body)
        self.assertIn("[说话人2] おはようございます", body)
        self.assertIn("00:00:00,000 --> 00:00:01,000", body)

    def test_srt_without_speaker_unchanged(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "テスト"}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.ja.srt"
            A.write_srt(segs, p)
            body = p.read_text(encoding="utf-8")
        self.assertIn("テスト", body)
        self.assertNotIn("[说话人", body)
        self.assertNotIn("[?]", body)

    def test_srt_unknown_speaker(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "テスト", "speaker": "?"}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.ja.srt"
            A.write_srt(segs, p)
            body = p.read_text(encoding="utf-8")
        self.assertIn("[?] テスト", body)


class TestWriteSrtAtomic(unittest.TestCase):
    """write_srt 必须原子：写失败不能把用户已有的 SRT 毁成半截。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "out.ja.srt"

    def tearDown(self):
        self._tmp.cleanup()

    def test_existing_file_is_replaced_completely(self):
        self.path.write_text("旧字幕\n", encoding="utf-8")
        A.write_srt([{"start": 0.0, "end": 1.0, "text": "新的"}], self.path)
        body = self.path.read_text(encoding="utf-8")
        self.assertNotIn("旧字幕", body)
        self.assertIn("新的", body)

    def test_failed_replace_keeps_the_old_file(self):
        self.path.write_text("旧字幕\n", encoding="utf-8")
        orig = A.os.replace
        A.os.replace = lambda *a, **k: (_ for _ in ()).throw(OSError("磁盘满"))
        try:
            with self.assertRaises(OSError):
                A.write_srt([{"start": 0.0, "end": 1.0, "text": "新的"}], self.path)
        finally:
            A.os.replace = orig
        self.assertEqual(self.path.read_text(encoding="utf-8"), "旧字幕\n")
        self.assertEqual([p.name for p in self.path.parent.iterdir()],
                         [self.path.name])

    def test_signature_and_return_value_unchanged(self):
        n = A.write_srt([{"start": 0.0, "end": 1.0, "text": "テスト"}], self.path)
        self.assertEqual(n, 1)
        self.assertEqual(self.path.read_text(encoding="utf-8"),
                         "1\n00:00:00,000 --> 00:00:01,000\nテスト\n")


class TestResolveDiarizeModel(unittest.TestCase):
    """本地 pyannote 模型优先于 HuggingFace 在线仓库。"""

    def setUp(self):
        self._orig = A.LOCAL_PYANNOTE_DIR
        self._tmp = tempfile.TemporaryDirectory()
        A.LOCAL_PYANNOTE_DIR = Path(self._tmp.name)

    def tearDown(self):
        A.LOCAL_PYANNOTE_DIR = self._orig
        self._tmp.cleanup()

    def _make_local(self):
        base = A.LOCAL_PYANNOTE_DIR
        cfg_dir = base / "speaker-diarization-3.1"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.yaml").write_text(
            "pipeline:\n  params:\n"
            "    embedding: pyannote/wespeaker-voxceleb-resnet34-LM\n"
            "    segmentation: pyannote/segmentation-3.0\n",
            encoding="utf-8")
        for sub in ("segmentation-3.0", "wespeaker-voxceleb-resnet34-LM"):
            (base / sub).mkdir(parents=True)
            (base / sub / "pytorch_model.bin").write_bytes(b"stub")

    def test_without_local_falls_back_to_hf_id(self):
        self.assertEqual(A.resolve_diarize_model(None), A.DEFAULT_DIARIZE_MODEL)

    def test_local_preferred_when_complete(self):
        self._make_local()
        ref = A.resolve_diarize_model(None)
        self.assertIn("local-pipeline", ref)
        self.assertTrue(ref.endswith("config.yaml"), ref)
        cfg = Path(ref).read_text(encoding="utf-8")
        self.assertIn("segmentation-3.0/pytorch_model.bin", cfg)
        self.assertIn("wespeaker-voxceleb-resnet34-LM/pytorch_model.bin", cfg)
        self.assertNotIn("pyannote/segmentation-3.0", cfg)
        self.assertNotIn("pyannote/wespeaker-voxceleb-resnet34-LM", cfg)

    def test_default_hf_id_also_prefers_local(self):
        self._make_local()
        self.assertEqual(A.resolve_diarize_model(A.DEFAULT_DIARIZE_MODEL),
                         A.resolve_diarize_model(None))

    def test_explicit_non_default_id_is_respected(self):
        self._make_local()
        self.assertEqual(A.resolve_diarize_model("my/custom-model"),
                         "my/custom-model")

    def test_incomplete_local_is_ignored(self):
        base = A.LOCAL_PYANNOTE_DIR
        (base / "speaker-diarization-3.1").mkdir(parents=True)
        (base / "speaker-diarization-3.1" / "config.yaml").write_text(
            "x", encoding="utf-8")
        self.assertEqual(A.resolve_diarize_model(None), A.DEFAULT_DIARIZE_MODEL)


class _FakeProc:
    """替身 Popen：run_diarization 现在用 Popen + 轮询，不能再 mock run。"""

    def __init__(self, returncode=0, stderr_text=""):
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO(stderr_text)
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.returncode


class TestRunDiarizationWiring(unittest.TestCase):
    """不起 pyannote，也要覆盖 run_diarization 真实执行路径。

    这类测试专门防「只用了但没 import 的名字」——例如 json：
    真实分割成功、却在读结果 JSON 时 NameError，然后静默退化成无标签。
    """

    def _patch(self, proc):
        orig = (A.subprocess.Popen, A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                A.resolve_diarize_model)
        A.DIARIZE_PYTHON = Path(sys.executable)
        A.DIARIZE_WORKER = Path(sys.executable)
        A.resolve_diarize_model = lambda m: A.DEFAULT_DIARIZE_MODEL
        A.subprocess.Popen = lambda *a, **k: proc
        return orig

    def _restore(self, orig):
        (A.subprocess.Popen, A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
         A.resolve_diarize_model) = orig

    def test_reads_worker_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                       {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}]
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"RIFF")
            # run_diarization 在子进程结束后读取 <wav>.diarize.json
            wav.with_suffix(".diarize.json").write_text(
                json.dumps(payload), encoding="utf-8")

            proc = _FakeProc(0)
            orig = self._patch(proc)
            try:
                args = types.SimpleNamespace(
                    diarize_model=None, device="cpu", num_speakers=None)
                got = A.run_diarization(wav, args)
            finally:
                self._restore(orig)

            self.assertEqual(got, payload)

    def test_worker_failure_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"RIFF")

            proc = _FakeProc(5, stderr_text="boom\n")
            orig = self._patch(proc)
            try:
                args = types.SimpleNamespace(
                    diarize_model=None, device="cpu", num_speakers=None)
                got = A.run_diarization(wav, args)
            finally:
                self._restore(orig)

            self.assertIsNone(got)


class TestRunDiarizationCancellation(unittest.TestCase):
    """分割子进程必须可取消、可超时。

    之前 run_diarization 用 subprocess.run 且没有超时：子进程一旦挂住，
    WebUI 的单 worker 线程就永久阻塞，点「停止」也没用——
    「只有停止却不能放弃、同一任务卡死进程」就是这么来的。
    """

    def _stub_worker(self, tmp: Path) -> Path:
        sleeper = tmp / "sleeper.py"
        sleeper.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
        return sleeper

    def test_should_stop_aborts_and_kills_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "a.wav"
            wav.write_bytes(b"RIFF")
            sleeper = self._stub_worker(tmp)

            orig = (A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                    A.resolve_diarize_model)
            seen = []
            A.DIARIZE_PYTHON = Path(sys.executable)
            A.DIARIZE_WORKER = sleeper
            A.resolve_diarize_model = lambda m: "stub"
            try:
                args = types.SimpleNamespace(
                    diarize_model=None, device="cpu", num_speakers=None)
                with self.assertRaises(RuntimeError) as cm:
                    A.run_diarization(
                        wav, args, out_json=tmp / "o.json", timeout=60,
                        on_proc=seen.append,
                        should_stop=lambda: True)
            finally:
                (A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                 A.resolve_diarize_model) = orig

            self.assertIn("__STOPPED__", str(cm.exception))
            # on_proc 应先拿到子进程句柄，结束后回调 None
            self.assertTrue(seen and seen[0] is not None)
            self.assertIsNone(seen[-1])
            # 子进程必须真的被回收，不能留在后台
            self.assertIsNotNone(seen[0].poll())

    def test_timeout_aborts_and_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "a.wav"
            wav.write_bytes(b"RIFF")
            sleeper = self._stub_worker(tmp)

            orig = (A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                    A.resolve_diarize_model)
            A.DIARIZE_PYTHON = Path(sys.executable)
            A.DIARIZE_WORKER = sleeper
            A.resolve_diarize_model = lambda m: "stub"
            try:
                args = types.SimpleNamespace(
                    diarize_model=None, device="cpu", num_speakers=None)
                import time as _t
                t0 = _t.time()
                got = A.run_diarization(wav, args, out_json=tmp / "o.json",
                                        timeout=1)
                elapsed = _t.time() - t0
            finally:
                (A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                 A.resolve_diarize_model) = orig

            self.assertIsNone(got)
            self.assertLess(elapsed, 20, "超时后应尽快返回，而不是等子进程")


if __name__ == "__main__":
    unittest.main(verbosity=2)
