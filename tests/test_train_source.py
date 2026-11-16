#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Task 3：训练集专用的高质量分离（从原片全采样率重新分离）。

背景（Lead 已实测）：现有 `sep\` 人声轨是**从 16 kHz 单声道输入**分离出来的，
10~16 kHz 能量占比只有 **2.492e-07** —— 「44.1 kHz」只是上采样出来的壳，
内容带宽只有 8 kHz。拿它切训练集会得到发闷的数据。
本模块从**原片全采样率**（mono 44.1 kHz）重新分离，同一 BS-Roformer 权重，
实测 10~16 kHz 能量占比 **6.101e-03**（原片 5.748e-03）。
（`high_band_ratio` 是按 spec 近似的 10~16 kHz 占比实现，绝对值可与上表不完全一致，
判据用「> 1e-3」与「> 现有 16k 轨的 100 倍」这两个量级判据。）

单元用例全部 monkeypatch 掉子进程（不跑 GPU、不跑 ffmpeg）；
真实链路在 `TestHqSeparationIntegration`（`JISHENG_RUN_INTEGRATION=1` 门控，默认 skip）。

测试使用系统临时目录。
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import train_source as ts  # noqa: E402


TMP_ROOT = Path(tempfile.gettempdir())

MEDIA_SRC = HERE / "compare" / "片段_16m00s_19m00s.mp4"
#: 现有 16 kHz 输入分离出来的对照轨（只读引用，不修改）
REF_16K = (HERE / "work" / "片段_16m00s_19m00s" / "sep"
           / "raw16k_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")

VOCALS_NAME = "raw44k_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav"
INST_NAME = "raw44k_(Instrumental)_model_bs_roformer_ep_317_sdr_12.wav"


def _enabled() -> bool:
    return os.environ.get("JISHENG_RUN_INTEGRATION") == "1"


def write_wav(path, seconds=1.0, sr=44100, channels=1, freq=440.0, amp=0.5):
    """写一条真 WAV（验证要读真文件的路径：ffprobe / 频段能量）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(sr * seconds))
    t = np.arange(n, dtype=np.float64) / float(sr)
    pcm = (np.clip(amp * np.sin(2 * np.pi * freq * t), -1.0, 1.0) * 32767.0).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return path


class FakePopen:
    """`subprocess.Popen` 的替身。

    * `write_on_run`：**进程一起来就写出**这些真 WAV（模拟它边跑边落盘）。
      刻意不在 `poll()` 里写：`alive=True` 的替身永远不会被 poll 出结果，
      那样「跑到一半被打断」就永远看不到半成品，测不出清理逻辑。
    * `alive=True`：`poll()` 永远返回 None，直到被 `terminate()`/`kill()`
      —— 用来测「长步骤被 should_stop 打断」。
    * stdout/stderr 是文本流，按行可迭代。
    """

    def __init__(self, args, stdout_text="", stderr_text="", returncode=0,
                 write_on_run=(), alive=False, **kw):
        self.args = list(args)
        self.returncode = returncode
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self.terminated = False
        self.killed = False
        self._alive = alive
        self._write_on_run = list(write_on_run)
        if self._write_on_run:
            for p in self._write_on_run:
                write_wav(p, seconds=0.2)
            self._write_on_run = []

    def poll(self):
        if self.terminated or self.killed:
            self.returncode = -15
            return self.returncode
        if self._alive:
            return None
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.terminated or self.killed:
            self.returncode = -15
        return self.returncode

    def communicate(self, timeout=None):
        return self.stdout.read(), self.stderr.read()


def fake_popen(case, *, stdout_text="", stderr_text="", returncode=0,
               write_on_run=(), alive=False):
    """按用例收集 argv 的 Popen 工厂；`case.popen_args` 存最后一次 argv，
    `case.popen_count` 存次数。用 `mock.patch.object(ts.subprocess, "Popen", ...)`。"""
    def factory(args, **kw):
        case.popen_args.append(list(args))
        case.popen_count += 1
        return FakePopen(args, stdout_text=stdout_text, stderr_text=stderr_text,
                         returncode=returncode, write_on_run=write_on_run,
                         alive=alive, **kw)
    return factory


class BaseCase(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="t3_", dir=str(TMP_ROOT)))
        self.addCleanup(shutil.rmtree, str(self.dir), ignore_errors=True)
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.media = self.dir / "片段.mp4"
        self.media.write_bytes(b"fake-media" * 1000)
        self.popen_args = []
        self.popen_count = 0
        self.calls = []

    # -- 便捷构造 ---------------------------------------------------------
    def _state_path(self):
        return self.work / "sep_hq" / ".state.json"

    def _sep_hq_dir(self):
        d = self.work / "sep_hq"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _make_vocals_hq(self):
        return write_wav(self._sep_hq_dir() / VOCALS_NAME, seconds=0.2)

    def _make_instrumental_hq(self):
        return write_wav(self._sep_hq_dir() / INST_NAME, seconds=0.2)

    def _make_raw44k(self):
        return write_wav(self.work / "raw44k.wav", seconds=0.2)

    def _make_raw16k(self):
        return write_wav(self.work / "raw16k.wav", seconds=0.2)

    def _make_sep_16k(self):
        d = self.work / "sep"
        d.mkdir(parents=True, exist_ok=True)
        return write_wav(d / "raw16k_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav",
                         seconds=0.2)

    def _write_hq_state(self):
        ts.write_state(self.work, ts.state_key(self.media, ts.DEFAULT_MODEL))


class TestHelpers(BaseCase):

    def test_work_audio_accepts_str_or_path(self):
        """`work` 传 str 也必须得到 work 下的**绝对**路径。

        实测过：`work_audio` 不做 `Path(work)` 归一，str 会折进 `Path()` 之外，
        拼出来变成**相对当前工作目录**的路径 —— 于是「文件明明在 work 下，
        raw.exists() 却是 False」，静默走了重抽分支。
        """
        self._make_raw44k()
        for given in (self.work, str(self.work)):
            p = ts.work_audio(given, ts.RAW_HQ_NAME)
            self.assertIsInstance(p, Path)
            self.assertTrue(p.is_absolute(), "得到相对路径：%s" % p)
            self.assertEqual(p, self.work / "raw44k.wav")
            self.assertTrue(p.exists())

    def test_find_vocals_skips_instrumental_only_dir(self):
        """目录里只有伴奏（或杂文件）时不能把人声轨认成它。"""
        d = self._sep_hq_dir()
        write_wav(d / "raw44k_(Instrumental)_m.wav", seconds=0.1)
        self.assertIsNone(ts.find_vocals(d))

    def test_find_vocals_prefers_vocals_over_instrumental(self):
        d = self._sep_hq_dir()
        inst = write_wav(d / "raw44k_(Instrumental)_m.wav", seconds=0.1)
        voc = write_wav(d / "raw44k_(Vocals)_m.wav", seconds=0.1)
        self.assertEqual(ts.find_vocals(d), voc)
        self.assertEqual(ts.find_instrumental(d), [inst])

    def test_find_vocals_on_missing_dir_returns_none(self):
        self.assertIsNone(ts.find_vocals(self.work / "没有这个目录"))


# ---------------------------------------------------------------------------
# pick_source 的四条分支
# ---------------------------------------------------------------------------

class TestPickSource(BaseCase):

    def test_hq_mode_reuses_fresh_sep_hq_without_rerun(self):
        """复用命中：不重抽、不重分（用调用计数器断言）。"""
        self._make_vocals_hq()
        self._write_hq_state()
        with mock.patch.object(ts, "separate_hq",
                               side_effect=lambda *a, **k: self.calls.append("sep")) as m_sep, \
                mock.patch.object(ts, "extract_full_rate",
                                  side_effect=lambda *a, **k: self.calls.append("extract")):
            path, kind = ts.pick_source(self.work, self.media, mode="hq")
        self.assertEqual(kind, "separated_hq")
        self.assertEqual(path.name, VOCALS_NAME)
        self.assertEqual(self.calls, [])
        m_sep.assert_not_called()

    def test_hq_mode_skips_extract_when_raw44k_already_there(self):
        """`work\\raw44k.wav` 还在（上次分离中途失败留下的）就不重抽，直接分离。"""
        raw = self._make_raw44k()
        out = self._sep_hq_dir()

        def fake_sep(work, **kw):
            self.calls.append("sep")
            ts.write_state(work, ts.state_key(self.media, ts.DEFAULT_MODEL))
            return write_wav(out / VOCALS_NAME, seconds=0.2)

        with mock.patch.object(ts, "extract_full_rate",
                               side_effect=lambda *a, **k: (self.calls.append("extract"),
                                                            raw)[1]), \
                mock.patch.object(ts, "separate_hq", side_effect=fake_sep):
            path, kind = ts.pick_source(self.work, self.media, mode="hq")
        self.assertEqual(kind, "separated_hq")
        self.assertEqual(self.calls, ["sep"])
        self.assertTrue(path.exists())

    def test_hq_mode_extracts_when_raw44k_absent(self):
        """首跑：work 下什么都没有 -> 抽 44.1 k，再分离（调用顺序断言）。"""
        out = self._sep_hq_dir()

        def fake_extract(media, work, **kw):
            self.calls.append("extract")
            return self._make_raw44k()

        def fake_sep(work, **kw):
            self.calls.append("sep")
            ts.write_state(work, ts.state_key(self.media, ts.DEFAULT_MODEL))
            return write_wav(out / VOCALS_NAME, seconds=0.2)

        with mock.patch.object(ts, "extract_full_rate", side_effect=fake_extract), \
                mock.patch.object(ts, "separate_hq", side_effect=fake_sep):
            path, kind = ts.pick_source(self.work, self.media, mode="hq")
        self.assertEqual(kind, "separated_hq")
        self.assertEqual(self.calls, ["extract", "sep"])
        self.assertTrue(path.exists())

    def test_fast_mode_uses_existing_sep_stem(self):
        p = self._make_sep_16k()
        path, kind = ts.pick_source(self.work, self.media, mode="fast")
        self.assertEqual(kind, "separated_16k")
        self.assertEqual(path, p)

    def test_fast_mode_falls_back_to_raw16k_when_no_sep_dir(self):
        raw = self._make_raw16k()
        path, kind = ts.pick_source(self.work, self.media, mode="fast")
        self.assertEqual(kind, "raw16k")
        self.assertEqual(path, raw)

    def test_fast_mode_ignores_sep_hq(self):
        """快速模式不认 sep_hq（那是高质量源的产物）。"""
        self._make_vocals_hq()
        raw = self._make_raw16k()
        path, kind = ts.pick_source(self.work, self.media, mode="fast")
        self.assertEqual(kind, "raw16k")
        self.assertEqual(path, raw)

    def test_fast_mode_prefers_sep_over_raw16k_when_both_exist(self):
        """`sep\\` 与 `raw16k.wav` 同时存在时必须选 `sep\\`（分支顺序不能反）。"""
        legacy = self._make_sep_16k()
        self._make_raw16k()
        path, kind = ts.pick_source(self.work, self.media, mode="fast")
        self.assertEqual(kind, "separated_16k")
        self.assertEqual(path, legacy)

    def test_fast_mode_with_nothing_extracts_32k_from_media(self):
        def fake_extract(media, work, **kw):
            self.calls.append((media, work, kw))
            return write_wav(Path(work) / "raw32k.wav", seconds=0.2,
                             sr=kw.get("sr", 32000))

        with mock.patch.object(ts, "extract_full_rate", side_effect=fake_extract):
            path, kind = ts.pick_source(self.work, self.media, mode="fast")
        self.assertEqual(kind, "media")
        self.assertEqual(path.name, "raw32k.wav")
        self.assertEqual(self.calls[0][2]["sr"], 32000)
        self.assertEqual(self.calls[0][2]["out_name"], "raw32k.wav")
        self.assertEqual(self.calls[0][0], self.media)
        with wave.open(str(path), "rb") as fh:
            self.assertEqual(fh.getframerate(), 32000)
            self.assertEqual(fh.getnchannels(), 1)

    def test_hq_mode_with_both_absent_runs_full_chain(self):
        """hq 模式在什么都没有时：抽 44.1 k -> 分离。"""
        def fake_extract(media, work, **kw):
            self.calls.append("extract")
            return self._make_raw44k()

        def fake_sep(work, **kw):
            self.calls.append("sep")
            self._write_hq_state()
            return self._make_vocals_hq()

        with mock.patch.object(ts, "extract_full_rate", side_effect=fake_extract), \
                mock.patch.object(ts, "separate_hq", side_effect=fake_sep):
            path, kind = ts.pick_source(self.work, self.media, mode="hq")
        self.assertEqual(kind, "separated_hq")
        self.assertEqual(self.calls, ["extract", "sep"])

    def test_missing_media_raises(self):
        with self.assertRaises(ts.SourceError):
            ts.pick_source(self.work, self.dir / "没有这个文件.mp4", mode="hq")


# ---------------------------------------------------------------------------
# extract_full_rate
# ---------------------------------------------------------------------------

class TestExtractFullRate(BaseCase):

    def test_command_is_mono_44k_pcm16_with_nostdin(self):
        out = self.work / "raw44k.wav"
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, write_on_run=[out])):
            got = ts.extract_full_rate(self.media, self.work)

        self.assertEqual(got, out)
        self.assertTrue(out.exists())
        args = self.popen_args[-1]
        self.assertEqual(args[0], ts.FFMPEG)
        self.assertIn("-nostdin", args)
        self.assertIn("-vn", args)
        self.assertIn("-hide_banner", args)
        for flag, val in (("-ac", "1"), ("-ar", "44100"), ("-c:a", "pcm_s16le")):
            self.assertIn(flag, args)
            self.assertEqual(args[args.index(flag) + 1], val)
        self.assertTrue(args[-1].endswith("raw44k.wav"))

    def test_sr_is_configurable(self):
        out = self.work / "raw32k.wav"
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, write_on_run=[out])):
            got = ts.extract_full_rate(self.media, self.work, sr=32000,
                                       out_name="raw32k.wav")
        self.assertEqual(got, out)
        args = self.popen_args[-1]
        self.assertEqual(args[args.index("-ar") + 1], "32000")

    def test_nonzero_exit_raises_with_stderr(self):
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, returncode=1,
                                                         stderr_text="坏输入\n")):
            with self.assertRaises(ts.SourceError) as cm:
                ts.extract_full_rate(self.media, self.work)
        self.assertIn("坏输入", str(cm.exception))
        self.assertFalse((self.work / "raw44k.wav").exists())

    def test_missing_media_raises(self):
        with self.assertRaises(ts.SourceError):
            ts.extract_full_rate(self.dir / "nope.mp4", self.work)

    def test_zero_byte_output_is_treated_as_failure(self):
        out = self.work / "raw44k.wav"
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, write_on_run=[out])):
            ts.extract_full_rate(self.media, self.work)
            out.write_bytes(b"")          # 模拟磁盘写满导致的零字节产物
            with mock.patch.object(ts.subprocess, "Popen",
                                   side_effect=fake_popen(self)):
                with self.assertRaises(ts.SourceError):
                    ts.extract_full_rate(self.media, self.work)


# ---------------------------------------------------------------------------
# separate_hq
# ---------------------------------------------------------------------------

class StubbedSeparationCase(BaseCase):
    def setUp(self):
        super().setUp()
        worker_python = mock.patch.object(ts, "SEP_PYTHON", Path(sys.executable))
        worker_python.start()
        self.addCleanup(worker_python.stop)


class TestSeparateHq(StubbedSeparationCase):

    def test_worker_argv_matches_sep_worker_protocol(self):
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        stdout = "RESULT=%s\n" % vocals
        stdout += "".join("\r %3d%%|#####     | %d/100 [00:00<00:00, 1.0it/s]\n" % (i, i)
                          for i in range(0, 100, 10))
        stderr = "".join("\r %3d%%|#####     | %d/100 [00:00<00:00, 1.0it/s]\n" % (i, i)
                         for i in range(0, 100, 10))
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, stdout_text=stdout,
                                                         stderr_text=stderr,
                                                         write_on_run=[vocals])):
            vocal = ts.separate_hq(self.work, model="模型.ckpt")

        args = self.popen_args[-1]
        self.assertEqual(args[0], str(ts.SEP_PYTHON))
        self.assertEqual(args[1], str(ts.SEP_WORKER))
        self.assertEqual(args[2], str(self.work / "raw44k.wav"))
        self.assertEqual(args[3], str(self.work / "sep_hq"))
        self.assertEqual(args[4], "模型.ckpt")
        self.assertEqual(vocal, vocals)

    def test_writes_state_on_success_when_given_media(self):
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(
                                      self, stdout_text="RESULT=%s\n" % vocals,
                                      write_on_run=[vocals])):
            ts.separate_hq(self.work, model=ts.DEFAULT_MODEL, media=self.media)
        state = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertEqual(state["key"], ts.state_key(self.media, ts.DEFAULT_MODEL))
        self.assertEqual(state["model"], ts.DEFAULT_MODEL)
        self.assertTrue(state["vocals"].endswith(VOCALS_NAME))

    def test_state_key_argument_wins_over_media(self):
        """`pick_source` 自己算好键传进来（省一次 stat），要原样写下去。"""
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(
                                      self, stdout_text="RESULT=%s\n" % vocals,
                                      write_on_run=[vocals])):
            ts.separate_hq(self.work, media=self.media, key=["定好的", "键"])
        state = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertEqual(state["key"], ["定好的", "键"])

    def test_bare_call_writes_model_only_state(self):
        """没给 media 也没给 key：状态文件按模型名留一条，但仍然可用。"""
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(
                                      self, stdout_text="RESULT=%s\n" % vocals,
                                      write_on_run=[vocals])):
            ts.separate_hq(self.work, model=ts.DEFAULT_MODEL)
        state = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertEqual(state["key"], [ts.DEFAULT_MODEL])
        self.assertTrue(vocals.exists())

    def test_globs_vocals_when_model_name_truncated(self):
        """audio-separator 会把模型名截断 -> 不能拼死文件名，必须 glob `(Vocals)`。"""
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(
                                      self, stdout_text="没有 RESULT 行\n",
                                      write_on_run=[vocals])):
            got = ts.separate_hq(
                self.work, model="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        self.assertEqual(got, vocals)

    def test_nonzero_exit_raises_and_leaves_no_state(self):
        self._make_raw44k()
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, returncode=1,
                                                         stderr_text="模型权重缺失\n")):
            with self.assertRaises(ts.SourceError) as cm:
                ts.separate_hq(self.work)
        self.assertIn("模型权重缺失", str(cm.exception))
        self.assertFalse(self._state_path().exists())

    def test_exit_zero_without_result_raises(self):
        self._make_raw44k()
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, stdout_text="什么也没有\n")):
            with self.assertRaises(ts.SourceError) as cm:
                ts.separate_hq(self.work)
        self.assertIn("人声轨", str(cm.exception))
        self.assertFalse(self._state_path().exists())

    def test_result_path_that_does_not_exist_falls_back_to_glob(self):
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        bogus = self.work / "sep_hq" / "不存在的人声.wav"
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(
                                      self, stdout_text="RESULT=%s\n" % bogus,
                                      write_on_run=[vocals])):
            got = ts.separate_hq(self.work, model=ts.DEFAULT_MODEL)
        self.assertEqual(got, vocals)

    def test_tqdm_noise_is_filtered_from_error_message(self):
        self._make_raw44k()
        noise = "".join("\r %3d%%|#####     | %d/100 [00:00<00:00, 1.0it/s]\n" % (i, i)
                        for i in range(0, 100, 10))
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, returncode=1,
                                                         stderr_text=noise)):
            with self.assertRaises(ts.SourceError) as cm:
                ts.separate_hq(self.work)
        self.assertNotIn("%|", str(cm.exception))
        self.assertIn("1", str(cm.exception))

    def test_missing_raw44k_raises(self):
        with self.assertRaises(ts.SourceError):
            ts.separate_hq(self.work)
    def test_missing_sep_worker_raises(self):
        self._make_raw44k()
        with mock.patch.object(ts, "SEP_PYTHON", self.dir / "nope.exe"):
            with self.assertRaises(ts.SourceError):
                ts.separate_hq(self.work)

    def test_progress_callback_receives_messages(self):
        self._make_raw44k()
        vocals = self._sep_hq_dir() / VOCALS_NAME
        msgs = []
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(
                                      self, stdout_text="RESULT=%s\n" % vocals,
                                      write_on_run=[vocals])):
            ts.separate_hq(self.work, progress=msgs.append)
        self.assertTrue(msgs)

    def test_failed_reseparation_keeps_state_of_untouched_good_track(self):
        """重分离失败不该毁掉上一次成功那条轨的状态（否则白扔一次 3 分钟分离）。

        本次要的键 == 状态里的键，且键里记的人声轨还在（`before` 快照里就有，
        不会被当成这次的新文件删掉）-> 状态必须留下。
        """
        self._make_raw44k()
        vocals = self._make_vocals_hq()
        self._write_hq_state()
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, returncode=1,
                                                         stderr_text="崩了\n")):
            with self.assertRaises(ts.SourceError):
                ts.separate_hq(self.work, media=self.media)
        self.assertTrue(self._state_path().exists(), "键相同的状态不该被删")
        self.assertEqual(ts.reuse_hq(self.work,
                                     ts.state_key(self.media, ts.DEFAULT_MODEL)),
                         vocals)

    def test_failed_reseparation_with_other_key_drops_state(self):
        """键对不上（换了原片/模型）-> 旧状态必须作废，不能指向不存在的轨。"""
        self._make_raw44k()
        self._make_vocals_hq()
        self._write_hq_state()
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, returncode=1,
                                                         stderr_text="崩了\n")):
            with self.assertRaises(ts.SourceError):
                ts.separate_hq(self.work, key=["另一个", "原片", "的键"])
        self.assertFalse(self._state_path().exists())

    def test_stopped_reseparation_keeps_state_of_untouched_good_track(self):
        self._make_raw44k()
        self._make_vocals_hq()
        self._write_hq_state()
        partial = self._sep_hq_dir() / "raw44k_(Instrumental)_half.wav"

        def factory(args, **kw):
            self.popen_count += 1
            return FakePopen(args, alive=True, write_on_run=[partial], **kw)

        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen", side_effect=factory):
            with self.assertRaises(ts.SourceCancelled):
                ts.separate_hq(self.work, media=self.media,
                               should_stop=lambda: self.popen_count > 0)
        self.assertFalse(partial.exists(), "半成品该删")
        self.assertTrue(self._state_path().exists(), "键相同的状态不该被删")


# ---------------------------------------------------------------------------
# 中断
# ---------------------------------------------------------------------------

class TestStop(StubbedSeparationCase):

    def test_stop_during_separation_raises_and_cleans_partials(self):
        self._make_raw44k()
        out = self._sep_hq_dir()
        partial = out / "raw44k_(Instrumental)_m.wav"
        holder = {}

        def factory(args, **kw):
            self.popen_count += 1
            proc = FakePopen(args, alive=True, write_on_run=[partial], **kw)
            holder["proc"] = proc
            return proc

        def should_stop():
            return self.popen_count > 0

        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen", side_effect=factory):
            with self.assertRaises(ts.SourceCancelled):
                ts.separate_hq(self.work, should_stop=should_stop)

        self.assertTrue(holder["proc"].terminated or holder["proc"].killed)
        self.assertFalse(self._state_path().exists())
        self.assertEqual(list(out.glob("*")), [],
                         "中断后不该留半成品：%s" % list(out.glob("*")))

    def test_stop_before_extract_is_honoured(self):
        """抽取阶段被 `should_stop` 打断：抛 SourceCancelled 且不留半条 raw44k。"""
        out = self.work / "raw44k.wav"
        holder = {}

        def factory(args, **kw):
            self.popen_count += 1
            proc = FakePopen(args, alive=True, write_on_run=[out], **kw)
            holder["proc"] = proc
            return proc

        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen", side_effect=factory):
            with self.assertRaises(ts.SourceCancelled):
                ts.extract_full_rate(self.media, self.work,
                                     should_stop=lambda: self.popen_count > 0)
        self.assertTrue(holder["proc"].terminated or holder["proc"].killed)
        self.assertFalse(out.exists(), "被停掉的抽取不该留下半条 raw44k")

    def test_pick_source_honours_stop_without_running_ffmpeg(self):
        """`pick_source` 把 `should_stop` 透到抽取层：一开始就要求停 -> 不跑 ffmpeg。"""
        def factory(args, **kw):
            self.popen_count += 1
            return FakePopen(args, alive=True, **kw)

        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen", side_effect=factory):
            with self.assertRaises(ts.SourceCancelled):
                ts.pick_source(self.work, self.media, mode="hq",
                               should_stop=lambda: True)
        self.assertEqual(self.popen_count, 1)
        self.assertFalse((self.work / "raw44k.wav").exists())

    def test_pick_source_propagates_cancellation(self):
        with mock.patch.object(ts, "extract_full_rate",
                               side_effect=ts.SourceCancelled("停了")):
            with self.assertRaises(ts.SourceCancelled):
                ts.pick_source(self.work, self.media, mode="hq")

    def test_cancelled_separation_does_not_mark_state_reusable(self):
        """中断 -> 下次必须重新分离（状态文件没写成）。"""
        self._make_raw44k()
        out = self._sep_hq_dir()
        partial = out / "raw44k_(Vocals)_partial.wav"

        def factory(args, **kw):
            self.popen_count += 1
            return FakePopen(args, alive=True, write_on_run=[partial], **kw)

        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen", side_effect=factory):
            with self.assertRaises(ts.SourceCancelled):
                ts.separate_hq(self.work, should_stop=lambda: self.popen_count > 0)
        self.assertIsNone(ts.reuse_hq(self.work, ts.state_key(self.media,
                                                             ts.DEFAULT_MODEL)))


# ---------------------------------------------------------------------------
# 复用判断 / 清理
# ---------------------------------------------------------------------------

class TestStateAndCleanup(BaseCase):

    def test_state_key_is_stable_and_reacts_to_each_field(self):
        k0 = ts.state_key(self.media, ts.DEFAULT_MODEL)
        self.assertEqual(k0, ts.state_key(self.media, ts.DEFAULT_MODEL))
        self.assertNotEqual(k0, ts.state_key(self.media, "别的模型.ckpt"))
        with self.media.open("ab") as fh:
            fh.write(b"x")
        self.assertNotEqual(k0, ts.state_key(self.media, ts.DEFAULT_MODEL))

    def test_state_key_reacts_to_mtime(self):
        k0 = ts.state_key(self.media, ts.DEFAULT_MODEL)
        st = self.media.stat()
        os.utime(self.media, (st.st_atime - 100, st.st_mtime - 100))
        self.assertNotEqual(k0, ts.state_key(self.media, ts.DEFAULT_MODEL))

    def test_corrupt_state_is_ignored(self):
        self._write_hq_state()
        self._state_path().write_text("{不是 json", encoding="utf-8")
        self.assertIsNone(ts.reuse_hq(self.work,
                                      ts.state_key(self.media, ts.DEFAULT_MODEL)))

    def test_state_hit_requires_vocals_file_to_exist(self):
        self._sep_hq_dir()
        self._write_hq_state()
        self.assertIsNone(ts.reuse_hq(self.work,
                                      ts.state_key(self.media, ts.DEFAULT_MODEL)))

    def test_state_hit_rejects_a_vocals_path_outside_work(self):
        """P3：`.state.json` 里记的绝对路径**必须落在 work 之内**才能复用。

        `.state.json` 是给人看、也可以手改的 JSON。里面记一条**别的作品目录**
        的人声轨时（审查者用合成数据复现过：把状态文件的 vocals 指到另一个
        work），老代码会照用不误 —— 导出就从别的集读音频，训练集里混进别人的
        声音，而且没有任何报错。
        """
        other = self.dir / "别的作品" / "sep_hq"
        other.mkdir(parents=True, exist_ok=True)
        outsider = write_wav(other / VOCALS_NAME, seconds=0.2)
        self._sep_hq_dir()
        key = ts.state_key(self.media, ts.DEFAULT_MODEL)
        ts.write_state(self.work, key, vocals=outsider)

        hit = ts.reuse_hq(self.work, key)

        self.assertNotEqual(hit, outsider, "复用了 work 之外的音源")
        # 退回 glob 之后 work 里没有可复用的轨 -> 未命中
        self.assertIsNone(hit)

    def test_state_hit_rejects_a_parent_dir_vocals_path(self):
        """`..` 绕出去的路径同样不算命中。"""
        outsider = write_wav(self.dir / "raw44k_(Vocals)_m.wav", seconds=0.2)
        self._sep_hq_dir()
        key = ts.state_key(self.media, ts.DEFAULT_MODEL)
        ts.write_state(self.work, key,
                       vocals=self.work / "sep_hq" / ".." / ".." / outsider.name)

        self.assertIsNone(ts.reuse_hq(self.work, key))

    def test_state_hit_still_works_for_an_in_work_path(self):
        """回归：work **之内**的记录照旧命中（不能把正常复用一起拦掉）。"""
        vocals = self._make_vocals_hq()
        key = ts.state_key(self.media, ts.DEFAULT_MODEL)
        ts.write_state(self.work, key, vocals=vocals)

        self.assertEqual(ts.reuse_hq(self.work, key), vocals)

    def test_state_hit_falls_back_to_glob_when_the_record_is_outside(self):
        """记录越界但 work 里**确实**有轨时：退回 glob 找到它，而不是未命中。"""
        vocals = self._make_vocals_hq()
        other = self.dir / "别的作品"
        other.mkdir(parents=True, exist_ok=True)
        outsider = write_wav(other / VOCALS_NAME, seconds=0.2)
        key = ts.state_key(self.media, ts.DEFAULT_MODEL)
        ts.write_state(self.work, key, vocals=outsider)

        self.assertEqual(ts.reuse_hq(self.work, key), vocals)

    def test_state_key_mismatch_is_a_miss(self):
        self._make_vocals_hq()
        self._write_hq_state()
        self.assertIsNone(ts.reuse_hq(self.work, ["别的", "键"]))

    def test_write_state_is_atomic_and_roundtrips(self):
        key = ts.state_key(self.media, ts.DEFAULT_MODEL)
        ts.write_state(self.work, key, vocals=self.work / "sep_hq" / "v.wav")
        state = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertEqual(state["key"], key)
        self.assertEqual(list(self._state_path().parent.glob("*.tmp")), [])

    def test_cleanup_deletes_only_raw44k_and_instrumental(self):
        raw = self._make_raw44k()
        inst = self._make_instrumental_hq()
        vocals = self._make_vocals_hq()
        self._write_hq_state()
        mb = ts.cleanup_intermediates(self.work)
        self.assertFalse(raw.exists(), "raw44k.wav 该删")
        self.assertFalse(inst.exists(), "伴奏轨该删")
        self.assertTrue(vocals.exists(), "HQ 人声轨必须保留")
        self.assertTrue(self._state_path().exists(), "状态文件必须保留")
        self.assertGreater(mb, 0.0)

    def test_cleanup_is_idempotent_and_returns_zero(self):
        self._make_vocals_hq()
        ts.cleanup_intermediates(self.work)
        self.assertEqual(ts.cleanup_intermediates(self.work), 0.0)

    def test_cleanup_on_empty_work_dir(self):
        self.assertEqual(ts.cleanup_intermediates(self.work), 0.0)

    def test_cleanup_survives_missing_work_dir(self):
        self.assertEqual(ts.cleanup_intermediates(self.dir / "不存在"), 0.0)

    def test_cleanup_never_touches_files_outside_work(self):
        """T6：`cleanup_intermediates` 只能删 `work` **之内**的文件。

        审查者的变异 W30 往 `targets` 里塞了一条
        `Path(work).parent / "raw44k.wav"`（work 的**上一级**同名文件），
        617 条测试全绿 —— 也就是说「越界删除」这件事当时没有任何断言盯着。
        这里在 work 外面（它的上一级）放一个**同名**文件：清理之后它必须还在。
        名字故意取成与 `RAW_HQ_NAME` 一样，越界删除才会被这条抓到。
        """
        outside = self.dir / ts.RAW_HQ_NAME
        payload = "work 外面的同名文件，清理不许碰".encode("utf-8") * 100
        outside.write_bytes(payload)
        self._make_raw44k()
        self._make_instrumental_hq()

        ts.cleanup_intermediates(self.work)

        self.assertTrue(outside.exists(),
                        "cleanup 删掉了 work 之外的文件：%s" % outside)
        self.assertEqual(outside.read_bytes(), payload)

    def test_cleanup_never_touches_a_sibling_works_same_named_file(self):
        """同上，换成「兄弟作品的同名中间产物」这个更真实的形状。"""
        sibling = self.dir / "别的作品"
        sibling.mkdir(parents=True, exist_ok=True)
        raw = write_wav(sibling / ts.RAW_HQ_NAME, seconds=0.2)
        inst_dir = sibling / "sep_hq"
        inst_dir.mkdir(parents=True, exist_ok=True)
        inst = write_wav(inst_dir / INST_NAME, seconds=0.2)
        self._make_raw44k()
        self._make_instrumental_hq()

        ts.cleanup_intermediates(self.work)

        self.assertTrue(raw.exists(), "删到了别的作品的 raw44k.wav")
        self.assertTrue(inst.exists(), "删到了别的作品的伴奏轨")

    def test_state_hit_survives_cleanup(self):
        """清理掉 raw44k 之后，第二次调用仍复用 HQ 人声轨、不重分。"""
        self._make_vocals_hq()
        self._make_raw44k()
        self._write_hq_state()
        ts.cleanup_intermediates(self.work)
        with mock.patch.object(ts, "separate_hq") as m_sep:
            path, kind = ts.pick_source(self.work, self.media, mode="hq")
        self.assertEqual(kind, "separated_hq")
        self.assertTrue(path.exists())
        m_sep.assert_not_called()

    def test_pick_source_reuses_after_a_failed_reseparation(self):
        """重分离失败一次之后，`pick_source` 仍该直接复用那条完好的 HQ 轨。"""
        self._make_vocals_hq()
        self._make_raw44k()
        self._write_hq_state()
        with mock.patch.object(ts, "_POLL_S", 0.001), \
                mock.patch.object(ts.subprocess, "Popen",
                                  side_effect=fake_popen(self, returncode=1,
                                                         stderr_text="崩了\n")):
            # 先把状态删掉，逼出一次真分离尝试（模拟「复用判断没命中」的情形）
            ts.state_path(self.work).unlink()
            with self.assertRaises(ts.SourceError):
                ts.separate_hq(self.work, media=self.media)
        self._write_hq_state()          # 上一次成功那条轨的状态（重分离失败后被保住）
        with mock.patch.object(ts, "extract_full_rate") as m_ext, \
                mock.patch.object(ts, "separate_hq") as m_sep:
            path, kind = ts.pick_source(self.work, self.media, mode="hq")
        self.assertEqual(kind, "separated_hq")
        self.assertTrue(path.exists())
        m_ext.assert_not_called()
        m_sep.assert_not_called()


# ---------------------------------------------------------------------------
# high_band_ratio
# ---------------------------------------------------------------------------

class TestHighBandRatio(BaseCase):

    def test_tone_at_14k_has_high_ratio(self):
        p = write_wav(self.work / "hi.wav", seconds=2.0, freq=14000.0)
        r = ts.high_band_ratio(p, 10000, 16000)
        self.assertGreater(r, 0.5, r)

    def test_tone_at_1k_has_near_zero_ratio(self):
        p = write_wav(self.work / "lo.wav", seconds=2.0, freq=1000.0)
        r = ts.high_band_ratio(p, 10000, 16000)
        self.assertLess(r, 1e-6, r)

    def test_44k_file_with_only_3k_tone_is_empty_above_10k(self):
        """44.1 kHz 文件里只有 3 kHz -> 10~16 kHz 仍是空的（壳就是这个意思）。"""
        p = write_wav(self.work / "t44.wav", seconds=2.0, sr=44100, freq=3000.0)
        self.assertLess(ts.high_band_ratio(p, 10000, 16000), 1e-9)

    def test_upsampling_16k_does_not_create_high_band(self):
        """16 kHz 内容上采样到 44.1 kHz 后，10~16 kHz 依然为空。"""
        src = write_wav(self.work / "t16.wav", seconds=2.0, sr=16000, freq=3000.0)
        up = self.work / "t16_up.wav"
        ts.ffmpeg_convert(src, up, sr=44100, ac=1)
        self.assertLess(ts.high_band_ratio(up, 10000, 16000), 1e-9)

    def test_broadband_noise_gives_a_mid_ratio(self):
        """宽带噪声的 10~16 kHz 占比 = 频段宽度占比（6/22.05 约 0.27），量级对得上。"""
        rng = np.random.RandomState(11)
        n = 44100 * 2
        pcm = (np.clip(0.3 * rng.randn(n), -1.0, 1.0) * 32767).astype("<i2")
        p = self.work / "noise.wav"
        with wave.open(str(p), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(44100)
            fh.writeframes(pcm.tobytes())
        r = ts.high_band_ratio(p, 10000, 16000)
        self.assertGreater(r, 0.15, r)
        self.assertLess(r, 0.40, r)

    def test_short_file_does_not_raise(self):
        p = write_wav(self.work / "short.wav", seconds=0.05, freq=8000.0)
        self.assertGreaterEqual(ts.high_band_ratio(p), 0.0)

    def test_median_matches_manual_median(self):
        p = write_wav(self.work / "m.wav", seconds=11.0, freq=8000.0)
        spots = [0.0, 3.0, 6.0, 9.0]
        manual = sorted(ts.high_band_ratio(p, start=s, seconds=1.0) for s in spots)
        mid = 0.5 * (manual[1] + manual[2])
        self.assertAlmostEqual(
            ts.high_band_ratio_median(p, spots=spots, seconds=1.0), mid, places=12)

    def test_median_on_unreadable_file_returns_zero(self):
        self.assertEqual(ts.high_band_ratio_median(self.work / "没有.wav"), 0.0)

    def test_wav_seconds(self):
        p = write_wav(self.work / "s.wav", seconds=2.5)
        self.assertAlmostEqual(ts._wav_seconds(p), 2.5, places=3)
        self.assertIsNone(ts._wav_seconds(self.work / "没有.wav"))


# ---------------------------------------------------------------------------
# 真实链路（默认 skip）
# ---------------------------------------------------------------------------

RUN_DIR = TMP_ROOT / "t3_integration"


@unittest.skipUnless(_enabled(), "需要 JISHENG_RUN_INTEGRATION=1 才运行")
@unittest.skipUnless(MEDIA_SRC.exists(), "缺少测试素材 %s" % MEDIA_SRC)
@unittest.skipUnless(REF_16K.exists(), "缺少 16 kHz 对照轨 %s" % REF_16K)
@unittest.skipUnless(ts.SEP_PYTHON.exists(), "缺少分离环境 %s" % ts.SEP_PYTHON)
class TestHqSeparationIntegration(unittest.TestCase):
    """Task 3 存在的唯一理由：真跑一次全采样率分离，量出 10~16 kHz 能量占比。

    两条音轨都按 44.1 kHz 分析；现有 16 kHz 轨先上采样到 44.1 kHz
    （否则它根本没有 10~16 kHz 这个频段，比较没有意义）。

    工作目录复用系统临时目录中的 `t3_integration`，已经跑过就直接复用 HQ
    人声轨（但这里**故意不清理**中间产物，免得反复跑掉 3 分钟 GPU）。
    """

    @classmethod
    def setUpClass(cls):
        import time
        cls.work = RUN_DIR / MEDIA_SRC.stem
        cls.work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cls.hq, cls.kind = ts.pick_source(cls.work, MEDIA_SRC, mode="hq")
        cls.sep_seconds = time.time() - t0
        cls.raw44k = ts.extract_full_rate(MEDIA_SRC, cls.work)
        cls.upsampled = cls.work / "ref16k_upsampled_44k.wav"
        ts.ffmpeg_convert(REF_16K, cls.upsampled, sr=44100, ac=1)

    @classmethod
    def tearDownClass(cls):
        # 跑完顺手清理中间产物（raw44k + 伴奏轨，约 260 MB），**保留 HQ 人声轨**，
        # 下次再跑这条用例直接复用、不重跑 3 分钟 GPU。
        freed = ts.cleanup_intermediates(cls.work)
        print("\n[集成] 工作目录 %s；清理中间产物释放 %.1f MB，保留 HQ 人声轨"
              % (cls.work, freed))

    def test_00_work_summary(self):
        print("\n[集成] source_kind=%s" % self.kind)
        print("[集成] HQ 轨 = %s" % self.hq)
        print("[集成] 抽取+分离耗时 %.1f s" % self.sep_seconds)
        self.assertEqual(self.kind, "separated_hq")

    def test_01_source_kind_is_hq(self):
        print("\n[集成] source_kind=%s  HQ=%s" % (self.kind, self.hq))
        print("[集成] 抽取+分离耗时 %.1f s" % self.sep_seconds)
        self.assertEqual(self.kind, "separated_hq")
        self.assertTrue(self.hq.exists())

    def test_02_hq_is_44100_hz(self):
        sr = ts.probe_sample_rate(self.hq)
        print("\n[集成] HQ 采样率 = %s Hz  (ffprobe)" % sr)
        self.assertEqual(sr, 44100)

    def test_03_high_band_ratio_absolute(self):
        hq = ts.high_band_ratio_median(self.hq)
        raw = ts.high_band_ratio_median(self.raw44k)
        ref = ts.high_band_ratio_median(self.upsampled)
        print("\n[集成] 10~16 kHz 能量占比中位数: HQ=%.6e  raw44k=%.6e  现有16k轨=%.6e"
              % (hq, raw, ref))
        print("[集成] 单点(0s): HQ=%.6e raw44k=%.6e 现有16k轨=%.6e"
              % (ts.high_band_ratio(self.hq), ts.high_band_ratio(self.raw44k),
                 ts.high_band_ratio(self.upsampled)))
        self.assertGreater(hq, 1e-3, "HQ 高频占比中位数 %.4e 低于绝对下限 1e-3" % hq)
        self.assertGreater(hq, raw * 0.3, "HQ %.4e 比原片 %.4e 低太多" % (hq, raw))

    def test_04_high_band_ratio_beats_16k_source(self):
        hq = ts.high_band_ratio_median(self.hq)
        ref = ts.high_band_ratio_median(self.upsampled)
        print("\n[集成] HQ 中位=%.6e vs 现有16k轨上采样到44.1k 中位=%.6e  (倍数 %.1f)"
              % (hq, ref, (hq / ref) if ref else float("inf")))
        self.assertGreater(hq, ref * 100,
                           "HQ %.4e 未达到现有 16k 轨 %.4e 的 100 倍" % (hq, ref))


if __name__ == "__main__":
    unittest.main(verbosity=2)
