#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""试听片段挑选的单元测试。

挑选/打分/时区逻辑用构造数据测；`TestCutAndCache` 会真的调用 ffmpeg
切真实的 16k 单声道 WAV，再用 stdlib `wave` 读回校验（需要 ffmpeg 在 PATH）。
"""

import json
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_clips as sc  # noqa: E402


def seg(start, end, speaker, text="台词"):
    return {"start": float(start), "end": float(end),
            "speaker": speaker, "text": text}


class TestOverlapMath(unittest.TestCase):
    def test_overlap_seconds_clamps_at_zero(self):
        self.assertEqual(sc.overlap_seconds(0.0, 1.0, 2.0, 3.0), 0.0)

    def test_overlap_seconds_basic(self):
        self.assertAlmostEqual(sc.overlap_seconds(0.0, 5.0, 3.0, 9.0), 2.0)

    def test_merged_seconds_unions_overlapping_intervals(self):
        # [0,6] 与 [3,5] 有并集 6 秒，不能算成 8 秒
        self.assertAlmostEqual(sc.merged_seconds([(0.0, 6.0), (3.0, 5.0)]), 6.0)

    def test_merged_seconds_disjoint(self):
        self.assertAlmostEqual(sc.merged_seconds([(0.0, 1.0), (2.0, 4.0)]), 3.0)

    def test_merged_seconds_empty(self):
        self.assertEqual(sc.merged_seconds([]), 0.0)

    def test_other_speaker_overlap_ignores_the_cluster_itself(self):
        dia = [{"start": 0.0, "end": 10.0, "speaker": "A"},
               {"start": 0.0, "end": 6.0, "speaker": "B"},
               {"start": 3.0, "end": 5.0, "speaker": "C"}]
        # B 覆盖 0~6，C 完全落在 B 里 -> 并集 6 秒；A 自己不参与
        self.assertAlmostEqual(sc.other_speaker_overlap(seg(0, 10, "A"), dia, "A"), 6.0)

    def test_other_speaker_overlap_is_capped_by_segment_length(self):
        dia = [{"start": 0.0, "end": 100.0, "speaker": "B"}]
        self.assertAlmostEqual(sc.other_speaker_overlap(seg(10, 14, "A"), dia, "A"), 4.0)


class TestScore(unittest.TestCase):
    def test_longer_scores_higher(self):
        self.assertGreater(sc.score_candidate(8.0, 0.0, 1.0),
                           sc.score_candidate(4.0, 0.0, 1.0))

    def test_less_overlap_scores_higher(self):
        self.assertGreater(sc.score_candidate(8.0, 0.0, 1.0),
                           sc.score_candidate(8.0, 0.5, 1.0))

    def test_louder_scores_higher(self):
        self.assertGreater(sc.score_candidate(8.0, 0.0, 1.0),
                           sc.score_candidate(8.0, 0.0, 0.0))

    def test_score_is_bounded(self):
        self.assertGreaterEqual(sc.score_candidate(3.0, 0.3, 0.4), 0.0)
        self.assertLessEqual(sc.score_candidate(3.0, 0.3, 0.4), 1.0)


class TestPickCandidates(unittest.TestCase):
    def test_rejects_outside_duration_window(self):
        segs = [seg(0, 1.0, "S"), seg(10, 14.0, "S"), seg(20, 40.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=3, episode_end=100.0)
        self.assertEqual([c["dur"] for c in out], [4.0])

    def test_ignores_other_clusters(self):
        segs = [seg(0, 5.0, "S"), seg(10, 15.0, "T")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=3, episode_end=100.0)
        self.assertEqual([round(c["start"], 1) for c in out], [0.0])

    def test_overlapped_segment_is_deferred_when_limit_is_reached(self):
        segs = [seg(0, 5.0, "S"), seg(10, 15.0, "S")]
        dia = [{"start": 0.0, "end": 5.0, "speaker": "S"},
               {"start": 1.0, "end": 4.0, "speaker": "T"}]
        out = sc.pick_clip_candidates(segs, dia, "S", limit=1, episode_end=100.0)
        self.assertEqual([round(c["start"], 1) for c in out], [10.0])
        self.assertFalse(out[0]["overlap"])

    def test_overlapped_segment_is_used_as_last_resort_and_flagged(self):
        segs = [seg(0, 6.0, "S"), seg(20, 26.0, "S")]
        dia = [{"start": 1.0, "end": 3.0, "speaker": "T"}]
        out = sc.pick_clip_candidates(segs, dia, "S", limit=2, episode_end=100.0)
        self.assertEqual([round(c["start"], 1) for c in out], [0.0, 20.0])
        self.assertTrue(out[0]["overlap"])
        self.assertAlmostEqual(out[0]["overlap_s"], 2.0)
        self.assertFalse(out[1]["overlap"])

    def test_one_clip_per_time_zone(self):
        # 整集 0~100 秒，limit=2 -> 两个区 [0,50) 与 [50,100]
        segs = [seg(0, 4.0, "S"), seg(10, 18.0, "S"), seg(60, 65.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=2, episode_end=100.0)
        self.assertEqual(sorted(round(c["start"], 1) for c in out), [10.0, 60.0])

    def test_spread_beats_raw_score_when_zones_compete(self):
        """时区分散必须优先于纯按分数：zone0 有两条同分，zone1 有一条更低分的，
        正确结果是 zone0 取一条 + zone1 取一条，而不是 zone0 的两条。"""
        segs = [seg(0, 8.0, "S"), seg(10, 18.0, "S"), seg(60, 65.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=2, episode_end=100.0)
        self.assertEqual([round(c["start"], 1) for c in out], [0.0, 60.0])

    def test_fills_by_score_when_a_zone_is_empty(self):
        segs = [seg(0, 8.0, "S"), seg(2, 9.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=2, episode_end=100.0)
        self.assertEqual(sorted(round(c["start"], 1) for c in out), [0.0, 2.0])

    def test_relaxed_window_used_when_too_few(self):
        segs = [seg(0, 12.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=3, episode_end=100.0)
        self.assertEqual([c["dur"] for c in out], [12.0])

    def test_never_exceeds_relaxed_max(self):
        segs = [seg(0, 20.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=3, episode_end=100.0)
        self.assertEqual(out, [])

    def test_numbers_runs_from_one_in_time_order(self):
        segs = [seg(10, 16.0, "S"), seg(0, 6.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=2, episode_end=100.0)
        self.assertEqual([c["n"] for c in out], [1, 2])
        self.assertEqual([round(c["start"], 1) for c in out], [0.0, 10.0])

    def test_episode_end_defaults_to_last_segment(self):
        segs = [seg(0, 5.0, "S"), seg(100, 106.0, "S")]
        out = sc.pick_clip_candidates(segs, [], "S", limit=2)
        self.assertEqual(len(out), 2)

    def test_audio_stats_drops_clipped_clip(self):
        segs = [seg(0, 5.0, "S"), seg(20, 25.0, "S")]

        def stats(start, end):
            # 第一段削波（peak 1.0），第二段正常
            return (0.2, 1.0) if start < 10 else (0.4, 0.7)

        out = sc.pick_clip_candidates(segs, [], "S", limit=3,
                                      episode_end=100.0, audio_stats=stats)
        self.assertEqual([round(c["start"], 1) for c in out], [20.0])
        self.assertIsNotNone(out[0]["peak_dbfs"])

    def test_audio_stats_prefers_louder_clip(self):
        segs = [seg(0, 5.0, "S"), seg(20, 25.0, "S")]

        def stats(start, end):
            return (0.1, 0.2) if start < 10 else (0.9, 0.9)

        out = sc.pick_clip_candidates(segs, [], "S", limit=1,
                                      episode_end=100.0, audio_stats=stats)
        self.assertEqual([round(c["start"], 1) for c in out], [20.0])

    def test_no_candidates_returns_empty(self):
        self.assertEqual(sc.pick_clip_candidates([], [], "S"), [])

    def test_zero_limit_returns_empty(self):
        segs = [seg(0, 5.0, "S")]
        self.assertEqual(sc.pick_clip_candidates(segs, [], "S", limit=0), [])


def write_wav16k(path, seconds, freq=440.0):
    """写一个真实的 16k 单声道 WAV，用于验证 ffmpeg 切出来的时长。"""
    rate = 16000
    n = int(rate * seconds)
    t = np.arange(n) / rate
    pcm = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())


def wav_info(path):
    with wave.open(str(path), "rb") as fh:
        return fh.getnchannels(), fh.getframerate(), fh.getnframes() / fh.getframerate()


class TestPreviewSource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_prefers_separated_vocals(self):
        (self.work / "vocals_16k.wav").write_bytes(b"RIFF")
        (self.work / "raw16k.wav").write_bytes(b"RIFF")
        self.assertEqual(sc.preview_source(self.work).name, "vocals_16k.wav")

    def test_falls_back_to_raw(self):
        (self.work / "raw16k.wav").write_bytes(b"RIFF")
        self.assertEqual(sc.preview_source(self.work).name, "raw16k.wav")

    def test_none_when_nothing_available(self):
        self.assertIsNone(sc.preview_source(self.work))


class _CutSpy:
    """临时把 sc.cut_clip 换成记录调用的 spy，退出时还原（手法同复用缓存测试）。"""

    def __enter__(self):
        self.calls = []
        self._orig = sc.cut_clip

        def spy(*a, **k):
            self.calls.append(a)
            return self._orig(*a, **k)

        sc.cut_clip = spy
        return self

    def __exit__(self, *exc):
        sc.cut_clip = self._orig
        return False


class TestCutAndCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.src = self.work / "vocals_16k.wav"
        write_wav16k(self.src, 4.0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cut_clip_produces_16k_mono_of_expected_length(self):
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        sc.cut_clip(self.src, 0.5, 1.5, dest)
        self.assertTrue(dest.is_file())
        ch, rate, dur = wav_info(dest)
        self.assertEqual(ch, 1)
        self.assertEqual(rate, 16000)
        self.assertAlmostEqual(dur, 1.5, delta=0.06)

    def test_clip_path_is_sanitised(self):
        self.assertEqual(sc.clip_path(self.work, "SPEAKER_00", 2).name, "SPEAKER_00_2.wav")
        self.assertEqual(sc.clip_path(self.work, "?", 1).name, "unknown_1.wav")
        self.assertEqual(sc.clip_path(self.work, "a/b", 1).name, "a_b_1.wav")

    def test_ensure_clips_cuts_and_records_paths(self):
        cands = [{"n": 1, "start": 0.0, "dur": 1.0},
                 {"n": 2, "start": 2.0, "dur": 1.0}]
        out = sc.ensure_clips(self.work, "SPEAKER_00", cands, self.src)
        self.assertEqual(len(out), 2)
        for rec, cand in zip(out, cands):
            self.assertTrue(Path(rec["path"]).is_file())
            # 候选自身的键必须原样带回来（调用方要靠 n/start/dur 渲染时间轴）
            self.assertEqual(rec["n"], cand["n"])
            self.assertEqual(rec["start"], cand["start"])
            self.assertEqual(rec["dur"], 1.0)

    def test_stale_candidate_range_is_recut(self):
        """断点续跑后重挑候选：n 相同但时间段变了，必须重切而不是复用旧音频。"""
        sc.ensure_clips(self.work, "SPEAKER_00",
                        [{"n": 1, "start": 0.0, "dur": 1.0}], self.src)
        with _CutSpy() as spy:
            again = sc.ensure_clips(self.work, "SPEAKER_00",
                                    [{"n": 1, "start": 1.5, "dur": 1.0}], self.src)
        self.assertEqual(len(spy.calls), 1)
        path = Path(again[0]["path"])
        self.assertAlmostEqual(wav_info(path)[2], 1.0, delta=0.06)
        # 内容确实是 start=1.5 那一段（和现切的一模一样）
        fresh = self.work / "fresh_from_150.wav"
        sc.cut_clip(self.src, 1.5, 1.0, fresh)
        self.assertEqual(path.read_bytes(), fresh.read_bytes())
        meta = json.loads(sc._clip_meta(path).read_text(encoding="utf-8"))
        self.assertEqual(meta["start"], 1.5)

    def test_source_change_invalidates_cache(self):
        """raw16k.wav -> vocals_16k.wav 换源后，旧片段必须重切。"""
        sc.ensure_clips(self.work, "SPEAKER_00",
                        [{"n": 1, "start": 0.0, "dur": 1.0}], self.src)
        other = self.work / "raw16k.wav"
        write_wav16k(other, 4.0, freq=880.0)
        with _CutSpy() as spy:
            sc.ensure_clips(self.work, "SPEAKER_00",
                            [{"n": 1, "start": 0.0, "dur": 1.0}], other)
        self.assertEqual(len(spy.calls), 1)

    def test_source_content_change_invalidates_cache(self):
        """同一文件名但内容变了（大小/mtime 变）也要重切。"""
        sc.ensure_clips(self.work, "SPEAKER_00",
                        [{"n": 1, "start": 0.0, "dur": 1.0}], self.src)
        write_wav16k(self.src, 5.0)          # 同路径、更长 -> size 必变
        with _CutSpy() as spy:
            sc.ensure_clips(self.work, "SPEAKER_00",
                            [{"n": 1, "start": 0.0, "dur": 1.0}], self.src)
        self.assertEqual(len(spy.calls), 1)

    def test_missing_signature_forces_recut(self):
        """切到一半被打断（有 WAV、没签名）下次必须重切。"""
        first = sc.ensure_clips(self.work, "SPEAKER_00",
                                [{"n": 1, "start": 0.0, "dur": 1.0}], self.src)
        sc._clip_meta(first[0]["path"]).unlink()
        with _CutSpy() as spy:
            again = sc.ensure_clips(self.work, "SPEAKER_00",
                                    [{"n": 1, "start": 0.0, "dur": 1.0}], self.src)
        self.assertEqual(len(spy.calls), 1)
        self.assertTrue(sc._clip_meta(again[0]["path"]).is_file())

    def test_ensure_clips_reuses_cache(self):
        cands = [{"n": 1, "start": 0.0, "dur": 1.0}]
        first = sc.ensure_clips(self.work, "SPEAKER_00", cands, self.src)
        before = Path(first[0]["path"]).stat().st_mtime_ns
        calls = []
        orig = sc.cut_clip

        def spy(*a, **k):
            calls.append(a)
            return orig(*a, **k)

        sc.cut_clip = spy
        try:
            again = sc.ensure_clips(self.work, "SPEAKER_00", cands, self.src)
        finally:
            sc.cut_clip = orig
        self.assertEqual(calls, [])
        self.assertEqual(Path(again[0]["path"]).stat().st_mtime_ns, before)


class TestAtomicClips(unittest.TestCase):
    """试听片段必须原子落盘：并发 GET /api/clip 不能读到半截 WAV。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.src = self.work / "vocals_16k.wav"
        write_wav16k(self.src, 6.0)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _tmp_leftovers(directory):
        return [p.name for p in Path(directory).iterdir() if ".tmp" in p.name]

    def test_cut_clip_leaves_no_temp_file_behind(self):
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        sc.cut_clip(self.src, 0.0, 2.0, dest)
        self.assertTrue(dest.is_file())
        self.assertAlmostEqual(wav_info(dest)[2], 2.0, delta=0.06)
        self.assertEqual(self._tmp_leftovers(dest.parent), [])

    def test_cut_clip_removes_the_temp_file_when_ffmpeg_fails(self):
        """失败不能留下垃圾临时文件（它们会被当成缓存内容留在目录里）。"""
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        with self.assertRaises(Exception):
            sc.cut_clip(self.work / "不存在的源.wav", 0.0, 1.0, dest)
        self.assertFalse(dest.is_file())
        self.assertEqual(self._tmp_leftovers(dest.parent), [])

    def test_cut_clip_replaces_existing_dest_completely(self):
        """dest 已存在（旧片段）时必须被整段替换，不能是「覆写一部分」。"""
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        sc.cut_clip(self.src, 0.0, 4.0, dest)
        sc.cut_clip(self.src, 0.0, 1.0, dest)
        self.assertAlmostEqual(wav_info(dest)[2], 1.0, delta=0.06)

    def test_old_clip_stays_intact_until_the_replacement_is_complete(self):
        """切新片段期间，dest 路径上必须是「旧片段原封不动」或「不存在」，
        绝不能是 ffmpeg 正在写的半截文件 —— 否则并发 GET /api/clip 会读坏。

        ffmpeg 参数由 spy 在真实调用前记录，所以不需要猜时序。
        """
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        sc.cut_clip(self.src, 0.0, 1.0, dest)
        old_bytes = dest.read_bytes()

        observed = []
        orig = sc.subprocess.run

        def spy(cmd, *a, **k):
            out = Path(cmd[-1])
            if out == dest:
                observed.append(("direct", None))
            else:
                # 写到别处：dest 此刻必须还是那条完整的旧片段
                observed.append(("elsewhere",
                                 dest.read_bytes() if dest.is_file() else None))
            return orig(cmd, *a, **k)

        sc.subprocess.run = spy
        try:
            sc.cut_clip(self.src, 0.0, 3.0, dest)
        finally:
            sc.subprocess.run = orig

        self.assertTrue(observed, "spy 没被调用，测试本身失效")
        for kind, snapshot in observed:
            self.assertNotEqual(kind, "direct",
                                "ffmpeg 直接写最终路径：并发读到半截 WAV")
            self.assertIn(snapshot, (None, old_bytes),
                          "切分期间 dest 已经变成了半截文件")
        self.assertAlmostEqual(wav_info(dest)[2], 3.0, delta=0.06)
        self.assertEqual(self._tmp_leftovers(dest.parent), [])

    def test_ensure_clips_with_missing_source_returns_empty(self):
        cands = [{"n": 1, "start": 0.0, "dur": 1.0}]
        out = sc.ensure_clips(self.work, "SPEAKER_00", cands,
                              self.work / "没有这个文件.wav")
        self.assertEqual(out, [])

    def test_ensure_clips_survives_concurrent_callers(self):
        """同一 work/聚类/候选并发跑 8 次：不抛，且最终文件是完整 WAV。

        预期帧数写死（16000 * 2.0），所以半截写入 / 拿旧长度覆写都会被读出来。
        """
        cands = [{"n": 1, "start": 0.0, "dur": 2.0}]
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        errors = []
        barrier = threading.Barrier(8, timeout=10)

        def worker():
            try:
                barrier.wait()
                sc.ensure_clips(self.work, "SPEAKER_00", cands, self.src)
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        self.assertTrue(all(not t.is_alive() for t in threads), "有线程卡住没结束")
        self.assertTrue(dest.is_file())
        ch, rate, dur = wav_info(dest)
        self.assertEqual((ch, rate), (1, 16000))
        self.assertAlmostEqual(dur, 2.0, delta=0.06)
        self.assertEqual(self._tmp_leftovers(dest.parent), [])

    def test_concurrent_recut_with_a_shorter_candidate_leaves_a_whole_file(self):
        """并发重切同一目标且新候选更短：文件长度必须是新的完整长度。

        非原子实现里，重复切同一路径会让新片段的写入覆盖到旧片段的长度上
        （或反过来），写出帧数与文件头声明不一致的 WAV。
        """
        dest = sc.clip_path(self.work, "SPEAKER_00", 1)
        sc.cut_clip(self.src, 0.0, 4.0, dest)          # 先放一条 4 秒的旧缓存
        long_c = [{"n": 1, "start": 0.0, "dur": 4.0}]
        short_c = [{"n": 1, "start": 0.0, "dur": 1.0}]
        errors = []

        def a():
            try:
                for _ in range(6):
                    sc.ensure_clips(self.work, "SPEAKER_00", long_c, self.src)
            except Exception as exc:      # noqa: BLE001
                errors.append("A:" + repr(exc))

        def b():
            try:
                for _ in range(6):
                    sc.ensure_clips(self.work, "SPEAKER_00", short_c, self.src)
            except Exception as exc:      # noqa: BLE001
                errors.append("B:" + repr(exc))

        threads = [threading.Thread(target=a, daemon=True),
                   threading.Thread(target=b, daemon=True)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        self.assertTrue(all(not t.is_alive() for t in threads))
        with wave.open(str(dest), "rb") as fh:      # 帧数必须自洽
            self.assertGreater(fh.getnframes(), 0)
        _, _, dur = wav_info(dest)
        self.assertIn(round(dur, 2), (1.0, 4.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
