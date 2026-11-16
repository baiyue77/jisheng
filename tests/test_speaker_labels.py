#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""说话人标注状态与字幕重写。"""

import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_labels as sl  # noqa: E402


def srt_of(data, segments, tmp):
    srt = Path(tmp) / "a.srt"
    txt = Path(tmp) / "a.txt"
    sl.rewrite_outputs(data, segments, srt, txt)
    return srt.read_text(encoding="utf-8"), txt.read_text(encoding="utf-8")


def line(start, end, text, speaker):
    return {"start": float(start), "end": float(end), "text": text, "speaker": speaker}


class _RecordingLock:
    """把「谁在锁里做过事」记下来的锁替换品，用来确定性地验证持锁行为。"""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.acquires = 0

    def __enter__(self):
        self.acquires += 1
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_gives_empty_labels(self):
        data = sl.load_labels(self.work)
        self.assertEqual(data["clusters"], {})
        self.assertEqual(data["segment_overrides"], {})

    def test_corrupt_file_gives_empty_labels(self):
        (self.work / "speakers.json").write_text("{ 不是 json", encoding="utf-8")
        self.assertEqual(sl.load_labels(self.work)["clusters"], {})

    def test_roundtrip_keeps_names_and_overrides(self):
        data = sl.new_labels("jid", "D:\\a\\EP01.mkv", "separated")
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        sl.set_override(data, 7, "SPEAKER_01")
        sl.save_labels(self.work, data)

        got = sl.load_labels(self.work)
        self.assertEqual(sl.cluster_name(got, "SPEAKER_00"), "艾丽丝")
        self.assertEqual(got["segment_overrides"]["7"], "SPEAKER_01")
        self.assertEqual(got["job"], "jid")

    def test_write_is_atomic_leaves_no_tmp(self):
        data = sl.new_labels()
        sl.save_labels(self.work, data)
        self.assertEqual([p.name for p in self.work.iterdir()], ["speakers.json"])

    def test_concurrent_saves_all_succeed(self):
        """HTTP 处理是多线程的，同一目录并发保存不能有任何一个抛异常。"""
        import threading

        errors = []

        def worker(i):
            data = sl.new_labels()
            sl.set_names(data, {"SPEAKER_00": f"角色{i}"})
            try:
                for _ in range(10):
                    sl.save_labels(self.work, data)
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual([p.name for p in self.work.iterdir()], ["speakers.json"])

    def test_save_lock_is_reentrant(self):
        """持锁的辅助函数里再调 save_labels 不能自死锁（_wipe_checkpoint 就是这种调用）。

        确定性判断：普通 Lock 下同一线程第二次获取必然阻塞，join(timeout) 会超时。
        """
        done = threading.Event()
        errors = []

        def worker():
            try:
                with sl._SAVE_LOCK:
                    with sl._SAVE_LOCK:
                        sl.save_labels(self.work, sl.new_labels())
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertTrue(done.is_set(), "嵌套获取 _SAVE_LOCK 把线程卡死了")
        self.assertEqual(errors, [])

    def test_bootstrap_keeps_existing_names(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        sl.save_labels(self.work, data)

        sl.bootstrap(self.work, "jid2", "m.mkv", ["SPEAKER_00", "SPEAKER_01"],
                     "separated")
        got = sl.load_labels(self.work)
        self.assertEqual(sl.cluster_name(got, "SPEAKER_00"), "艾丽丝")
        self.assertIn("SPEAKER_01", got["clusters"])

    def test_non_dict_json_falls_back(self):
        (self.work / "speakers.json").write_text("[1,2,3]", encoding="utf-8")
        self.assertEqual(sl.load_labels(self.work)["clusters"], {})

    def test_missing_clusters_key_falls_back(self):
        (self.work / "speakers.json").write_text('{"version":1}', encoding="utf-8")
        self.assertEqual(sl.load_labels(self.work)["clusters"], {})

    def test_wrong_typed_fields_do_not_crash(self):
        (self.work / "speakers.json").write_text(
            '{"clusters":{"SPEAKER_00":{"name":5}},"segment_overrides":"abc",'
            '"work_name":7}', encoding="utf-8")
        data = sl.load_labels(self.work)
        self.assertEqual(data["segment_overrides"], {})
        self.assertIsNone(sl.cluster_name(data, "SPEAKER_00"))
        self.assertEqual(sl.get_work_name(data), "")
        self.assertIsNone(sl.effective_speaker(data, 0, line(0, 1, "x", None)))

    def test_bom_does_not_lose_names(self):
        (self.work / "speakers.json").write_text(
            "\ufeff" + '{"clusters":{"SPEAKER_00":{"name":"艾丽丝"}}}',
            encoding="utf-8")
        self.assertEqual(sl.cluster_name(sl.load_labels(self.work), "SPEAKER_00"),
                         "艾丽丝")

    def test_cluster_stored_as_bare_string_becomes_a_record(self):
        (self.work / "speakers.json").write_text(
            '{"clusters":{"SPEAKER_00":"艾丽丝"}}', encoding="utf-8")
        self.assertEqual(sl.cluster_name(sl.load_labels(self.work), "SPEAKER_00"),
                         "艾丽丝")

    def test_extra_cluster_keys_survive_a_reload(self):
        (self.work / "speakers.json").write_text(
            '{"clusters":{"SPEAKER_00":{"name":"艾丽丝","embedding":"page:1"}}}',
            encoding="utf-8")
        data = sl.load_labels(self.work)
        self.assertEqual(data["clusters"]["SPEAKER_00"]["embedding"], "page:1")

    def test_load_labels_takes_the_write_lock(self):
        """读也要持锁：读句柄会让写方的 os.replace 在 Windows 上报 WinError 5。

        用「锁获取计数」确定性判断，不做时间假设 —— 上一版用常驻线程 + 短等待，
        在没有锁的实现下也会通过，等于没测。
        """
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        sl.save_labels(self.work, data)          # 用真锁正常落盘

        spy = _RecordingLock()
        original = sl._SAVE_LOCK
        sl._SAVE_LOCK = spy
        try:
            got = sl.load_labels(self.work)
        finally:
            sl._SAVE_LOCK = original

        self.assertEqual(got["clusters"]["SPEAKER_00"]["name"], "艾丽丝")
        self.assertEqual(spy.acquires, 1, "load_labels 必须在写锁保护下读文件")


class TestSegmentOverridesSanitising(unittest.TestCase):
    """speakers.json 是给人手改的：segment_overrides 的值类型写错不能让请求 500。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, overrides_json):
        (self.work / "speakers.json").write_text(
            '{"clusters":{"SPEAKER_00":{"name":"艾丽丝"},'
            '"SPEAKER_01":{"name":"鲁迪乌斯"}},'
            '"segment_overrides":' + overrides_json + '}', encoding="utf-8")

    def test_list_value_falls_back_to_the_segment_cluster(self):
        self._write('{"0": ["SPEAKER_01"]}')
        data = sl.load_labels(self.work)
        # 坏条目整条丢掉，而不是 str() 成 "['SPEAKER_01']" 这种幽灵角色
        self.assertEqual(data["segment_overrides"], {})
        self.assertEqual(
            sl.effective_speaker(
                data, 0, line(0, 1, "x", "SPEAKER_00")), "艾丽丝")

    def test_int_and_none_values_are_dropped(self):
        self._write('{"0": 5, "1": null}')
        data = sl.load_labels(self.work)
        self.assertEqual(data["segment_overrides"], {})
        self.assertEqual(
            sl.effective_speaker(
                data, 0, line(0, 1, "x", "SPEAKER_00")), "艾丽丝")
        self.assertEqual(
            sl.effective_speaker(
                data, 1, line(0, 1, "x", "SPEAKER_00")), "艾丽丝")

    def test_string_values_still_work(self):
        self._write('{"0": "SPEAKER_01"}')
        data = sl.load_labels(self.work)
        self.assertEqual(
            sl.effective_speaker(
                data, 0, line(0, 1, "x", "SPEAKER_00")), "鲁迪乌斯")

    def test_keys_are_normalised_to_str(self):
        self._write('{"7": "SPEAKER_01"}')
        data = sl.load_labels(self.work)
        self.assertIn("7", data["segment_overrides"])

    def test_whitespace_only_string_is_dropped(self):
        self._write('{"0": "   "}')
        self.assertEqual(sl.load_labels(self.work)["segment_overrides"], {})

    def test_question_mark_override_survives(self):
        self._write('{"0": "?"}')
        data = sl.load_labels(self.work)
        self.assertEqual(
            sl.effective_speaker(data, 0, line(0, 1, "x", "SPEAKER_00")), "?")


class TestAtomicRewrite(unittest.TestCase):
    """重写字幕必须原子：写失败不能把用户已有的 SRT/TXT 毁成半截。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.srt = self.work / "a.srt"
        self.txt = self.work / "a.txt"

    def tearDown(self):
        self._tmp.cleanup()

    def _labels(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        return data

    def test_existing_srt_is_replaced_completely(self):
        self.srt.write_text("旧内容\n", encoding="utf-8")
        self.txt.write_text("旧内容\n", encoding="utf-8")
        sl.rewrite_outputs(self._labels(),
                           [line(0, 1, "新句子", "SPEAKER_00")], self.srt, self.txt)
        body = self.srt.read_text(encoding="utf-8")
        self.assertNotIn("旧内容", body)
        self.assertIn("[艾丽丝] 新句子", body)
        self.assertEqual(self.txt.read_text(encoding="utf-8"), "[艾丽丝] 新句子")

    def test_failed_replace_keeps_the_old_srt_and_txt(self):
        """os.replace 抛异常（磁盘满/被占用）时原文件必须一字未动。"""
        self.srt.write_text("用户的旧字幕\n", encoding="utf-8")
        self.txt.write_text("用户的旧文本\n", encoding="utf-8")
        orig = sl.A.os.replace
        sl.A.os.replace = lambda *a, **k: (_ for _ in ()).throw(
            OSError("磁盘满"))
        try:
            with self.assertRaises(OSError):
                sl.rewrite_outputs(self._labels(),
                                   [line(0, 1, "新句子", "SPEAKER_00")],
                                   self.srt, self.txt)
        finally:
            sl.A.os.replace = orig
        self.assertEqual(self.srt.read_text(encoding="utf-8"), "用户的旧字幕\n")
        self.assertEqual(self.txt.read_text(encoding="utf-8"), "用户的旧文本\n")

    def test_no_temp_file_left_after_a_successful_rewrite(self):
        sl.rewrite_outputs(self._labels(),
                           [line(0, 1, "句子", "SPEAKER_00")], self.srt, self.txt)
        self.assertEqual(sorted(p.name for p in self.work.iterdir()),
                         ["a.srt", "a.txt"])

    def test_no_temp_file_left_after_a_failed_rewrite(self):
        orig = sl.A.os.replace
        sl.A.os.replace = lambda *a, **k: (_ for _ in ()).throw(
            OSError("磁盘满"))
        try:
            with self.assertRaises(OSError):
                sl.rewrite_outputs(self._labels(),
                                   [line(0, 1, "句子", "SPEAKER_00")],
                                   self.srt, self.txt)
        finally:
            sl.A.os.replace = orig
        self.assertEqual([p.name for p in self.work.iterdir()], [])


class TestNaming(unittest.TestCase):
    def test_blank_name_clears(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        sl.set_names(data, {"SPEAKER_00": "   "})
        self.assertIsNone(sl.cluster_name(data, "SPEAKER_00"))

    def test_work_name_roundtrip(self):
        data = sl.new_labels()
        sl.set_work_name(data, "无职转生")
        self.assertEqual(sl.get_work_name(data), "无职转生")

    def test_effective_speaker_falls_back_to_cluster(self):
        data = sl.new_labels()
        self.assertEqual(sl.effective_speaker(data, 0, line(0, 1, "x", "SPEAKER_03")),
                         "SPEAKER_03")

    def test_effective_speaker_uses_name(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_03": "艾丽丝"})
        self.assertEqual(sl.effective_speaker(data, 0, line(0, 1, "x", "SPEAKER_03")),
                         "艾丽丝")

    def test_effective_speaker_none_stays_none(self):
        data = sl.new_labels()
        self.assertIsNone(sl.effective_speaker(data, 0, line(0, 1, "x", None)))

    def test_effective_speaker_unknown_stays_question_mark(self):
        data = sl.new_labels()
        self.assertEqual(sl.effective_speaker(data, 0, line(0, 1, "x", "?")), "?")

    def test_override_wins(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"})
        sl.set_override(data, 1, "SPEAKER_01")
        segs = [line(0, 1, "我", "SPEAKER_00"), line(2, 3, "不是", "SPEAKER_00")]
        self.assertEqual(sl.effective_speaker(data, 1, segs[1]), "鲁迪乌斯")
        self.assertEqual(sl.effective_speaker(data, 0, segs[0]), "艾丽丝")

    def test_apply_labels_does_not_mutate_input(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        segs = [line(0, 1, "x", "SPEAKER_00")]
        out = sl.apply_labels(data, segs)
        self.assertEqual(segs[0]["speaker"], "SPEAKER_00")
        self.assertEqual(out[0]["speaker"], "艾丽丝")


class TestRewrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_name_replaces_generic_label(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        body, txt = srt_of(data, [line(0, 1, "你好", "SPEAKER_00")], self.work)
        self.assertIn("[艾丽丝] 你好", body)
        self.assertNotIn("说话人1", body)
        self.assertEqual(txt, "[艾丽丝] 你好")

    def test_unnamed_cluster_keeps_generic_label(self):
        data = sl.new_labels()
        body, _ = srt_of(data, [line(0, 1, "x", "SPEAKER_01")], self.work)
        self.assertIn("[说话人2] x", body)

    def test_merge_two_clusters_share_one_name(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝", "SPEAKER_01": "艾丽丝"})
        segs = [line(0, 1, "a", "SPEAKER_00"), line(2, 3, "b", "SPEAKER_01")]
        body, _ = srt_of(data, segs, self.work)
        self.assertEqual(body.count("[艾丽丝]"), 2)

    def test_question_mark_survives(self):
        data = sl.new_labels()
        body, _ = srt_of(data, [line(0, 1, "啊", "?")], self.work)
        self.assertIn("[?] 啊", body)

    def test_none_speaker_gets_no_label(self):
        data = sl.new_labels()
        body, txt = srt_of(data, [line(0, 1, "旁白", None)], self.work)
        self.assertIn("旁白", body)
        self.assertNotIn("[", body)
        self.assertEqual(txt, "旁白")

    def test_empty_text_lines_are_dropped(self):
        data = sl.new_labels()
        segs = [line(0, 1, "   ", "SPEAKER_00"), line(2, 3, "有内容", "SPEAKER_00")]
        n = sl.rewrite_outputs(data, segs, self.work / "a.srt", self.work / "a.txt")
        self.assertEqual(n, 1)

    def test_rewrite_returns_line_count(self):
        data = sl.new_labels()
        segs = [line(0, 1, "a", "SPEAKER_00"), line(2, 3, "b", "SPEAKER_00")]
        n = sl.rewrite_outputs(data, segs, self.work / "a.srt", self.work / "a.txt")
        self.assertEqual(n, 2)

    def test_rewrite_does_not_double_prefix_already_labelled_text(self):
        """喂一句已经带 [艾丽丝] 的文本，不能再套一层变成 [[艾丽丝]]。

        双重前缀实际渲染成 `[艾丽丝] [艾丽丝] 你好`（中间有空格），所以
        `assertNotIn("[[", body)` 是永远成立的空断言。这里逐行断言：
        字幕文本行至多一个 `[...] ` 前缀。
        """
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        srt = self.work / "a.srt"
        txt = self.work / "a.txt"
        sl.rewrite_outputs(data, [line(0, 1, "[艾丽丝] 你好", "SPEAKER_00")], srt, txt)
        body = srt.read_text(encoding="utf-8")
        self.assertEqual(body.count("[艾丽丝]"), 1)

        text_lines = [ln for ln in body.splitlines()
                      if ln and not ln.isdigit() and "-->" not in ln]
        self.assertEqual(text_lines, ["[艾丽丝] 你好"])
        for ln in text_lines:
            self.assertRegex(ln, r"(?:\[[^\]]*\] )?[^\[]*",
                             f"这一行出现了不止一个说话人前缀：{ln}")
            self.assertEqual(len(re.findall(r"\[[^\]]*\]", ln)), 1, ln)
        self.assertEqual(txt.read_text(encoding="utf-8"), "[艾丽丝] 你好")

    def test_rewrite_outputs_persists_nothing(self):
        data = sl.new_labels()
        sl.set_names(data, {"SPEAKER_00": "艾丽丝"})
        self.assertFalse((self.work / "speakers.json").exists())
        sl.rewrite_outputs(data, [line(0, 1, "你好", "SPEAKER_00")],
                           self.work / "a.srt", self.work / "a.txt")
        self.assertFalse((self.work / "speakers.json").exists())

    def test_txt_skips_blank_text_like_srt(self):
        data = sl.new_labels()
        segs = [line(0, 1, "   ", "SPEAKER_00"), line(2, 3, "有内容", "SPEAKER_00")]
        n = sl.rewrite_outputs(data, segs, self.work / "a.srt", self.work / "a.txt")
        self.assertEqual(n, 1)
        self.assertEqual((self.work / "a.txt").read_text(encoding="utf-8"),
                         "[说话人1] 有内容")


if __name__ == "__main__":
    unittest.main(verbosity=2)
