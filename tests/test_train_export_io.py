#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Task 2：训练集落盘 / 命名净化 / GPT-SoVITS `.list` / 导出报告。

规则来自设计 §6.3（目录）、§8（命名净化）、§9（`.list` 三条硬约束）与计划
「Task 2 必须处理的接缝」10 条。

ffmpeg 边界（`train_export.cut_segment`）在逻辑用例里换成**写真实 WAV 的替身**，
所以这些用例不依赖 ffmpeg；`TestRealFfmpeg` 真跑 ffmpeg，并用 stdlib `wave` 读回校验。

测试使用系统临时目录。
"""

import copy
import json
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

import speaker_labels as sl  # noqa: E402
import train_export as te  # noqa: E402


TMP_ROOT = Path(tempfile.gettempdir())


def labels_with(mapping, overrides=None, work_name=""):
    data = sl.new_labels()
    sl.set_names(data, mapping or {})
    if work_name:
        sl.set_work_name(data, work_name)
    for idx, cluster in (overrides or {}).items():
        sl.set_override(data, idx, cluster)
    return data


def seg(start, end, text, speaker):
    return {"start": float(start), "end": float(end),
            "text": text, "speaker": speaker}


def fake_silence(*spans):
    """常量 silence_finder：只报完全落在查询区间内的那几段静音。"""
    spans = [tuple(map(float, s)) for s in spans]

    def finder(start, end):
        return [s for s in spans if s[0] >= start and s[1] <= end]

    return finder


def write_wav(path, seconds, sr=32000, amp=0.0, freq=440.0):
    """写一条真实可读的 WAV（amp=0.0 -> 数字静音）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(sr * float(seconds)))
    t = np.arange(n) / sr
    pcm = (np.sin(2 * np.pi * freq * t) * (amp * 32767)).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


def write_bursts(path, parts, sr=32000, amp=0.35, freq=440.0):
    """把 [(秒数, 有声?), ...] 拼成一条源音频，供真 ffmpeg 用例使用。"""
    chunks = []
    for seconds, loud in parts:
        n = int(round(sr * float(seconds)))
        t = np.arange(n) / sr
        a = amp if loud else 0.0
        chunks.append((np.sin(2 * np.pi * freq * t) * (a * 32767)).astype(np.int16))
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


def wav_info(path):
    with wave.open(str(path), "rb") as fh:
        return (fh.getnchannels(), fh.getsampwidth(), fh.getframerate(),
                fh.getnframes() / fh.getframerate())


def wav_peak_abs(path):
    with wave.open(str(path), "rb") as fh:
        data = fh.readframes(fh.getnframes())
    if not data:
        return 0
    samples = np.frombuffer(data, dtype="<i2")
    return int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0


def read_list_lines(path):
    """按 GPT-SoVITS 的读法取行：`f.read().strip("\\n").split("\\n")`。"""
    raw = Path(path).read_bytes()
    return raw.decode("utf-8").strip("\n").split("\n")


def make_fake_cut(out_sr=32000, fail_start=None, empty_start=None):
    """替身 cut_segment：写一条真实 WAV，长度就是请求的时长。"""

    def fake_cut(src, start, dur, dest, *, sr=te.DEFAULT_SAMPLE_RATE, trim=True,
                 trim_threshold_db=te.TRIM_THRESHOLD_DB, should_stop=None):
        if fail_start is not None and abs(float(start) - float(fail_start)) < 1e-6:
            raise te.ExportError("替身 ffmpeg 失败")
        dest = Path(dest)
        if empty_start is not None and abs(float(start) - float(empty_start)) < 1e-6:
            write_wav(dest, 0.0, sr=out_sr)
        else:
            write_wav(dest, dur, sr=out_sr)
        return dest

    return fake_cut


class ExportCase(unittest.TestCase):
    """每条用例一个独立临时目录（系统临时目录），收尾删干净。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="t2_", dir=str(TMP_ROOT)))
        self.src = self.dir / "vocals.wav"
        write_wav(self.src, 1.0)                 # 逻辑用例的源只要求「存在」
        self.out = self.dir / "out"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def use_fake_cutter(self, **kw):
        patcher = mock.patch.object(te, "cut_segment", make_fake_cut(**kw))
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def export(self, segments, labels, work_name="无职转生", **kw):
        kw.setdefault("silence_finder", fake_silence())
        return te.export_work(segments, labels, self.src, self.out,
                              work_name=work_name, **kw)

    @property
    def work(self):
        return self.out / "无职转生"


# ---------------------------------------------------------------------------
# 命名净化（设计 §8）
# ---------------------------------------------------------------------------

class TestTrainingFormats(ExportCase):
    def test_manifest_and_audio_mapping(self):
        self.use_fake_cutter()
        labels = labels_with({'SPEAKER_00': '艾莉丝'})
        labels['language'] = 'zh'
        segments = [seg(0, 3, '你好|朋友', 'SPEAKER_00'),
                    seg(4, 7, '未命名', 'SPEAKER_01')]
        for kind in ('txt', 'fish_speech', 'indextts'):
            with self.subTest(kind=kind):
                result = self.export(segments, labels, work_name=kind, format=kind)
                self.assertEqual(result['format'], kind)
                self.assertEqual(result['n_list'], 1)
                self.assertNotIn('list_path', result)
                manifest = Path(result['manifest_path'])
                lines = manifest.read_text(encoding='utf-8').splitlines()
                if kind == 'indextts':
                    record = json.loads(lines[0])
                    self.assertEqual(record['text'], '你好|朋友')
                    self.assertEqual(record['speaker'], '艾莉丝')
                    self.assertEqual(record['language'], 'zh')
                    audio = manifest.parent / record['audio']
                else:
                    row = lines[1].split('\t')
                    self.assertEqual(row[1:4], ['艾莉丝', 'zh', '你好|朋友'])
                    audio = manifest.parent / row[0]
                self.assertTrue(audio.is_file())
                if kind == 'fish_speech':
                    self.assertEqual(audio.with_suffix('.lab').read_text(encoding='utf-8').strip(), '你好|朋友')
                    self.assertEqual(list(Path(result['work_dir']).rglob('*.wav')), [audio])


class TestSanitize(unittest.TestCase):
    def test_filename_sanitises_windows_chars(self):
        self.assertEqual(te.sanitize_component('a<>:"/\\|?*b'), "a" + "_" * 9 + "b")
        self.assertEqual(te.sanitize_component("A?B"), "A_B")

    def test_fullwidth_question_mark_kept(self):
        self.assertEqual(te.sanitize_component("是啊？・…"), "是啊？・…")

    def test_control_chars_replaced(self):
        self.assertEqual(te.sanitize_component("a\x00b\x1fc\x7fd"), "a_b_c_d")
        self.assertEqual(te.sanitize_component("一\n二\r三\t四"), "一_二_三_四")

    def test_windows_device_names_are_prefixed(self):
        self.assertEqual(te.sanitize_component('NUL.txt'), '_NUL.txt')
        self.assertEqual(te.sanitize_component('com¹'), '_com¹')
        self.assertEqual(te.sanitize_component('auxiliary'), 'auxiliary')

    def test_trailing_dots_and_spaces_removed(self):
        self.assertEqual(te.sanitize_component("艾丽丝... "), "艾丽丝")
        self.assertEqual(te.sanitize_component("  艾丽丝  "), "艾丽丝")

    def test_empty_and_none(self):
        self.assertEqual(te.sanitize_component(None), "")
        self.assertEqual(te.sanitize_component("   "), "")
        self.assertEqual(te.sanitize_component("???"), "___")


class TestNaming(unittest.TestCase):
    def test_filename_truncates_text_to_40(self):
        stem = te.make_stem("艾丽丝", "あ" * 100)
        self.assertEqual(stem, "艾丽丝_" + "あ" * 40)
        self.assertLessEqual(len(stem), te.MAX_STEM_CHARS)

    def test_filename_truncates_whole_stem_to_80(self):
        stem = te.make_stem("角" * 60, "あ" * 40)
        self.assertEqual(len(stem), te.MAX_STEM_CHARS)
        self.assertTrue(stem.startswith("角" * 60 + "_"))
        # 角色名自己就超长时也要收敛到 80
        self.assertEqual(len(te.make_stem("角" * 200, "あ" * 40)), te.MAX_STEM_CHARS)

    def test_suffix_is_reserved_inside_the_limit(self):
        stem = te.make_stem("角" * 60, "あ" * 40, suffix="_2")
        self.assertEqual(len(stem), te.MAX_STEM_CHARS)
        self.assertTrue(stem.endswith("_2"))

    def test_fullwidth_kept_in_filename(self):
        self.assertEqual(te.make_stem("艾丽丝", "是啊？・…"), "艾丽丝_是啊？・…")
        self.assertEqual(te.make_stem("艾丽丝", "半角?问号"), "艾丽丝_半角_问号")

    def test_empty_text_gives_role_only(self):
        self.assertEqual(te.make_stem("艾丽丝", "   "), "艾丽丝")
        self.assertEqual(te.make_stem("艾丽丝", ""), "艾丽丝")
        self.assertEqual(te.make_stem("艾丽丝", None), "艾丽丝")

    def test_whitespace_collapsed_in_filename(self):
        self.assertEqual(te.make_stem("艾丽丝", "是啊  鲁迪\n也不知道"),
                         "艾丽丝_是啊 鲁迪 也不知道")


class TestAssignNames(unittest.TestCase):
    def test_duplicate_stems_get_suffix(self):
        entries = [{"dir": "D", "name_part": "艾丽丝", "text": "うん",
                    "index": 0, "start": 0.0, "end": 3.0},
                   {"dir": "D", "name_part": "艾丽丝", "text": "うん",
                    "index": 1, "start": 10.0, "end": 13.0},
                   {"dir": "D", "name_part": "艾丽丝", "text": "うん",
                    "index": 2, "start": 20.0, "end": 23.0}]

        te.assign_names(entries)

        self.assertEqual([e["name"] for e in entries],
                         ["艾丽丝_うん.wav", "艾丽丝_うん_2.wav", "艾丽丝_うん_3.wav"])

    def test_same_index_different_start_gets_distinct_names(self):
        """接缝第 1 条：拆长后多段共用同一 index，名字必须按 (index, start) 区别开。"""
        entries = [{"dir": "艾丽丝", "name_part": "艾丽丝", "text": "y",
                    "index": 0, "start": 30.0, "end": 40.0},
                   {"dir": "艾丽丝", "name_part": "艾丽丝", "text": "y",
                    "index": 0, "start": 0.0, "end": 10.0}]

        te.assign_names(entries)

        self.assertEqual(len({e["name"] for e in entries}), 2)
        by_start = sorted(entries, key=lambda e: e["start"])
        self.assertEqual(by_start[0]["name"], "艾丽丝_y.wav")
        self.assertEqual(by_start[1]["name"], "艾丽丝_y_2.wav")

    def test_different_dirs_do_not_collide(self):
        entries = [{"dir": "艾丽丝", "name_part": "艾丽丝", "text": "うん",
                    "index": 0, "start": 0.0, "end": 3.0},
                   {"dir": "_短句", "name_part": "艾丽丝", "text": "うん",
                    "index": 1, "start": 0.0, "end": 3.0}]

        te.assign_names(entries)

        self.assertEqual([e["name"] for e in entries],
                         ["艾丽丝_うん.wav", "艾丽丝_うん.wav"])

    def test_windows_case_insensitive_collision(self):
        """Windows 文件名不分大小写，Al_x 与 AL_x 是同一个文件。"""
        entries = [{"dir": "D", "name_part": "Al", "text": "x",
                    "index": 0, "start": 0.0, "end": 3.0},
                   {"dir": "D", "name_part": "AL", "text": "x",
                    "index": 1, "start": 5.0, "end": 8.0}]

        te.assign_names(entries)

        self.assertEqual(len({e["name"].casefold() for e in entries}), 2)


# ---------------------------------------------------------------------------
# 静音检测（ffmpeg silencedetect 的解析）
# ---------------------------------------------------------------------------

SILENCEDETECT_LOG = """\
ffmpeg version 8.1.2 Copyright (c) 2000-2025 the FFmpeg developers
Input #0, wav, from 'in.wav':
  Duration: 00:00:10.00, bitrate: 512 kb/s
[silencedetect @ 0x55d1] silence_start: 2.5
[silencedetect @ 0x55d1] silence_end: 4.0 | silence_duration: 1.5
[silencedetect @ 0x55d1] silence_start: 7.25
[silencedetect @ 0x55d1] silence_end: 8 | silence_duration: 0.75
"""


class TestSilenceParse(unittest.TestCase):
    def test_parses_pairs_in_time_order(self):
        self.assertEqual(te.parse_silencedetect(SILENCEDETECT_LOG),
                         [(2.5, 4.0), (7.25, 8.0)])

    def test_unpaired_start_is_dropped(self):
        log = "[silencedetect @ 0x1] silence_start: 1.0\n"
        self.assertEqual(te.parse_silencedetect(log), [])

    def test_unpaired_end_is_dropped(self):
        log = "[silencedetect @ 0x1] silence_end: 3.0 | silence_duration: 1.0\n"
        self.assertEqual(te.parse_silencedetect(log), [])

    def test_out_of_order_input_is_sorted(self):
        log = ("[x] silence_start: 9.0\n[x] silence_end: 9.5\n"
               "[x] silence_start: 1.0\n[x] silence_end: 1.5\n")
        self.assertEqual(te.parse_silencedetect(log), [(1.0, 1.5), (9.0, 9.5)])

    def test_plain_log_has_no_spans(self):
        self.assertEqual(te.parse_silencedetect(""), [])
        self.assertEqual(te.parse_silencedetect("Duration: 00:00:05.00\n"), [])
        self.assertEqual(te.parse_silencedetect(None), [])


# ---------------------------------------------------------------------------
# `.list` 单行格式（设计 §9 三条硬约束）
# ---------------------------------------------------------------------------

class TestListLine(unittest.TestCase):
    def test_text_with_pipe_is_replaced(self):
        line = te.list_line(r"C:\out\艾丽丝\艾丽丝_x.wav", "艾丽丝", "前|后|中")
        parts = line.split("|")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[3], "前｜后｜中")
        self.assertEqual(parts[2], "JA")

    def test_newlines_in_text_are_neutralised(self):
        line = te.list_line("w.wav", "艾丽丝", "a\r\nb\rc\nd")
        self.assertEqual(len(line.split("|")), 4)
        self.assertNotIn("\r", line)
        self.assertNotIn("\n", line)

    def test_speaker_field_is_the_raw_role_name(self):
        self.assertEqual(te.list_line("w.wav", "艾丽丝", "台词").split("|")[1], "艾丽丝")

    def test_pipe_in_speaker_is_escaped(self):
        line = te.list_line("w.wav", "艾|丽", "台词")
        self.assertEqual(len(line.split("|")), 4)
        self.assertEqual(line.split("|")[1], "艾｜丽")


# ---------------------------------------------------------------------------
# 落盘与报告（替身 ffmpeg）
# ---------------------------------------------------------------------------

class TestExportList(ExportCase):
    def test_list_parses_with_gpt_sovits_rule(self):
        """关键验收：按 GPT-SoVITS 自己的解析方式逐行验证生成的 `.list`。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"})
        segs = [seg(0.0, 5.0, "是啊，鲁迪也不知道呢", "SPEAKER_00"),
                seg(6.0, 11.0, "我才没有那种兴趣", "SPEAKER_01"),
                seg(12.0, 17.0, "带|竖线|的台词", "SPEAKER_00"),
                seg(18.0, 23.0, "换\n行\r台词", "SPEAKER_01")]

        res = self.export(segs, labels)

        list_path = Path(res["list"])
        raw = list_path.read_bytes()
        # 显式断言文件字节里没有 \r（Path.write_text 在 Windows 会写出 CRLF）
        self.assertNotIn(b"\r", raw)
        lines = raw.decode("utf-8").strip("\n").split("\n")
        self.assertEqual(len(lines), 4)
        for line in lines:
            wav_name, spk_name, language, text = line.split("|")   # 必须恰好 4 段
            self.assertTrue(Path(wav_name).is_file(), wav_name)
            self.assertNotIn("|", text)
            self.assertNotIn("\r", text)
            self.assertNotIn("\n", text)
            self.assertEqual(language, "JA")
            self.assertTrue(spk_name)
        self.assertEqual([ln.split("|")[1] for ln in lines],
                         ["艾丽丝", "鲁迪乌斯", "艾丽丝", "鲁迪乌斯"])
        self.assertEqual([ln.split("|")[3] for ln in lines],
                         ["是啊，鲁迪也不知道呢", "我才没有那种兴趣",
                          "带｜竖线｜的台词", "换 行 台词"])
        self.assertEqual(res["n_list"], 4)

    def test_text_with_pipe_is_replaced(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "前|后", "SPEAKER_00")]

        res = self.export(segs, labels)

        lines = read_list_lines(res["list"])
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(lines[0].split("|")), 4)
        self.assertEqual(lines[0].split("|")[3], "前｜后")
        self.assertTrue(lines[0].split("|")[0].endswith("前_后.wav"))

    def test_list_is_lf_and_ends_with_newline(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels)

        raw = Path(res["list"]).read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.endswith(b"\n"))

    def test_list_has_exactly_four_fields_or_write_fails(self):
        """写 `.list` 时自校验：任何一行不是 4 段就必须报错，而不是悄悄写坏文件。"""
        path = Path(self.dir) / "bad.list"
        with self.assertRaises(te.ExportError):
            te.write_list(path, [("w|w.wav", "艾丽丝", "台词")])


class TestExportBuckets(ExportCase):
    def test_short_and_unnamed_not_in_list(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})      # SPEAKER_01 没名字
        segs = [seg(0.0, 5.0, "正常台词", "SPEAKER_00"),
                seg(6.0, 7.5, "嗯", "SPEAKER_00"),
                seg(10.0, 15.0, "没名字", "SPEAKER_01")]

        res = self.export(segs, labels)

        lines = read_list_lines(res["list"])
        self.assertEqual(len(lines), 1)
        self.assertIn("正常台词", lines[0])
        self.assertEqual(res["n_short"], 1)
        self.assertEqual(res["n_unknown"], 1)
        self.assertEqual(sorted(p.name for p in (self.work / "_短句").glob("*.wav")),
                         ["艾丽丝_嗯.wav"])
        self.assertEqual(
            sorted(p.name for p in (self.work / "_未命名" / "说话人2").glob("*.wav")),
            ["说话人2_没名字.wav"])

    def test_unnamed_clips_land_in_the_bucket_and_never_in_the_list(self):
        """未命名聚类照常落盘，但一行都不进 `.list`（设计 §9）。

        这里必须**同时**放一条已命名的片段：一个片段都进不了 `.list` 时整趟导出会被
        拒绝（见 `TestEmptyListIsRefused`），只放未命名片段就测不到「照样落盘」这件事。
        """
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})      # SPEAKER_01 没名字

        res = self.export([seg(0.0, 5.0, "有名字", "SPEAKER_00"),
                           seg(6.0, 11.0, "没名字", "SPEAKER_01")], labels)

        self.assertEqual(res["n_list"], 1)
        self.assertEqual(res["n_unknown"], 1)
        self.assertEqual(res["n_written"], 2)
        self.assertTrue((self.work / "_未命名" / "说话人2" / "说话人2_没名字.wav").is_file())
        lines = [ln for ln in read_list_lines(res["list"]) if ln]
        self.assertEqual(len(lines), 1)
        self.assertIn("有名字", lines[0])
        self.assertNotIn("没名字", lines[0])

    def test_empty_list_refusal_leaves_nothing_behind(self):
        """一个片段都进不了 `.list` 时整趟导出被拒绝：不写 0 字节 `.list`，
        连报告和目录都不留 —— 半个产物比没有产物更难解释。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})      # SPEAKER_01 没名字

        with self.assertRaises(te.ExportError):
            self.export([seg(0.0, 5.0, "没名字", "SPEAKER_01")], labels)

        self.assertFalse((self.out / ("无职转生" + te.LIST_SUFFIX)).exists())
        self.assertFalse((self.out / ("无职转生" + te.REPORT_SUFFIX)).exists())
        self.assertFalse(self.work.exists())

    def test_source_warning_survives_a_mixed_export(self):
        """`source_kind` 的警告不能被别的东西顶掉。

        这里写死文案而不是 `te.source_warning("raw16k")`：与
        `test_source_warning_texts_are_pinned_verbatim` 同一个理由 ——
        两边同源的断言，改坏了文案也照样绿。
        """
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        res = self.export([seg(0.0, 5.0, "有名字", "SPEAKER_00"),
                           seg(6.0, 11.0, "没名字", "SPEAKER_01")], labels,
                          source_kind="raw16k")

        self.assertIn("未分离，直接用 16 kHz 中间产物：高频已丢失且带 BGM",
                      res["warning"])

    def test_normal_export_has_no_warning(self):
        """非空 `.list` + 高质量源 = 没有警告（保持既有语义）。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels)

        self.assertIsNone(res["warning"])

    def test_reserved_role_names_fall_back_to_unnamed(self):
        """角色名撞上保留 bucket 名（`_短句` / `_未命名`）时必须当没起名处理。

        否则 `_ClipBuilder.build` 里 `bucket = self.speaker` 恰好等于 `SHORT_BUCKET`，
        片段被标 `short=True`、落 `_短句\\`、又被 `is_exportable` 挡在 `.list` 外 ——
        用户看到的是「写了 1 条、list 0 条、原因 short」。
        """
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": te.SHORT_BUCKET,
                              "SPEAKER_01": te.UNNAMED_BUCKET,
                              # 第三条正常命名：全都没名字时整趟导出会被拒绝，
                              # 那样就验不了「保留名回退成未命名」这段落盘行为
                              "SPEAKER_02": "艾丽丝"})
        segs = [seg(0.0, 5.0, "甲", "SPEAKER_00"),
                seg(6.0, 11.0, "乙", "SPEAKER_01"),
                seg(12.0, 17.0, "丙", "SPEAKER_02")]

        res = self.export(segs, labels)

        self.assertEqual(te._speaker_of(labels, 0, segs[0]), te.UNNAMED_BUCKET)
        self.assertEqual(te._speaker_of(labels, 1, segs[1]), te.UNNAMED_BUCKET)
        self.assertEqual(res["report"]["short_clips"], 0)
        self.assertEqual(res["n_list"], 1)
        self.assertEqual(res["n_written"], 3)
        self.assertFalse((self.work / te.SHORT_BUCKET).exists())
        self.assertEqual(
            sorted(p.name for p in (self.work / te.UNNAMED_BUCKET / "说话人1").glob("*.wav")),
            ["说话人1_甲.wav"])
        self.assertEqual(
            sorted(p.name for p in (self.work / te.UNNAMED_BUCKET / "说话人2").glob("*.wav")),
            ["说话人2_乙.wav"])
        self.assertEqual(sorted(r["reason"] for r in res["report"]["not_in_list"]),
                         ["unnamed", "unnamed"])

    def test_unnamed_dir_uses_cluster_label(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "未知三", "SPEAKER_02"),
                seg(10.0, 15.0, "问号", "?"),
                # 前两条都是未命名：必须再放一条已命名的，否则整趟导出被拒绝
                seg(20.0, 25.0, "正常", "SPEAKER_00")]

        self.export(segs, labels)

        self.assertTrue((self.work / "_未命名" / "说话人3").is_dir())
        self.assertTrue((self.work / "_未命名" / "未知").is_dir())

    def test_drop_short_skips_writing_short_clips(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "正常", "SPEAKER_00"),
                seg(6.0, 7.5, "嗯", "SPEAKER_00")]

        res = self.export(segs, labels, drop_short=True)

        self.assertFalse((self.work / "_短句").exists())
        self.assertEqual(res["n_short"], 1)
        self.assertEqual(res["n_written"], 1)
        reasons = [s["reason"] for s in res["report"]["skipped"]]
        self.assertEqual(reasons, ["short"])


class TestExportNaming(ExportCase):
    def test_same_index_split_pieces_get_distinct_names(self):
        """接缝第 1 条：40 s 句切 4 段，index 与文本全都一样，文件名必须各不相同。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 40.0, "yyyy", "SPEAKER_00")]
        finder = fake_silence((9.8, 10.2), (19.8, 20.2), (29.8, 30.2))

        res = te.export_work(segs, labels, self.src, self.out,
                             work_name="无职转生", silence_finder=finder)

        clips = te.plan_clips(segs, labels, finder)
        self.assertEqual({c["index"] for c in clips}, {0})
        self.assertEqual([c["text"] for c in clips], ["y", "y", "y", "y"])
        self.assertEqual(
            sorted(p.name for p in (self.work / "艾丽丝").glob("*.wav")),
            ["艾丽丝_y.wav", "艾丽丝_y_2.wav", "艾丽丝_y_3.wav", "艾丽丝_y_4.wav"])
        lines = read_list_lines(res["list"])
        self.assertEqual(len(lines), 4)
        self.assertEqual(len({ln.split("|")[0] for ln in lines}), 4)

    def test_duplicate_text_same_role_gets_suffix(self):
        """接缝第 4 条第二种来源：同角色不同句但台词相同（两声「うん」）。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "うん", "SPEAKER_00"),
                seg(10.0, 13.0, "うん", "SPEAKER_00")]

        res = self.export(segs, labels)

        self.assertEqual(sorted(p.name for p in (self.work / "艾丽丝").glob("*.wav")),
                         ["艾丽丝_うん.wav", "艾丽丝_うん_2.wav"])
        lines = read_list_lines(res["list"])
        self.assertEqual(len(lines), 2)
        self.assertEqual(len({ln.split("|")[0] for ln in lines}), 2)
        for line in lines:
            self.assertTrue(Path(line.split("|")[0]).is_file())


class TestExportStructure(ExportCase):
    def test_reexport_removes_stale_role_dirs(self):
        """设计 §9：重复导出覆盖 `.list`/报告，音频按角色目录重建，不残留旧目录。"""
        self.use_fake_cutter()
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00")]

        first = self.export(segs, labels_with({"SPEAKER_00": "艾丽丝"}))
        self.assertTrue((self.work / "艾丽丝").is_dir())

        second = self.export(segs, labels_with({"SPEAKER_00": "鲁迪乌斯"}))

        self.assertFalse((self.work / "艾丽丝").exists(), "旧角色目录必须被清掉")
        self.assertTrue((self.work / "鲁迪乌斯").is_dir())
        self.assertEqual(len(read_list_lines(second["list"])), 1)
        self.assertIn("鲁迪乌斯", Path(second["list"]).read_text(encoding="utf-8"))
        self.assertNotIn("艾丽丝", Path(second["list"]).read_text(encoding="utf-8"))
        self.assertEqual(first["list"], second["list"])
        # 报告被覆盖，不是追加
        report = json.loads(Path(second["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(report["work"], "无职转生")
        self.assertEqual(report["previous_clip_count"], 1)
        self.assertEqual(report["previous"]["list_lines"], 1)

    def test_report_records_overwrite_counts(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "一", "SPEAKER_00"),
                seg(6.0, 11.0, "二", "SPEAKER_00")]

        self.export(segs, labels)
        again = self.export(segs, labels)

        self.assertEqual(again["report"]["previous_clip_count"], 2)
        self.assertEqual(again["report"]["previous"]["list_lines"], 2)
        self.assertTrue(again["report"]["previous"]["report_exists"])

    def test_first_export_has_zero_previous(self):
        self.use_fake_cutter()
        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                          labels_with({"SPEAKER_00": "艾丽丝"}))
        self.assertEqual(res["report"]["previous_clip_count"], 0)
        self.assertEqual(res["report"]["previous"]["list_lines"], 0)
        self.assertFalse(res["report"]["previous"]["report_exists"])

    def test_stale_export_survives_a_failed_export(self):
        """音源丢失时不能先把上一次的成果删掉。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels)
        before = sorted(p.name for p in (self.work / "艾丽丝").glob("*.wav"))

        with self.assertRaises(te.ExportError):
            te.export_work([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels,
                           self.dir / "没有这个文件.wav", self.out,
                           work_name="无职转生", silence_finder=fake_silence())

        self.assertEqual(sorted(p.name for p in (self.work / "艾丽丝").glob("*.wav")),
                         before)
        self.assertTrue((self.out / "无职转生.list").is_file())

    def test_stale_export_survives_a_disk_precheck_failure(self):
        """磁盘预检失败同样不许动上一次的产物（删旧 `.list` 必须排在预检之后）。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels)
        before = sorted(p.name for p in (self.work / "艾丽丝").glob("*.wav"))

        with self.assertRaises(te.InsufficientSpace):
            self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels,
                        free_bytes=1024)

        self.assertEqual(sorted(p.name for p in (self.work / "艾丽丝").glob("*.wav")),
                         before)
        self.assertTrue((self.out / "无职转生.list").is_file())
        self.assertTrue((self.out / "无职转生.导出报告.json").is_file())

    def test_no_temp_files_left_behind(self):
        self.use_fake_cutter()
        self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                    labels_with({"SPEAKER_00": "艾丽丝"}))
        self.assertEqual([p.name for p in self.out.rglob("*") if p.suffix == ".tmp"], [])

    def test_work_name_from_labels_is_used(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"}, work_name="无职转生")

        res = te.export_work([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels,
                             self.src, self.out, silence_finder=fake_silence())

        self.assertEqual(Path(res["list"]).name, "无职转生.list")
        self.assertTrue((self.work / "艾丽丝").is_dir())


class TestExportInterrupt(ExportCase):
    """F1 不变量：`.list` 的存在必须与「音频目录被重建」同生共死。

    只要 `.list` 在盘上，它每一行指向的 wav 就必须都存在 —— 否则目录看起来是
    一次成功导出，实际是「旧 `.list` + 新（空/半）音频目录」，GPT-SoVITS 的
    `1-get-text.py` 打开这些路径会直接炸在建库中途。
    """

    def test_list_never_points_at_deleted_wavs_after_cancel(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "一", "SPEAKER_00"),
                seg(6.0, 11.0, "二", "SPEAKER_00"),
                seg(12.0, 17.0, "三", "SPEAKER_00")]
        first = self.export(segs, labels)
        list_path = Path(first["list"])
        report_path = Path(first["report_path"])
        self.assertEqual(len(read_list_lines(list_path)), 3)

        calls = []

        def stop():
            calls.append(1)
            # 预检 1 次 + 第 1 条片段 1 次 -> 第 2 条片段之前停
            return len(calls) >= 3

        with self.assertRaises(te.ExportCancelled):
            self.export(segs, labels, should_stop=stop)

        # 中断的那次只切出 1 条 wav（音频目录已经被重建过）
        self.assertEqual(len(list(self.work.rglob("*.wav"))), 1)
        if list_path.exists():
            for line in read_list_lines(list_path):
                if not line.strip():
                    continue
                wav = line.split("|")[0]
                self.assertTrue(Path(wav).is_file(),
                                "`.list` 指向了已被删掉的 wav：%s" % wav)
        self.assertFalse(report_path.exists(), "中断后旧报告必须已经不在了")

    def test_stale_list_is_removed_before_the_audio_dir_is_rebuilt(self):
        """目录被重建 + `.list` 还在 = 假成品：重建之前就必须把它删掉。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "一", "SPEAKER_00")]

        first = self.export(segs, labels)
        list_path = Path(first["list"])
        self.assertTrue(list_path.is_file())
        self.assertTrue((self.work / "艾丽丝").is_dir())

        calls = []

        def stop():
            calls.append(1)
            # 预检放过，第 1 条片段之前停：此刻音频目录已经被整棵重建过了
            return len(calls) >= 2

        with self.assertRaises(te.ExportCancelled):
            self.export(segs, labels, should_stop=stop)

        self.assertFalse(list_path.exists())
        self.assertFalse(Path(first["report_path"]).exists())
        self.assertEqual(list(self.work.rglob("*.wav")), [])


class TestExportReport(ExportCase):
    def test_report_counts_match_reality(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"})
        segs = [seg(0.0, 5.0, "一", "SPEAKER_00"),
                seg(6.0, 7.5, "短句", "SPEAKER_00"),
                seg(9.0, 14.0, "二", "SPEAKER_01"),
                seg(15.0, 20.0, "三", "SPEAKER_02"),
                # 25.2 s 处切一刀 -> 一个空文本碎片（跳过）+ 一条 13.8 s（超长）
                seg(25.0, 39.0, "あ" * 14, "SPEAKER_00")]

        res = te.export_work(segs, labels, self.src, self.out, work_name="无职转生",
                             silence_finder=fake_silence((25.15, 25.25)))
        rep = res["report"]
        files = list(self.work.rglob("*.wav"))
        lines = read_list_lines(res["list"])

        self.assertEqual(rep["total_clips"], 6)
        self.assertEqual(rep["written_clips"], len(files))
        self.assertEqual(rep["written_clips"], 5)          # 有一条空文本碎片没落盘
        self.assertEqual(rep["skipped_clips"], 1)
        self.assertEqual(rep["empty_text_skipped"], 1)
        self.assertEqual(rep["over_limit_clips"], 1)
        self.assertEqual(rep["total_clips"],
                         rep["named_clips"] + rep["short_clips"] + rep["unnamed_clips"])
        self.assertEqual(rep["total_clips"],
                         rep["written_clips"] + rep["skipped_clips"])
        self.assertEqual(rep["list_clips"], len(lines))
        self.assertEqual(rep["skipped_clips"], len(rep["skipped"]))
        self.assertEqual(rep["short_clips"], 2)
        self.assertEqual(rep["unnamed_clips"], 1)
        self.assertEqual(rep["named_clips"], 3)
        self.assertEqual(rep["list_clips"], 3)
        self.assertEqual(sum(rep["dirs"].values()), rep["written_clips"])
        self.assertEqual(rep["sample_rate"], 32000)
        self.assertEqual(rep["source_kind"], "separated_hq")
        self.assertEqual(rep["channels"], 1)
        self.assertEqual(rep["sample_width_bytes"], 2)
        # 时长与体积要和磁盘上的文件对得上
        real_seconds = sum(wav_info(self.work / m["path"])[3]
                           for m in rep["written"])
        self.assertAlmostEqual(rep["total_seconds"], real_seconds, places=3)
        self.assertEqual(rep["actual_bytes"], sum(p.stat().st_size for p in files))
        self.assertEqual(rep["estimated_bytes"],
                         int(round(rep["planned_seconds"] * 32000 * 2)))
        self.assertEqual(rep["list_seconds"],
                         round(sum(wav_info(self.work / m["path"])[3]
                                   for m in rep["written"] if m["in_list"]), 3))
        # 每条落盘片段都记了目录与路径
        for rec in rep["written"]:
            self.assertTrue((self.work / rec["path"]).is_file())
            self.assertIn(rec["dir"], rep["dirs"])

    def test_over_limit_and_empty_text_are_recorded(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        # 0.2 s 处有静音 -> 切出 [0,0.2] 的空文本碎片；剩下 13.8 s 仍超 12 s
        segs = [seg(0.0, 14.0, "あ" * 14, "SPEAKER_00")]
        finder = fake_silence((0.15, 0.25))

        res = te.export_work(segs, labels, self.src, self.out,
                             work_name="无职转生", silence_finder=finder)
        rep = res["report"]
        clips = te.plan_clips(segs, labels, finder)

        self.assertTrue(any(not c["text"].strip() for c in clips), clips)
        self.assertEqual(rep["empty_text_skipped"], 1)
        self.assertEqual(rep["over_limit_clips"], 1)
        self.assertEqual([s["reason"] for s in rep["skipped"]], ["empty_text"])
        self.assertEqual(rep["total_clips"], 2)
        self.assertEqual(rep["written_clips"], 1)

    def test_empty_text_clips_are_not_written_anywhere(self):
        """接缝第 5 条：空文本片段一律不落盘，任何 bucket 里都不能留 `角色_.wav`。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 14.0, "あ" * 14, "SPEAKER_00")]

        res = te.export_work(segs, labels, self.src, self.out,
                             work_name="无职转生",
                             silence_finder=fake_silence((0.15, 0.25)))

        self.assertEqual(res["report"]["empty_text_skipped"], 1)
        self.assertFalse((self.work / "_短句").exists())
        written = list(self.work.rglob("*.wav"))
        self.assertEqual(len(written), 1)
        for p in written:
            self.assertFalse(p.stem.endswith("_"), p.name)
            self.assertGreater(wav_info(p)[3], 2.0)

    def test_ffmpeg_error_is_recorded_and_does_not_abort(self):
        self.use_fake_cutter(fail_start=6.0)
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "一", "SPEAKER_00"),
                seg(6.0, 11.0, "二", "SPEAKER_00")]

        res = self.export(segs, labels)

        rep = res["report"]
        self.assertEqual(rep["written_clips"], 1)
        self.assertEqual([s["reason"] for s in rep["skipped"]], ["ffmpeg_error"])
        self.assertEqual(len(rep["ffmpeg_errors"]), 1)
        self.assertEqual(len(read_list_lines(res["list"])), 1)

    def test_empty_after_trim_is_recorded(self):
        self.use_fake_cutter(empty_start=0.0)
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        # 第二条正常：全都被裁空时整趟导出会被拒绝，那样就读不到 skipped 原因了
        res = self.export([seg(0.0, 5.0, "全静音", "SPEAKER_00"),
                           seg(6.0, 11.0, "正常", "SPEAKER_00")], labels)

        rep = res["report"]
        self.assertEqual([s["reason"] for s in rep["skipped"]], ["empty_after_trim"])
        lines = [ln for ln in read_list_lines(res["list"]) if ln]
        self.assertEqual(len(lines), 1)
        self.assertIn("正常", lines[0])

    def test_not_in_list_reasons_cover_short_and_unnamed(self):
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "正常", "SPEAKER_00"),
                seg(6.0, 7.5, "短", "SPEAKER_00"),
                seg(9.0, 14.0, "没名字", "SPEAKER_01")]

        res = self.export(segs, labels)

        reasons = sorted(r["reason"] for r in res["report"]["not_in_list"])
        self.assertEqual(reasons, ["short", "unnamed"])

    def test_unnamed_short_clip_reason_is_unnamed(self):
        """1.5 s 的**未命名**片段 reason 必须是 `unnamed`，不能写成 `short` ——
        写 `short` 会让人以为它有角色名、只是太短。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "正常", "SPEAKER_00"),
                seg(6.0, 7.5, "啊", "SPEAKER_01")]

        res = self.export(segs, labels)

        recs = [r for r in res["report"]["not_in_list"]
                if r["dir"].startswith(te.UNNAMED_BUCKET)]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["reason"], "unnamed")

    def test_short_skipped_counts_what_drop_short_actually_dropped(self):
        """`drop_short=True` 时用户看到的「丢掉 N 条」必须和 `skipped_clips` 对齐。

        未命名的 1.5 s 片段也会被丢掉，但它路由到 `_未命名\\`（`short_clips`
        按 `_短句\\` 桶统计，只数有角色名的那条），所以必须有独立的 `short_skipped`。
        """
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})      # SPEAKER_01 没名字
        segs = [seg(0.0, 5.0, "正常", "SPEAKER_00"),
                seg(6.0, 7.5, "嗯", "SPEAKER_00"),
                seg(9.0, 10.5, "啊", "SPEAKER_01")]

        res = self.export(segs, labels, drop_short=True)

        rep = res["report"]
        self.assertEqual(rep["short_skipped"], 2)
        self.assertEqual([s["reason"] for s in rep["skipped"]], ["short", "short"])
        self.assertEqual(rep["skipped_clips"], 2)
        self.assertEqual(rep["short_clips"], 1)     # 只有有角色名的那条进 `_短句` 桶
        self.assertFalse((self.work / te.SHORT_BUCKET).exists())
        self.assertEqual(res["n_written"], 1)

    def test_short_clips_counts_the_planned_bucket_not_the_disk(self):
        """`short_clips` 是「计划里路由到 `_短句\\` 的条数」，与是否真落盘无关。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "正常", "SPEAKER_00"),
                seg(6.0, 7.5, "嗯", "SPEAKER_00")]

        res = self.export(segs, labels, drop_short=True)

        self.assertFalse((self.work / te.SHORT_BUCKET).exists())   # 一条都没落盘
        self.assertEqual(res["report"]["short_clips"], 1)          # 计划里还是 1 条
        self.assertEqual(res["report"]["short_skipped"], 1)


class TestCleanupPartialExport(ExportCase):
    """P6：被中止的导出能安全清掉**自己**的半截产物，且只清自己那一棵。

    `export_work` 是「先删旧 `.list`/旧报告，再整棵重建 `<作品名>\\`」，
    所以中断时盘上会留下没有 `.list` 指向的半截 wav。删掉它们是对的
    （否则日志说「已作废」而文件还在，用户迟早会被误导），但边界必须钉死。
    """

    def _partial(self, name="无职转生", role="艾丽丝"):
        d = self.out / name / role
        d.mkdir(parents=True, exist_ok=True)
        wav = d / "艾丽丝_第一句.wav"
        write_wav(wav, 0.5, amp=0.3)
        return d, wav

    def test_removes_its_own_half_written_dir(self):
        d, _ = self._partial()
        info = te.cleanup_partial_export(self.out, "无职转生")
        self.assertTrue(info["removed"], info)
        self.assertFalse(d.exists())
        self.assertGreater(info["freed_mb"], 0.0)

    def test_keeps_the_dir_while_its_list_still_points_at_it(self):
        """`.list` 还在 = 这棵目录可能仍被引用（上一次成功导出 / 本次已写完）。"""
        d, wav = self._partial()
        (self.out / "无职转生.list").write_text("x|y|JA|z\n", encoding="utf-8",
                                                newline="\n")
        info = te.cleanup_partial_export(self.out, "无职转生")
        self.assertFalse(info["removed"], info)
        self.assertTrue(d.exists())
        self.assertTrue(wav.exists(), "`.list` 指向的 wav 被删了")
        self.assertIn(".list", info["reason"])

    def test_never_touches_another_works_dir(self):
        mine, _ = self._partial("无职转生")
        other, other_wav = self._partial("别的作品")
        info = te.cleanup_partial_export(self.out, "无职转生")
        self.assertTrue(info["removed"], info)
        self.assertFalse(mine.exists())
        self.assertTrue(other.exists(), "删到了别的作品的目录")
        self.assertTrue(other_wav.exists())

    def test_refuses_paths_outside_the_export_root(self):
        """`..` / 盘符根都删不到东西（与 `_rebuild_work_dir` 同一判据）。"""
        outside = self.dir / "外面"
        outside.mkdir(parents=True, exist_ok=True)
        (outside / "x.wav").write_text("x", encoding="utf-8")
        for bad in ("..", "../外面", "", "."):
            with self.subTest(work_name=bad):
                info = te.cleanup_partial_export(self.out, bad)
                self.assertFalse(info["removed"], info)
        self.assertTrue(outside.exists())

    def test_missing_dir_is_a_noop(self):
        info = te.cleanup_partial_export(self.out, "从来没导出过")
        self.assertFalse(info["removed"], info)

    def test_empty_work_name_is_refused(self):
        info = te.cleanup_partial_export(self.out, "")
        self.assertFalse(info["removed"], info)
        self.assertTrue(info["reason"])


class TestExportGuards(ExportCase):
    def test_export_does_not_mutate_inputs(self):
        """接缝第 9 条：落盘流程同样不得改动入参。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"})
        segs = [seg(0.0, 14.0, "长句", "SPEAKER_00"),
                seg(20.0, 25.0, "没名字", "SPEAKER_01")]
        segs_before = copy.deepcopy(segs)
        labels_before = copy.deepcopy(labels)

        te.export_work(segs, labels, self.src, self.out, work_name="无职转生",
                       silence_finder=fake_silence((6.8, 7.2)))

        self.assertEqual(segs, segs_before)
        self.assertEqual(labels, labels_before)

    def test_empty_work_name_is_refused(self):
        self.use_fake_cutter()
        with self.assertRaises(te.ExportError):
            te.export_work([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                           labels_with({"SPEAKER_00": "艾丽丝"}),
                           self.src, self.out, work_name="   ",
                           silence_finder=fake_silence())
        self.assertFalse(self.out.exists())

    def test_missing_source_is_refused(self):
        with self.assertRaises(te.ExportError):
            te.export_work([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                           labels_with({"SPEAKER_00": "艾丽丝"}),
                           self.dir / "nope.wav", self.out, work_name="无职转生")

    def test_bad_sample_rate_is_refused(self):
        self.use_fake_cutter()
        for bad in (22050, 8000, 0, -1):
            with self.assertRaises(te.ExportError):
                self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                            labels_with({"SPEAKER_00": "艾丽丝"}), sr=bad)

    def test_44100_is_allowed(self):
        self.use_fake_cutter(out_sr=44100)
        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                          labels_with({"SPEAKER_00": "艾丽丝"}), sr=44100)
        self.assertEqual(res["report"]["sample_rate"], 44100)
        path = Path(read_list_lines(res["list"])[0].split("|")[0])
        self.assertEqual(wav_info(path)[2], 44100)

    def test_disk_precheck_refuses_when_too_little_space(self):
        self.use_fake_cutter()
        with self.assertRaises(te.InsufficientSpace):
            self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                        labels_with({"SPEAKER_00": "艾丽丝"}), free_bytes=1024)
        self.assertFalse(self.work.exists())

    def test_disk_precheck_passes_with_plenty_of_space(self):
        self.use_fake_cutter()
        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                          labels_with({"SPEAKER_00": "艾丽丝"}),
                          free_bytes=10 * 1024 ** 3)
        self.assertEqual(res["n_written"], 1)

    def test_should_stop_before_first_clip_aborts(self):
        self.use_fake_cutter()
        with self.assertRaises(te.ExportCancelled):
            self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                        labels_with({"SPEAKER_00": "艾丽丝"}),
                        should_stop=lambda: True)
        self.assertFalse((self.out / "无职转生.list").exists())
        self.assertFalse((self.out / "无职转生.导出报告.json").exists())

    def test_report_and_list_paths(self):
        self.use_fake_cutter()
        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                          labels_with({"SPEAKER_00": "艾丽丝"}))
        self.assertEqual(Path(res["list"]).name, "无职转生.list")
        self.assertEqual(Path(res["report_path"]).name, "无职转生.导出报告.json")
        self.assertTrue(Path(res["report_path"]).is_file())
        self.assertEqual(res["work_dir"], str(self.work))
        self.assertEqual(res["folders"], {"艾丽丝": 1})
        self.assertEqual(res["n_clips"], 1)
        self.assertEqual(res["n_total"], 1)

    def test_source_warning_for_band_limited_sources(self):
        self.assertIsNone(te.source_warning("separated_hq"))
        for kind in ("separated_16k", "raw16k", "media"):
            self.assertTrue(te.source_warning(kind))
        self.assertIsNone(te.source_warning("没见过的东西"))

    def test_source_warning_texts_are_pinned_verbatim(self):
        r"""逐条钉住实际文案（T3）。

        老断言是 `assertEqual(result.warning, te.source_warning(kind))` ——
        两边同源，同义反复：审查者的变异 W14 把 `separated_16k` 的文案换成
        「自造文案：这个源不行」，617 条**全绿**。文案是给用户看的**结论**
        （这条音源为什么不适合克隆音色），改错了用户就会照着做错的决定，
        所以这里逐字写死；要改文案就得**故意**改这个测试。

        每条还叠一层语义锚点：`16 kHz`（来源带宽）、`未分离`（有没有分离）、
        `带 BGM`（干不干净）。锚点取自「这句话在说什么」，措辞微调时不会误报，
        但把整句替换成另一句话时一定会红 —— 万一将来有人把上面的整句断言
        改成「只比锚点」，这一层仍然拦得住 W14。
        """
        expected = {
            "separated_16k":
                "复用既有 16 kHz 输入分离出的人声轨：8 kHz 以上为空，克隆音色会发闷",
            "raw16k": "未分离，直接用 16 kHz 中间产物：高频已丢失且带 BGM",
            "media": "直接从原片切：未分离，带 BGM",
        }
        for kind, text in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(te.source_warning(kind), text,
                                 "%s 的 warning 文案被改了" % kind)
        self.assertIn("16 kHz", te.source_warning("separated_16k"))
        self.assertIn("人声轨", te.source_warning("separated_16k"))
        self.assertIn("未分离", te.source_warning("raw16k"))
        self.assertIn("带 BGM", te.source_warning("raw16k"))
        self.assertIn("原片", te.source_warning("media"))
        self.assertIn("带 BGM", te.source_warning("media"))
        self.assertIsNone(te.source_warning("separated_hq"))

    def test_the_pinned_text_is_what_the_export_returns(self):
        """上面钉的是 `source_warning()`；这条钉的是 `export_work` 真的把它带出来。"""
        self.use_fake_cutter()
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels,
                          source_kind="separated_16k")

        self.assertEqual(
            res["warning"],
            "复用既有 16 kHz 输入分离出的人声轨：8 kHz 以上为空，克隆音色会发闷")

    def test_export_forwards_trim_threshold(self):
        """Task 8 要按片校 trim 阈值：`export_work` 必须能把它透到 `cut_segment`。"""
        seen = []

        def fake_cut(src, start, dur, dest, *, sr=te.DEFAULT_SAMPLE_RATE, trim=True,
                     trim_threshold_db=te.TRIM_THRESHOLD_DB, should_stop=None):
            seen.append(trim_threshold_db)
            write_wav(dest, dur, sr=sr)
            return dest

        with mock.patch.object(te, "cut_segment", fake_cut):
            self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                        labels_with({"SPEAKER_00": "艾丽丝"}),
                        trim_threshold_db=-34.0)

        self.assertEqual(seen, [-34.0])

    def test_export_defaults_trim_threshold_to_minus_40(self):
        seen = []

        def fake_cut(src, start, dur, dest, *, sr=te.DEFAULT_SAMPLE_RATE, trim=True,
                     trim_threshold_db=te.TRIM_THRESHOLD_DB, should_stop=None):
            seen.append(trim_threshold_db)
            write_wav(dest, dur, sr=sr)
            return dest

        with mock.patch.object(te, "cut_segment", fake_cut):
            res = self.export([seg(0.0, 5.0, "台词", "SPEAKER_00")],
                              labels_with({"SPEAKER_00": "艾丽丝"}))

        self.assertEqual(seen, [te.TRIM_THRESHOLD_DB])
        self.assertEqual(res["report"]["trim_threshold_db"], te.TRIM_THRESHOLD_DB)


class TestDiskHelpers(unittest.TestCase):
    def test_estimate_bytes(self):
        self.assertEqual(te.estimate_bytes(1.0, 32000), 64000)
        self.assertEqual(te.estimate_bytes(0.0, 44100), 0)

    def test_disk_free_bytes_is_positive(self):
        self.assertGreater(te.disk_free_bytes(TMP_ROOT), 0)


class TestCutCommand(ExportCase):
    """钉住那条 ffmpeg 命令行（参数顺序与禁用项都是硬要求）。"""

    def _capture(self, **kw):
        seen = {}

        def fake_run(cmd, should_stop=None):
            seen["cmd"] = list(cmd)
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            write_wav(out, 1.0, amp=0.1)             # 伪造产物，让归一化能读
            return 0, ""

        with mock.patch.object(te, "_run_ffmpeg", fake_run):
            te.cut_segment("in.wav", 1.5, 2.0, self.dir / "c.wav", **kw)
        return seen["cmd"]

    def test_cut_command_keeps_ss_before_i_and_nostdin(self):
        cmd = self._capture(sr=32000)
        joined = " ".join(cmd)
        self.assertIn("-nostdin", cmd)
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertLess(cmd.index("-t"), cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-ar") + 1], "32000")
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "pcm_s16le")
        self.assertIn("silenceremove", joined)
        self.assertIn("areverse", joined)
        # 设计 §8：绝不用 loudnorm（会压动态、改音色）
        self.assertNotIn("loudnorm", joined)
        self.assertNotIn("dynaudnorm", joined)

    def test_trim_can_be_turned_off(self):
        cmd = self._capture(sr=44100, trim=False)
        self.assertNotIn("-af", cmd)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "44100")

    def test_trim_threshold_reaches_the_filter(self):
        """N4：trim 阈值可注入（本机实测余量只有 10.4 dB，换更安静的音源要能调）。"""
        cmd = self._capture(trim_threshold_db=-35.0)
        af = cmd[cmd.index("-af") + 1]
        self.assertIn("-35dB", af)
        self.assertIn("detection=peak", af)
        self.assertNotIn("-40dB", af)

    def test_trim_threshold_default_is_minus_40(self):
        cmd = self._capture()
        af = cmd[cmd.index("-af") + 1]
        self.assertIn("-40dB", af)

    def test_ffmpeg_error_raises_and_leaves_no_temp_file(self):
        def fake_run(cmd, should_stop=None):
            return 1, "boom"

        with mock.patch.object(te, "_run_ffmpeg", fake_run):
            with self.assertRaises(te.ExportError):
                te.cut_segment("in.wav", 0.0, 2.0, self.dir / "c.wav")
        self.assertFalse((self.dir / "c.wav").exists())
        self.assertEqual([p.name for p in self.dir.glob("*.tmp.wav")], [])

    def test_peak_normalisation_is_linear_gain(self):
        path = self.dir / "quiet.wav"
        write_wav(path, 1.0, amp=0.1, freq=440.0)
        peak_before = wav_peak_abs(path)
        dbfs = te.normalize_peak_wav(path)
        self.assertAlmostEqual(dbfs, te.PEAK_DBFS, delta=0.05)
        self.assertGreater(wav_peak_abs(path), peak_before)

    def test_peak_normalisation_skips_silence(self):
        path = self.dir / "silent.wav"
        write_wav(path, 0.5, amp=0.0)
        self.assertIsNone(te.normalize_peak_wav(path))

    def test_short_clip_via_plan_goes_to_short_bucket(self):
        """接线：plan_clips 的 `_短句` 与导出目录用的是同一套 bucket 常量。"""
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        clips = te.plan_clips([seg(0.0, 1.0, "短", "SPEAKER_00")], labels)
        self.assertEqual(clips[0]["speaker"], te.SHORT_BUCKET)
        self.assertFalse(te.is_exportable(clips[0]))


# ---------------------------------------------------------------------------
# 真跑 ffmpeg（用 stdlib wave 读回校验）
# ---------------------------------------------------------------------------

@unittest.skipUnless(shutil.which("ffmpeg"), "需要 ffmpeg 在 PATH 上")
class TestRealFfmpeg(ExportCase):
    def setUp(self):
        super().setUp()
        self.src = self.dir / "real.wav"
        write_bursts(self.src, [(1.5, False), (2.0, True), (1.5, False)])

    def test_audio_is_32k_mono_16bit(self):
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        res = te.export_work([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels,
                             self.src, self.out, work_name="作品",
                             silence_finder=fake_silence())

        line = read_list_lines(res["list"])[0]
        wav_path = Path(line.split("|")[0])
        self.assertTrue(wav_path.is_file())
        channels, width, rate, seconds = wav_info(wav_path)
        self.assertEqual(channels, 1)
        self.assertEqual(width, 2)
        self.assertEqual(rate, 32000)
        # 首尾各 1.5 s 静音被 trim 掉
        self.assertAlmostEqual(seconds, 2.0, delta=0.25)

    def test_peak_is_normalised_to_minus_1_dbfs(self):
        labels = labels_with({"SPEAKER_00": "艾丽丝"})

        res = te.export_work([seg(0.0, 5.0, "台词", "SPEAKER_00")], labels,
                             self.src, self.out, work_name="作品",
                             silence_finder=fake_silence())

        wav_path = Path(read_list_lines(res["list"])[0].split("|")[0])
        peak = wav_peak_abs(wav_path)
        self.assertGreater(peak, 0)
        dbfs = 20.0 * np.log10(peak / 32768.0)
        self.assertAlmostEqual(dbfs, -1.0, delta=0.15)

    def test_silence_finder_finds_the_gap(self):
        finder = te.make_silence_finder(self.src)

        spans = finder(0.0, 5.0)

        self.assertEqual(len(spans), 2, spans)
        self.assertAlmostEqual(spans[0][0], 0.0, delta=0.06)
        self.assertAlmostEqual(spans[0][1], 1.5, delta=0.06)
        self.assertAlmostEqual(spans[1][0], 3.5, delta=0.06)
        self.assertAlmostEqual(spans[1][1], 5.0, delta=0.06)

    def test_long_sentence_is_split_with_the_real_finder(self):
        """接缝：真实 silencedetect 也要能驱动拆长（不注入假 finder）。"""
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        parts = [(3.0, True), (0.6, False), (9.0, True), (0.6, False), (3.0, True)]
        write_bursts(self.src, parts)

        res = te.export_work([seg(0.0, 16.2, "长句" * 4, "SPEAKER_00")], labels,
                             self.src, self.out, work_name="作品")

        rep = res["report"]
        self.assertEqual(rep["silence_error"], "")
        self.assertEqual(rep["total_clips"], 2, rep)
        for line in read_list_lines(res["list"]):
            self.assertEqual(len(line.split("|")), 4)
            self.assertTrue(Path(line.split("|")[0]).is_file())


class TestEmptyListIsRefused(ExportCase):
    r"""没有任何片段能进 `.list` 时**拒绝导出**，绝不写出 0 字节的 `.list`。

    0 字节 `.list` 是个哑雷：GPT-SoVITS 的
    `f.read().strip("\n").split("\n")` 会得到 `['']`，解包
    `wav, spk, lang, text = line.split("|")` 当场报错 —— 用户要到建库那一步才发现，
    而且此时已经没有「哪里错了」的线索。所以这里在**计划**阶段就拒绝，一个字节都不写。
    """

    def test_export_refuses_when_no_clip_is_named(self):
        labels = labels_with({})                 # 一个角色都没命名
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00"),
                seg(6.0, 11.0, "台词二", "SPEAKER_01")]
        self.use_fake_cutter()

        with self.assertRaises(te.ExportError):
            self.export(segs, labels)

        # 关键：不能留下 0 字节的 .list，也不能留下半截目录
        self.assertFalse((self.out / "无职转生.list").exists())
        self.assertFalse(self.work.exists())

    def test_precheck_refuses_before_anything_is_written(self):
        labels = labels_with({})
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00")]

        with self.assertRaises(te.ExportError):
            te.precheck_work(segs, labels, self.src, self.out, work_name="无职转生")

        self.assertFalse(self.out.exists(), "预检阶段就该拒绝，且一个字节都不写")

    def test_the_error_says_why_and_what_to_do(self):
        labels = labels_with({})
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00")]

        with self.assertRaises(te.ExportError) as ctx:
            self.export(segs, labels)

        msg = str(ctx.exception)
        self.assertIn("没有任何片段能进训练集", msg)
        self.assertIn("命名", msg)               # 必须给出可执行的下一步

    def test_short_clips_alone_are_refused_and_the_reason_says_short(self):
        """全是短句时也要拒绝，且原因必须指向「短」。

        注意**不能**提示「取消勾选『丢弃短句』」—— 那个开关只决定短片段切不切，
        `is_exportable()` 里短句本来就不进 `.list`，勾不勾都一样。给一个按了也没用的
        建议比不给更糟。
        """
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 1.0, "很短", "SPEAKER_00")]

        with self.assertRaises(te.ExportError) as ctx:
            self.export(segs, labels, drop_short=True)

        msg = str(ctx.exception)
        self.assertIn("短句", msg)
        self.assertNotIn("丢弃短句", msg)

    def test_one_named_clip_still_exports(self):
        """阳性对照：拒绝只针对「一个都没有」，不能把正常导出一起拦掉。"""
        labels = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00")]
        self.use_fake_cutter()

        res = self.export(segs, labels)

        self.assertEqual(res["n_list"], 1)
        self.assertGreater((self.out / "无职转生.list").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
