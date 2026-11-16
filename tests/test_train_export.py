#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""训练集切分核心（train_export.plan_clips）—— 纯逻辑，无任何文件系统访问。

规则来自设计 §8：合并（间隔 < 300 ms，上限 10 s）、拆长（单句 > 12 s 按内部静音切）、
弃短（最终 < 2 s 归 _短句）、归类（角色名取 speaker_labels.effective_speaker）。

静音检测是注入点：silence_finder(start, end) -> [(s, e), ...]。
本文件全部用假实现，所以不碰 ffmpeg、不碰磁盘。
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_labels as sl  # noqa: E402
import train_export as te  # noqa: E402


def labels_with(mapping, overrides=None):
    """造一份标注状态：mapping = {聚类: 角色名}，overrides = {句号: 聚类}。"""
    data = sl.new_labels()
    sl.set_names(data, mapping or {})
    for idx, cluster in (overrides or {}).items():
        sl.set_override(data, idx, cluster)
    return data


def seg(start, end, text, speaker):
    return {"start": float(start), "end": float(end),
            "text": text, "speaker": speaker}


def fake_silence(*spans):
    """返回一个常量 silence_finder：任何区间都报这几段静音。"""
    spans = [tuple(map(float, s)) for s in spans]

    def finder(start, end):
        return [s for s in spans if s[0] >= start and s[1] <= end]

    return finder


class TestMerge(unittest.TestCase):
    def test_merge_adjacent_same_speaker_within_gap(self):
        """间隔 0.2 s < 0.3 s：同角色两句合成一条。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "前半句", "SPEAKER_00"),
                seg(3.2, 6.0, "后半句", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        clip = clips[0]
        self.assertEqual(clip["start"], 0.0)
        self.assertEqual(clip["end"], 6.0)
        self.assertAlmostEqual(clip["dur"], 6.0, places=6)
        self.assertEqual(clip["speaker"], "艾丽丝")

    def test_no_merge_across_different_speaker(self):
        """间隔 0 但角色不同（两个聚类各有一个名字）：绝不合并。"""
        data = labels_with({"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"})
        segs = [seg(0.0, 3.0, "甲说的话", "SPEAKER_00"),
                seg(3.0, 6.0, "乙说的话", "SPEAKER_01")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([c["speaker"] for c in clips], ["艾丽丝", "鲁迪乌斯"])
        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 3.0), (3.0, 6.0)])

    def test_no_merge_when_gap_too_large(self):
        """间隔 0.35 s > 0.3 s：同角色也不合并。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "前", "SPEAKER_00"),
                seg(3.35, 6.0, "后", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 2)
        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 3.0), (3.35, 6.0)])

    def test_merge_respects_10s_cap(self):
        """每句 4.9 s、间隔 0.05 s：前两句合成 9.85 s，第三句不许再并（会超 10 s）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 4.9, "一", "SPEAKER_00"),
                seg(4.95, 9.85, "二", "SPEAKER_00"),
                seg(9.90, 14.80, "三", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 2)
        self.assertAlmostEqual(clips[0]["dur"], 9.85, places=6)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (0.0, 9.85))
        self.assertEqual((clips[1]["start"], clips[1]["end"]), (9.90, 14.80))
        self.assertEqual(clips[0]["pieces"], [0, 1])
        self.assertEqual(clips[1]["pieces"], [2])

    def test_merge_exactly_10s_is_allowed(self):
        """边界断言：合并后**正好** 10.0 s 允许（上限是 > 10 s 才不并）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 4.9, "一", "SPEAKER_00"),
                seg(5.0, 10.0, "二", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[0]["dur"], 10.0, places=6)

    def test_gap_exactly_equal_to_merge_gap_is_not_merged(self):
        """F3 边界：间隔**正好等于** `merge_gap` → 不合并（规则是「间隔 < 阈值」）。

        这条是唯一能真正钉住这个边界的写法：判据一旦被写成 `gap <= merge_gap`
        （把 `<` 放成 `<=`），它就变 1 条。所以间隔取**可精确表示**的值 0.25，
        并配下面 0.249 那条成对。
        """
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "前", "SPEAKER_00"),
                seg(3.25, 6.0, "后", "SPEAKER_00")]
        self.assertEqual(3.25 - 3.0, 0.25)          # 前提：这个间隔在浮点里是精确的

        clips = te.plan_clips(segs, data, merge_gap=0.25)

        self.assertEqual(len(clips), 2)
        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 3.0), (3.25, 6.0)])

    def test_gap_just_under_merge_gap_is_merged(self):
        """F3 边界：间隔 0.249 s < 0.25 s → 合并成一条（覆盖 0~6.0）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "前", "SPEAKER_00"),
                seg(3.249, 6.0, "后", "SPEAKER_00")]

        clips = te.plan_clips(segs, data, merge_gap=0.25)

        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (0.0, 6.0))
        self.assertEqual(clips[0]["text"], "前 后")
        self.assertEqual(clips[0]["pieces"], [0, 1])

    def test_default_300ms_gap_boundary_is_the_same_rule(self):
        """默认阈值 0.3 s 下的同一条规则：3.2 s 处起句（间隔 0.2 s）合并，3.3 s 处（0.3 s）不合并。

        注意这条**钉不住** `<` 边界本身：`3.3 - 3.0` 在浮点里是 0.2999999999999998，
        比 `merge_gap`（0.3）小一点，所以 `gap < merge_gap` 与 `gap <= merge_gap` 结果相同。
        真正的边界由上面 0.25 那对钉住；这条只说明默认阈值下的行为不变。
        """
        data = labels_with({"SPEAKER_00": "艾丽丝"})

        near = te.plan_clips([seg(0.0, 3.0, "前", "SPEAKER_00"),
                              seg(3.2, 6.0, "后", "SPEAKER_00")], data)
        self.assertEqual(len(near), 1)

        far = te.plan_clips([seg(0.0, 3.0, "前", "SPEAKER_00"),
                             seg(3.3, 6.0, "后", "SPEAKER_00")], data)
        self.assertEqual(len(far), 2)


class TestMergeText(unittest.TestCase):
    def test_merged_text_keeps_every_character_in_time_order(self):
        """拼接后的文本按时间顺序、不丢字（只用空白当间隔符）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "是啊", "SPEAKER_00"),
                seg(3.1, 6.0, "鲁迪也不知道呢", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        text = clips[0]["text"]
        self.assertEqual(text, "是啊 鲁迪也不知道呢")
        # 不丢字：去掉空白后就是两段原文的按序拼接
        self.assertEqual("".join(text.split()), "是啊鲁迪也不知道呢")
        # 首句的每个字都在第二句之前出现
        self.assertLess(text.index("是啊"), text.index("鲁迪也不知道呢"))

    def test_single_segment_text_is_untouched(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "  原样保留  ", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(clips[0]["text"], "原样保留")


class TestSplitLong(unittest.TestCase):
    def test_long_sentence_is_split_at_silence(self):
        """14 s 的单句 + 注入一处 6.8~7.2 s 静音 → 两段，而不是超过 12 s 的一段。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 14.0, "あ" * 7 + "い" * 7, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((6.8, 7.2)))

        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0]["start"], 0.0)
        self.assertEqual(clips[0]["end"], 7.0)
        self.assertEqual(clips[1]["start"], 7.0)
        self.assertEqual(clips[1]["end"], 14.0)
        for clip in clips:
            self.assertLessEqual(clip["dur"], 12.0)
        # 两段都能追溯到原始句号 0
        self.assertEqual([c["index"] for c in clips], [0, 0])
        # 文本也跟着静音位置对半分，一个字都没丢
        self.assertEqual(clips[0]["text"], "あ" * 7)
        self.assertEqual(clips[1]["text"], "い" * 7)
        self.assertEqual(clips[0]["text"] + clips[1]["text"],
                         "あ" * 7 + "い" * 7)

    def test_long_sentence_without_silence_stays_whole(self):
        """没有内部静音就没法切：原样返回一段（并标出来，报告能记一笔）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 14.0, "长句", "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence())

        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (0.0, 14.0))
        self.assertTrue(clips[0].get("over_limit"))

    def test_split_when_right_side_is_exactly_6s(self):
        """F1 边界：13 s 的单句 + 静音 6.8~7.2（中点 7.0 = 13 − 6）→ **必须**切成两条。

        右边那截正好等于下限 `split_over/2 = 6 s`。判据写成 `> floor` 时这里一刀不切，
        整条 13 s 带着 `over_limit=True` 留下来。
        """
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 13.0, "x" * 13, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((6.8, 7.2)))

        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 7.0), (7.0, 13.0)])
        self.assertEqual([c["dur"] for c in clips], [7.0, 6.0])
        self.assertEqual("".join(c["text"] for c in clips), "x" * 13)
        for clip in clips:
            self.assertLessEqual(clip["dur"], 12.0)
            self.assertFalse(clip.get("over_limit"))

    def test_split_18s_at_midpoint_12s_gives_12_and_6(self):
        """F1 边界：18 s + 中点 12.0 → 12.0 + 6.0 两条（右段正好 6 s）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 18.0, "y" * 18, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((11.8, 12.2)))

        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 12.0), (12.0, 18.0)])
        self.assertEqual([c["dur"] for c in clips], [12.0, 6.0])
        self.assertEqual("".join(c["text"] for c in clips), "y" * 18)
        # 12.0 正好是目标上限，不算「仍然超长」
        self.assertFalse(clips[0].get("over_limit"))
        self.assertFalse(clips[1].get("over_limit"))

    def test_split_14s_at_midpoint_8s_gives_8_and_6(self):
        """F1 同类边界：14 s + 中点 8.0 → 8.0 + 6.0 两条（右段正好 6 s）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 14.0, "z" * 14, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((7.8, 8.2)))

        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 8.0), (8.0, 14.0)])
        self.assertEqual([c["dur"] for c in clips], [8.0, 6.0])
        self.assertEqual("".join(c["text"] for c in clips), "z" * 14)

    def test_12s_exactly_is_not_split(self):
        """边界断言：正好 12.0 s 不切（规则是 > 12 s）—— 静音也不看。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 12.0, "x" * 6, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((5.9, 6.1)))

        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[0]["dur"], 12.0, places=6)

    def test_silence_too_close_to_the_end_does_not_cut(self):
        """静音贴着句尾（右边只剩 3 s）时不切：切了只能得一个碎尾巴。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 15.0, "w" * 15, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((11.9, 12.1)))

        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (0.0, 15.0))
        self.assertTrue(clips[0].get("over_limit"))

    def test_multiple_silences_are_all_used_when_needed(self):
        """40 s 的句子 + 三处静音（中点 10/20/30）→ 每一处都切：10 / 10 / 10 / 10。

        14 s 那条只用一处静音；这条钉住「多处静音时逐处切开，不留超长截」。
        """
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 40.0, "y" * 40, "SPEAKER_00")]

        clips = te.plan_clips(segs, data,
                             fake_silence((9.8, 10.2), (19.8, 20.2), (29.8, 30.2)))

        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)])
        self.assertEqual("".join(c["text"] for c in clips), "y" * 40)
        for clip in clips:
            self.assertLessEqual(clip["dur"], 12.0)

    def test_all_cut_pieces_share_the_same_original_index(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(5.0, 45.0, "z" * 40, "SPEAKER_00")]

        clips = te.plan_clips(segs, data,
                             fake_silence((14.8, 15.2), (24.8, 25.2), (34.8, 35.2)))

        self.assertEqual({c["index"] for c in clips}, {0})
        self.assertEqual([c["pieces"] for c in clips], [[0], [0], [0], [0]])
        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(5.0, 15.0), (15.0, 25.0), (25.0, 35.0), (35.0, 45.0)])
        self.assertEqual("".join(c["text"] for c in clips), "z" * 40)


class TestBuckets(unittest.TestCase):
    def test_short_clip_goes_to_short_bucket(self):
        """最终 1.5 s → speaker == _短句；且不影响另一条正常片段。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 1.5, "短", "SPEAKER_00"),
                seg(10.0, 16.0, "长", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([c["speaker"] for c in clips], ["_短句", "艾丽丝"])
        short = clips[0]
        self.assertEqual(short["index"], 0)
        self.assertEqual((short["start"], short["end"]), (0.0, 1.5))
        self.assertEqual(short["text"], "短")
        # 另一条按角色名归类，没被短句带跑
        self.assertEqual((clips[1]["start"], clips[1]["end"]), (10.0, 16.0))
        self.assertEqual(clips[1]["text"], "长")

    def test_exactly_2s_is_kept(self):
        """边界断言：正好 2.0 s **保留**（设计写的是「< 2 秒丢弃」）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 2.0, "正好两秒", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["speaker"], "艾丽丝")

    def test_just_under_2s_is_dropped_to_short(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 1.99, "差一点", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(clips[0]["speaker"], "_短句")

    def test_unnamed_and_question_go_to_unnamed(self):
        """空名（只有聚类键）与 `?` 都归 _未命名。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})   # SPEAKER_01 没起名
        segs = [seg(0.0, 5.0, "没名字的聚类", "SPEAKER_01"),
                seg(20.0, 25.0, "问号", "?"),
                seg(40.0, 45.0, "旁白", None),
                seg(60.0, 65.0, "有名字", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([c["speaker"] for c in clips],
                         ["_未命名", "_未命名", "_未命名", "艾丽丝"])

    def test_buckets_are_kept_in_time_order(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "一", "SPEAKER_00"),
                seg(20.0, 21.5, "短", "SPEAKER_00"),
                seg(40.0, 45.0, "二", "SPEAKER_01")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([c["start"] for c in clips], [0.0, 20.0, 40.0])
        self.assertEqual([c["speaker"] for c in clips],
                         ["艾丽丝", "_短句", "_未命名"])


class TestOverride(unittest.TestCase):
    def test_single_sentence_override_is_respected(self):
        """单句改判：第 2 句从 SPEAKER_00 改判到 SPEAKER_01，归属跟着变，也不合并。"""
        data = labels_with({"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"},
                           overrides={1: "SPEAKER_01"})
        segs = [seg(0.0, 3.0, "甲", "SPEAKER_00"),
                seg(3.1, 6.0, "乙", "SPEAKER_00"),
                seg(6.1, 10.0, "丙", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([c["speaker"] for c in clips],
                         ["艾丽丝", "鲁迪乌斯", "艾丽丝"])
        self.assertEqual([c["pieces"] for c in clips], [[0], [1], [2]])

    def test_override_to_question_mark_goes_unnamed(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"}, overrides={0: "?"})
        segs = [seg(0.0, 5.0, "甲", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(clips[0]["speaker"], "_未命名")


class TestShape(unittest.TestCase):
    def test_clip_has_required_keys_and_dur_matches(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(1.0, 6.5, "台词", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        clip = clips[0]
        for key in ("speaker", "index", "start", "end", "dur", "text"):
            self.assertIn(key, clip)
        self.assertEqual(clip["index"], 0)
        self.assertAlmostEqual(clip["dur"], clip["end"] - clip["start"], places=9)
        self.assertAlmostEqual(clip["dur"], 5.5, places=6)

    def test_empty_segments_returns_empty_list(self):
        self.assertEqual(te.plan_clips([], labels_with({})), [])

    def test_blank_text_segment_is_dropped(self):
        """空文本不产出片段（合并规则里也要求「文本都非空」）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "   ", "SPEAKER_00"),
                seg(10.0, 15.0, "有内容", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["index"], 1)
        self.assertEqual(clips[0]["text"], "有内容")

    def test_blank_text_does_not_block_merge_when_gap_is_small(self):
        """F2 契约：空文本段**不构成合并屏障** —— 它被整条丢弃，合并只看时间间隔。

        空句两侧间隔 0.05 s + 0.0 s（都 < 0.3 s）→ 两侧同角色合成 1 条，覆盖 0~6.0。
        空文本段是 ASR 的伪边界（假句），拿它当屏障会把后半句切成 0.8 s 的碎条，
        随后被 `_短句` 丢掉 —— 等于白扔素材。
        """
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "前", "SPEAKER_00"),
                seg(5.05, 5.2, "   ", "SPEAKER_00"),
                seg(5.2, 6.0, "后", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (0.0, 6.0))
        self.assertEqual(clips[0]["text"], "前 后")
        self.assertEqual(clips[0]["pieces"], [0, 2])

    def test_time_gap_blocks_merge_across_a_blank_segment(self):
        """空句两侧间隔 1.1 s > 0.3 s：挡住合并的是**时间间隔**，不是那句空文本。

        （这条原先叫 test_blank_text_blocks_merge —— 名字是错的：它靠的是 1.1 s 间隔，
        空句根本不参与判据。保留用例，改名字与注释，让名字描述它实际测的东西。）
        """
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "前", "SPEAKER_00"),
                seg(5.1, 6.0, "   ", "SPEAKER_00"),
                seg(6.1, 11.0, "后", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([c["text"] for c in clips], ["前", "后"])
        self.assertEqual([c["pieces"] for c in clips], [[0], [2]])

    def test_silence_finder_default_does_not_crash_the_plain_path(self):
        """不注入 finder 时也不能抛（默认实现留给 Task 2，且不该被短句/不需要切的路径调用）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "短句", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)

    def test_split_pieces_go_through_merge_with_the_next_sentence(self):
        """拆长后再走一趟合并：切出来的尾段(7 s)与下一句(14.1~18.0)间隔 0.1 s 且同角色，
        合起来 11 s > 10 s 上限 → 不合并；下一句自己起一条。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 14.0, "本" * 14, "SPEAKER_00"),
                seg(14.1, 18.0, "次", "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((6.8, 7.2)))

        self.assertEqual([(c["start"], c["end"]) for c in clips],
                         [(0.0, 7.0), (7.0, 14.0), (14.1, 18.0)])
        self.assertEqual([c["text"] for c in clips], ["本" * 7, "本" * 7, "次"])


class TestOverLimit(unittest.TestCase):
    """F5：`over_limit` = 「这一条仍然 > split_over(12 s)」，不是「> merge_max(10 s)」。"""

    def test_11s_single_segment_is_not_over_limit(self):
        """11.0 s 的单句：虽然 > 10 s 合并上限，但没超 12 s 目标 → 不标 over_limit。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 11.0, "十一秒", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[0]["dur"], 11.0, places=6)
        self.assertFalse(clips[0].get("over_limit"),
                         "11 s 没超 12 s 目标，不该被标成超长")

    def test_12s_single_segment_is_not_over_limit(self):
        """12.0 s（不多不少）也不该标：规则是 > 12 s 才拆/才超。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 12.0, "十二秒", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertFalse(clips[0].get("over_limit"))

    def test_13_5s_with_sparse_silence_is_over_limit(self):
        """13.5 s 单句 + 贴着句尾的静音（右边只剩 1.4 s，切不开）→ 保持整条且 over_limit=True。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 13.5, "十三点五秒", "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((11.9, 12.1)))

        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (0.0, 13.5))
        self.assertTrue(clips[0].get("over_limit"))

    def test_normal_clip_is_not_over_limit(self):
        """正常片段（6 s）不带 over_limit：Task 2 靠 key 缺失判「没有超长」。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 6.0, "正常", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertNotIn("over_limit", clips[0])
        self.assertFalse(clips[0].get("over_limit"))

    def test_split_pieces_of_a_13s_sentence_are_not_over_limit(self):
        """13 s 切开成 7 + 6 s 后，两条都不再超长（证明 over_limit 跟着**结果**走）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 13.0, "x" * 13, "SPEAKER_00")]

        clips = te.plan_clips(segs, data, fake_silence((6.8, 7.2)))

        self.assertEqual(len(clips), 2)
        for clip in clips:
            self.assertLessEqual(clip["dur"], 12.0)
            self.assertFalse(clip.get("over_limit"))


class TestIsExportable(unittest.TestCase):
    """F4：`is_exportable()` —— Task 2 判「进不进 `.list`」的唯一门，必须单独钉住。"""

    def test_named_speaker_with_text_is_exportable(self):
        self.assertTrue(te.is_exportable(
            {"speaker": "艾丽丝", "text": "台词", "start": 0.0, "end": 3.0}))

    def test_short_bucket_is_not_exportable(self):
        self.assertFalse(te.is_exportable(
            {"speaker": te.SHORT_BUCKET, "text": "短"}))
        # 常量值本身就是「_短句」，别被改名骗过去
        self.assertFalse(te.is_exportable({"speaker": "_短句", "text": "短"}))

    def test_unnamed_bucket_is_not_exportable(self):
        self.assertFalse(te.is_exportable(
            {"speaker": te.UNNAMED_BUCKET, "text": "没名字"}))
        self.assertFalse(te.is_exportable({"speaker": "_未命名", "text": "没名字"}))

    def test_blank_speaker_is_not_exportable(self):
        self.assertFalse(te.is_exportable({"speaker": "", "text": "台词"}))
        self.assertFalse(te.is_exportable({"speaker": "   ", "text": "台词"}))
        self.assertFalse(te.is_exportable({"speaker": None, "text": "台词"}))

    def test_blank_text_is_not_exportable(self):
        self.assertFalse(te.is_exportable({"speaker": "艾丽丝", "text": "   "}))
        self.assertFalse(te.is_exportable({"speaker": "艾丽丝", "text": ""}))
        self.assertFalse(te.is_exportable({"speaker": "艾丽丝", "text": None}))

    def test_missing_text_key_is_not_exportable(self):
        self.assertFalse(te.is_exportable({"speaker": "艾丽丝"}))

    def test_missing_speaker_key_is_not_exportable(self):
        self.assertFalse(te.is_exportable({"text": "台词"}))

    def test_non_dict_and_none_are_not_exportable(self):
        for bad in (None, [], "艾丽丝", 3, ("speaker", "text"), 0.0):
            self.assertFalse(te.is_exportable(bad), "非 dict 一律不可导出：%r" % (bad,))

    def test_real_plan_clips_output_is_exportable(self):
        """接线测试：plan_clips 产出的正常片段确实过得了这道门（防两套判据漂移）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual(len(clips), 1)
        self.assertTrue(te.is_exportable(clips[0]))

    def test_plan_clips_buckets_are_filtered_by_is_exportable(self):
        """接线测试：`_短句` / `_未命名` 都被同一道门挡在外面。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 1.5, "短", "SPEAKER_00"),
                seg(20.0, 25.0, "没名字", "SPEAKER_01"),
                seg(40.0, 45.0, "正常", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)

        self.assertEqual([te.is_exportable(c) for c in clips],
                         [False, False, True])


class TestKwargs(unittest.TestCase):
    """F4：4 个阈值 kwarg 必须真的生效（原来一个测试都没传过）。"""

    def test_merge_gap_kwarg_widens_the_window(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "前", "SPEAKER_00"),
                seg(6.0, 9.0, "后", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)                    # 默认 0.3 s：不并
        self.assertEqual(len(clips), 2)

        wide = te.plan_clips(segs, data, merge_gap=5.0)      # 间隔 3 s < 5 s：并
        self.assertEqual(len(wide), 1)
        self.assertEqual((wide[0]["start"], wide[0]["end"]), (0.0, 9.0))
        self.assertAlmostEqual(wide[0]["dur"], 9.0, places=6)

    def test_merge_gap_kwarg_can_narrow_the_window(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "前", "SPEAKER_00"),
                seg(3.2, 6.0, "后", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)                    # 默认 0.3 s：并
        self.assertEqual(len(clips), 1)

        narrow = te.plan_clips(segs, data, merge_gap=0.1)    # 间隔 0.2 s > 0.1 s：不并
        self.assertEqual(len(narrow), 2)

    def test_merge_max_kwarg_blocks_a_merge_that_would_fit_by_default(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 3.0, "一", "SPEAKER_00"),
                seg(3.1, 6.0, "二", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)                    # 跨度 6.0 ≤ 10：并
        self.assertEqual(len(clips), 1)

        tight = te.plan_clips(segs, data, merge_max=5.0)     # 跨度 6.0 > 5：不并
        self.assertEqual(len(tight), 2)
        self.assertEqual([(c["start"], c["end"]) for c in tight],
                         [(0.0, 3.0), (3.1, 6.0)])

    def test_split_over_kwarg_raises_the_split_threshold(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 14.0, "x" * 14, "SPEAKER_00")]
        finder = fake_silence((6.8, 7.2))

        clips = te.plan_clips(segs, data, finder)            # 默认 12 s：拆成两条
        self.assertEqual(len(clips), 2)

        wide = te.plan_clips(segs, data, finder, split_over=20.0)   # 14 ≤ 20：不拆
        self.assertEqual(len(wide), 1)
        self.assertEqual((wide[0]["start"], wide[0]["end"]), (0.0, 14.0))
        self.assertFalse(wide[0].get("over_limit"))

    def test_split_over_kwarg_lowers_the_split_threshold(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 11.0, "y" * 11, "SPEAKER_00")]
        finder = fake_silence((5.3, 5.7))                    # 中点 5.5，两侧各 5.5 s

        clips = te.plan_clips(segs, data, finder)            # 默认 12 s：11 < 12，不拆
        self.assertEqual(len(clips), 1)

        tight = te.plan_clips(segs, data, finder, split_over=10.0)  # 11 > 10：拆
        self.assertEqual([(c["start"], c["end"]) for c in tight],
                         [(0.0, 5.5), (5.5, 11.0)])
        for clip in tight:
            self.assertFalse(clip.get("over_limit"))

    def test_drop_short_kwarg_moves_the_short_boundary(self):
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 1.5, "短", "SPEAKER_00"),
                seg(20.0, 26.0, "长", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)                    # 默认 2 s：1.5 s 进 _短句
        self.assertEqual([c["speaker"] for c in clips], ["_短句", "艾丽丝"])

        loose = te.plan_clips(segs, data, drop_short=1.0)    # 1.5 ≥ 1.0：两条都留
        self.assertEqual([c["speaker"] for c in loose], ["艾丽丝", "艾丽丝"])

        strict = te.plan_clips(segs, data, drop_short=7.0)   # 6.0 < 7.0：两条都归 _短句
        self.assertEqual([c["speaker"] for c in strict], ["_短句", "_短句"])


class TestPurity(unittest.TestCase):
    """F4：`plan_clips` 是纯函数 —— 不得就地修改入参（Task 2 接缝第 4 条）。"""

    def test_inputs_are_not_mutated(self):
        import copy

        data = labels_with({"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪乌斯"},
                           overrides={2: "SPEAKER_01"})
        segs = [seg(0.0, 14.0, "长句" * 3, "SPEAKER_00"),
                seg(14.1, 15.0, "空", "SPEAKER_00"),
                seg(15.05, 15.2, "   ", "SPEAKER_00"),
                seg(15.2, 20.0, "后", "SPEAKER_00"),
                seg(20.1, 30.0, "改判", "SPEAKER_00"),
                seg(40.0, 41.5, "短", "SPEAKER_01")]

        segs_before = copy.deepcopy(segs)
        data_before = copy.deepcopy(data)

        te.plan_clips(segs, data, fake_silence((6.8, 7.2)))

        self.assertEqual(segs, segs_before, "plan_clips 改动了入参 segments")
        self.assertEqual(data, data_before, "plan_clips 改动了入参 labels")

    def test_returned_clips_do_not_alias_input_dicts(self):
        """返回值里也不能挂着入参 dict 的引用（改产物不该改到入参）。"""
        data = labels_with({"SPEAKER_00": "艾丽丝"})
        segs = [seg(0.0, 5.0, "台词", "SPEAKER_00")]

        clips = te.plan_clips(segs, data)
        clips[0]["text"] = "被改过"

        self.assertEqual(segs[0]["text"], "台词")


if __name__ == "__main__":
    unittest.main(verbosity=2)
