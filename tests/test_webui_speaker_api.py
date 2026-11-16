#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""说话人标注接口（模块级函数，不启动 HTTP 服务）。"""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_labels as sl  # noqa: E402
import speaker_registry as sr  # noqa: E402
import webui  # noqa: E402


def setUpModule():
    """兜底：**任何**用例都不许往用户真实的 `speakers_registry\\` 里写东西。

    大部分用例走 SpeakerApiBase（它会指到自己的临时根），这里再兜一层 ——
    将来有人加了不继承该基类的用例，也不会污染真实库。

    这一层能成立的前提是「没人把根改回 None」。`sr.set_default_root(None)` 不是
    「复位」，而是**指回真实库** —— 曾经 tearDown 里就写了这一句，于是类按字母序
    跑完第一个用例，兜底就作废了（`probe_pollution.py` 的探针看得到真实根）。
    所以这里连「进入本模块时的根」一起记下来，tearDownModule 恢复的也是它。
    """
    global _GUARD_ROOT, _GUARD_PREV
    _GUARD_ROOT = tempfile.TemporaryDirectory(prefix="jisheng_registry_guard_")
    _GUARD_PREV = sr.default_root()
    sr.set_default_root(_GUARD_ROOT.name)


def tearDownModule():
    """恢复「进入本模块时的根」，并且真的把临时根删掉。"""
    global _GUARD_ROOT
    sr.set_default_root(_GUARD_PREV)
    if _GUARD_ROOT is not None:
        _GUARD_ROOT.cleanup()
        _GUARD_ROOT = None

DIA = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
       {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
       {"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"},
       {"start": 40.0, "end": 45.0, "speaker": "?"}]

SEGS = [{"start": 0.0, "end": 4.0, "text": "第一句", "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 15.0, "text": "第二句", "speaker": "SPEAKER_01"},
        {"start": 20.0, "end": 26.0, "text": "第三句", "speaker": "SPEAKER_00"},
        {"start": 40.0, "end": 44.0, "text": "听不清", "speaker": "?"}]


class SpeakerApiBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._work_dir = webui.WORK_DIR
        self._out_root = webui.OUTPUT_ROOT
        self._jobs = webui.JOBS
        webui.WORK_DIR = self.root / "work"
        webui.OUTPUT_ROOT = self.root / "output"
        webui.JOBS = {}
        # api_assign 命名成功会调 speaker_registry.enroll()。库根必须指向临时目录，
        # 否则整套测试会往**用户真实的** speakers_registry\ 里写角色。
        self._reg_prev = sr.default_root()
        sr.set_default_root(str(self.root / "registry"))
        self.workdir = webui.WORK_DIR / "EP01"
        self.workdir.mkdir(parents=True)
        (self.workdir / "diarization.json").write_text(
            json.dumps(DIA, ensure_ascii=False), encoding="utf-8")
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in SEGS) + "\n",
            encoding="utf-8")

    def tearDown(self):
        webui.WORK_DIR = self._work_dir
        webui.OUTPUT_ROOT = self._out_root
        webui.JOBS = self._jobs
        # 恢复**进入本用例前**的根，绝不写 None —— set_default_root(None) 是把根
        # 指回真实库（`<项目>\speakers_registry`），会让后续不继承本基类的用例
        # 直接写进用户数据。
        sr.set_default_root(getattr(self, "_reg_prev", None))
        self._tmp.cleanup()

    def make_job(self, jid="j1", out_dir=None):
        out = Path(out_dir) if out_dir else self.root / "output"
        out.mkdir(parents=True, exist_ok=True)
        webui.JOBS[jid] = {"id": jid, "media": str(self.root / "EP01.mkv"),
                           "work": str(self.workdir), "out_dir": str(out),
                           "logs": [], "status": "done", "stage": "完成"}
        return webui.JOBS[jid]


class TestRegistryGuard(unittest.TestCase):
    """库根必须是隔离根：**任何**用例期间都不许指回真实库（F5 的兜底）。"""

    REAL = Path(sr.project_dir()) / sr._STORAGE_DIRNAME

    def test_default_root_is_not_the_real_library(self):
        root = Path(sr.default_root())
        self.assertNotEqual(root, self.REAL,
                            "用例跑在真实库根上：enroll 会写进用户数据")
        self.assertNotIn(str(self.REAL), str(sr.vectors_path()))

    def test_probe_of_a_class_that_does_not_extend_the_base(self):
        """模拟「将来有人加了不继承 SpeakerApiBase 的用例」。

        类按字母序执行，所以这条用例前面已经跑过若干继承基类的用例 ——
        旧实现（tearDown 里 set_default_root(None)）到这里时根已经变回真实库。
        """
        self.assertEqual(Path(sr.default_root()), Path(_GUARD_ROOT.name))
        self.assertNotEqual(Path(sr.vectors_path()).parent, self.REAL)


class TestSpeakerList(SpeakerApiBase):
    def test_unknown_job_is_404(self):
        obj, code = webui.api_speakers("nope")
        self.assertEqual(code, 404)

    def test_work_name_mode_works_without_live_job(self):
        obj, code = webui.api_speakers(None, "EP01")
        self.assertEqual(code, 200)
        self.assertEqual([c["id"] for c in obj["clusters"]],
                         ["SPEAKER_00", "SPEAKER_01"])

    def test_missing_diarization_is_409(self):
        self.make_job()
        (self.workdir / "diarization.json").unlink()
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 409)

    def test_clusters_carry_counts_and_labels(self):
        self.make_job()
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 200)
        first = obj["clusters"][0]
        self.assertEqual(first["id"], "SPEAKER_00")
        self.assertEqual(first["label"], "[说话人1]")
        self.assertEqual(first["n_segments"], 2)
        self.assertEqual(first["name"], "")
        self.assertEqual(obj["unknown_segments"], 1)

    def test_clips_are_proposed_even_without_audio(self):
        self.make_job()
        obj, _ = webui.api_speakers("j1")
        clips = obj["clusters"][0]["clips"]
        self.assertTrue(clips)
        self.assertIn("overlap", clips[0])
        self.assertEqual(obj["source"], "")

    def test_no_workdir_is_404(self):
        obj, code = webui.api_speakers(None, "NOT_THERE")
        self.assertEqual(code, 404)

    def test_wrong_shaped_diarization_is_409(self):
        """能解析但形状不对（不是对象数组）也必须 409，不能 500。"""
        self.make_job()
        for bad in ("5", "[1,2]", '{"a":1}'):
            (self.workdir / "diarization.json").write_text(bad, encoding="utf-8")
            obj, code = webui.api_speakers("j1")
            self.assertEqual(code, 409, bad)

    def test_bad_timestamps_in_diarization_is_409(self):
        self.make_job()
        (self.workdir / "diarization.json").write_text(
            '[{"start":"abc","end":1.0,"speaker":"SPEAKER_00"}]', encoding="utf-8")
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 409)

    def test_nan_timestamp_in_diarization_is_409(self):
        """NaN 能过 float()，但会让 total_s 变成 NaN、json 输出非法 JSON，必须 409。"""
        self.make_job()
        (self.workdir / "diarization.json").write_text(
            '[{"start":NaN,"end":1.0,"speaker":"SPEAKER_00"}]', encoding="utf-8")
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 409)

    def test_bad_timestamps_in_segments_is_409(self):
        self.make_job()
        (self.workdir / "segments.jsonl").write_text(
            '{"start":"x","end":1.0,"text":"a","speaker":"SPEAKER_00"}\n',
            encoding="utf-8")
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 409)


class TestSpeakerSegments(SpeakerApiBase):
    """「单句改判」界面需要逐句信息：i/start/end/text。"""

    def test_cluster_carries_its_segments_with_line_numbers(self):
        self.make_job()
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 200)
        first = obj["clusters"][0]
        self.assertEqual([s["i"] for s in first["segments"]], [0, 2])
        self.assertEqual([s["text"] for s in first["segments"]], ["第一句", "第三句"])
        self.assertEqual(first["segments"][0]["start"], 0.0)
        self.assertEqual(first["segments"][0]["end"], 4.0)

    def test_segments_are_sorted_by_line_number(self):
        self.make_job()
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in reversed(SEGS)) + "\n",
            encoding="utf-8")
        obj, _ = webui.api_speakers("j1")
        for c in obj["clusters"]:
            idx = [s["i"] for s in c["segments"]]
            self.assertEqual(idx, sorted(idx))

    def test_text_is_truncated_to_200_chars(self):
        self.make_job()
        (self.workdir / "segments.jsonl").write_text(
            json.dumps({"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
                        "text": "あ" * 240}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        obj, _ = webui.api_speakers("j1")
        seg = obj["clusters"][0]["segments"][0]
        self.assertEqual(len(seg["text"]), 200)

    def test_text_is_stripped(self):
        self.make_job()
        (self.workdir / "segments.jsonl").write_text(
            json.dumps({"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
                        "text": "  有空白  "}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["clusters"][0]["segments"][0]["text"], "有空白")

    def test_unknown_block_lists_the_question_mark_segments(self):
        self.make_job()
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["unknown"]["n"], 1)
        self.assertEqual([s["i"] for s in obj["unknown"]["segments"]], [3])
        self.assertEqual(obj["unknown"]["segments"][0]["text"], "听不清")
        # 老字段保留，前端已在用
        self.assertEqual(obj["unknown_segments"], 1)
        self.assertEqual(obj["unknown"]["n"], obj["unknown_segments"])

    def test_unknown_block_is_present_and_empty_without_question_marks(self):
        self.make_job()
        (self.workdir / "segments.jsonl").write_text(
            json.dumps({"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
                        "text": "只有一句"}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["unknown"], {"n": 0, "segments": []})

    def test_segment_index_feeds_straight_into_reassign(self):
        """segments[i].i 必须是 api_reassign 认的那个 index，否则界面点哪句都改不中。"""
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝",
                                                 "SPEAKER_01": "鲁迪乌斯"}})
        obj, _ = webui.api_speakers("j1")
        target = [s for s in obj["clusters"][0]["segments"]
                  if s["text"] == "第三句"][0]
        res, code = webui.api_reassign(
            {"job": "j1", "index": target["i"], "to": "SPEAKER_01"})
        self.assertEqual(code, 200)
        srt = Path(res["srt"]).read_text(encoding="utf-8")
        self.assertIn("[鲁迪乌斯] 第三句", srt)
        self.assertIn("[艾丽丝] 第一句", srt)

    def test_unknown_index_also_feeds_into_reassign(self):
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝"}})
        obj, _ = webui.api_speakers("j1")
        idx = obj["unknown"]["segments"][0]["i"]
        res, code = webui.api_reassign(
            {"job": "j1", "index": idx, "to": "SPEAKER_00"})
        self.assertEqual(code, 200)
        srt = Path(res["srt"]).read_text(encoding="utf-8")
        self.assertIn("[艾丽丝] 听不清", srt)

    def test_segment_override_changes_the_listed_cluster(self):
        """改判之后那句要出现在新聚类下面、从旧聚类里消失。"""
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝",
                                                 "SPEAKER_01": "鲁迪乌斯"}})
        webui.api_reassign({"job": "j1", "index": 2, "to": "SPEAKER_01"})
        obj, _ = webui.api_speakers("j1")
        by_id = {c["id"]: c for c in obj["clusters"]}
        # 改判过来的第 2 句要出现在 SPEAKER_01 底下，并从 SPEAKER_00 里消失
        self.assertEqual([s["i"] for s in by_id["SPEAKER_01"]["segments"]], [1, 2])
        self.assertEqual([s["i"] for s in by_id["SPEAKER_00"]["segments"]], [0])
        names = {c["id"]: c["name"] for c in obj["clusters"]}
        self.assertEqual(names["SPEAKER_01"], "鲁迪乌斯")
        # 卡片头写的句数必须等于展开的逐句行数（同一口径：当前标注归属）
        self.assertEqual(by_id["SPEAKER_01"]["n_segments"], 2)
        self.assertEqual(by_id["SPEAKER_00"]["n_segments"], 1)

    def test_n_segments_always_equals_the_listed_rows(self):
        """卡片句数（n_segments）恒等于逐句列表长度 —— 两者必须同口径。

        改判前也断言一次，这样「句数用分割口径、列表用标注口径」的不一致
        在任何一个改动方向上都当场红。
        """
        self.make_job()

        def check(tag):
            obj, code = webui.api_speakers("j1")
            self.assertEqual(code, 200, tag)
            for c in obj["clusters"]:
                self.assertEqual(c["n_segments"], len(c["segments"]),
                                 f"{tag}: {c['id']} 卡片句数与逐句行数不一致")
                self.assertEqual(
                    c["total_s"], round(sum(s["end"] - s["start"]
                                            for s in c["segments"]), 1),
                    f"{tag}: {c['id']} 卡片时长与逐句列表不一致")

        check("未改判")
        webui.api_reassign({"job": "j1", "index": 2, "to": "SPEAKER_01"})
        check("改判一句之后")

    def test_card_counts_zero_when_every_line_is_reassigned_away(self):
        """把一个聚类的句子全改判成 [?] 后，卡片不能再写「2 句」而展开 0 行。"""
        self.make_job()
        for i in (0, 2):
            _, code = webui.api_reassign({"job": "j1", "index": i, "to": "?"})
            self.assertEqual(code, 200)
        obj, _ = webui.api_speakers("j1")
        by_id = {c["id"]: c for c in obj["clusters"]}
        self.assertEqual(by_id["SPEAKER_00"]["n_segments"], 0)
        self.assertEqual(by_id["SPEAKER_00"]["segments"], [])
        self.assertEqual(by_id["SPEAKER_00"]["total_s"], 0.0)
        # 逐句列表里现在应该有 3 句 [?]（原来的 3 号 + 改判过来的 0/2）
        self.assertEqual(obj["unknown"]["n"], 3)

    def test_clips_keep_the_partition_scope_after_a_reassign(self):
        """试听片段来自分割结果，不跟着 override 变 —— 这条语义是刻意的。"""
        self.make_job()
        before, _ = webui.api_speakers("j1")
        before = {c["id"]: c["clips"] for c in before["clusters"]}
        webui.api_reassign({"job": "j1", "index": 0, "to": "SPEAKER_01"})
        after, _ = webui.api_speakers("j1")
        after = {c["id"]: c["clips"] for c in after["clusters"]}
        # 逐句列表换了主人，但片段候选仍按「分割结果」挑，所以原样不动
        self.assertEqual(before, after)
        self.assertTrue(after["SPEAKER_00"])


class TestOtherOverrideTargets(SpeakerApiBase):
    """改判到「新建角色…」（new:<名字>）之后那句必须还能在面板里找到。

    否则用户误点一次「新建角色…」就再也改不回来 —— 那句话在所有列表里都不出现。
    """

    def _reassign_new(self, name="路人甲", index=0):
        obj, code = webui.api_reassign({"job": "j1", "index": index,
                                        "to": "new:" + name})
        self.assertEqual(code, 200)
        return obj

    def test_new_role_segment_is_listed_in_others(self):
        self.make_job()
        self._reassign_new()
        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 200)
        self.assertEqual(obj["others"]["n"], 1)
        self.assertEqual([s["i"] for s in obj["others"]["segments"]], [0])
        self.assertEqual(obj["others"]["segments"][0]["text"], "第一句")
        self.assertEqual(obj["others"]["segments"][0]["start"], 0.0)
        self.assertEqual(obj["others"]["segments"][0]["end"], 4.0)

    def test_others_does_not_duplicate_the_normal_cluster_listing(self):
        """只有「不在 clusters 里」的 override 键才进 others。"""
        self.make_job()
        webui.api_reassign({"job": "j1", "index": 2, "to": "SPEAKER_01"})
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["others"], {"n": 0, "segments": []})
        self.assertEqual(obj["unknown"]["n"], 1)

    def test_others_is_present_and_empty_without_new_roles(self):
        self.make_job()
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["others"], {"n": 0, "segments": []})

    def test_reassigning_the_other_line_back_to_a_cluster_clears_it(self):
        """误点「新建角色…」必须能撤销：把它改判回真正的聚类键即可。"""
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝"}})
        self._reassign_new()
        res, code = webui.api_reassign({"job": "j1", "index": 0,
                                        "to": "SPEAKER_00"})
        self.assertEqual(code, 200)
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["others"], {"n": 0, "segments": []})
        by_id = {c["id"]: c for c in obj["clusters"]}
        self.assertEqual([s["i"] for s in by_id["SPEAKER_00"]["segments"]], [0, 2])
        srt = Path(res["srt"]).read_text(encoding="utf-8")
        self.assertIn("[艾丽丝] 第一句", srt)

    def test_override_key_missing_from_labels_is_not_listed(self):
        """手改过 speakers.json、override 指向一个 clusters 里没有的键时，
        面板不能列出一句「无法撤销」的行（下拉里根本没有那个目标）。"""
        self.make_job()
        labels = sl.load_labels(self.workdir)
        labels["segment_overrides"] = {"1": "幽灵角色"}
        sl.save_labels(self.workdir, labels)
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["others"], {"n": 0, "segments": []})

    def test_stale_new_role_key_still_listed_so_it_can_be_undone(self):
        """真正的 new: 角色在 labels['clusters'] 里有键，所以始终可达。"""
        self.make_job()
        self._reassign_new("路人甲")
        labels = sl.load_labels(self.workdir)
        self.assertIn("路人甲", labels["clusters"])
        obj, _ = webui.api_speakers("j1")
        listed = {c["id"] for c in obj["clusters"]}
        self.assertNotIn("路人甲", listed)        # 它本来就不在分割结果里
        self.assertEqual(obj["others"]["n"], 1)   # 但在 others 里可撤销


class TestAssign(SpeakerApiBase):
    def test_assign_names_rewrites_srt_and_txt(self):
        self.make_job()
        obj, code = webui.api_assign({"job": "j1",
                                      "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200)
        self.assertTrue(obj["rewritten"])
        srt = Path(obj["srt"]).read_text(encoding="utf-8")
        self.assertIn("[艾丽丝] 第一句", srt)
        self.assertIn("[说话人2] 第二句", srt)
        self.assertIn("[?] 听不清", srt)

    def test_assign_merges_two_clusters(self):
        self.make_job()
        webui.api_assign({"job": "j1",
                          "names": {"SPEAKER_00": "艾丽丝", "SPEAKER_01": "艾丽丝"}})
        labels = sl.load_labels(self.workdir)
        self.assertEqual(sl.cluster_name(labels, "SPEAKER_00"), "艾丽丝")
        self.assertEqual(sl.cluster_name(labels, "SPEAKER_01"), "艾丽丝")

    def test_assign_saves_work_name(self):
        self.make_job()
        webui.api_assign({"job": "j1", "work_name": "无职转生",
                          "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(sl.get_work_name(sl.load_labels(self.workdir)), "无职转生")

    def test_assign_is_idempotent_on_the_output(self):
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝"}})
        obj, _ = webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝"}})
        srt = Path(obj["srt"]).read_text(encoding="utf-8")
        self.assertEqual(srt.count("[艾丽丝]"), 2)

    def test_assign_ignores_unknown_cluster_keys(self):
        """请求里带一个拼错的键，不能凭空造出一个幽灵角色。"""
        self.make_job()
        obj, code = webui.api_assign({"job": "j1", "names": {
            "SPEAKER_00": "艾丽丝", "SPEAKER_99": "幽灵"}})
        self.assertEqual(code, 200)
        labels = sl.load_labels(self.workdir)
        self.assertIn("SPEAKER_00", labels["clusters"])
        self.assertNotIn("SPEAKER_99", labels["clusters"])

    def test_assign_rejects_non_object_names(self):
        self.make_job()
        obj, code = webui.api_assign({"job": "j1", "names": "abc"})
        self.assertEqual(code, 400)

    def test_assign_ignores_non_string_name_values(self):
        """value 不是字符串既不能抛异常，也不能把坏值写进状态。"""
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝"}})
        obj, code = webui.api_assign({"job": "j1", "names": {"SPEAKER_00": 5}})
        self.assertEqual(code, 200)
        self.assertEqual(
            sl.cluster_name(sl.load_labels(self.workdir), "SPEAKER_00"), "艾丽丝")

    def test_assign_rejects_non_string_work_name(self):
        self.make_job()
        obj, code = webui.api_assign({"job": "j1", "work_name": 7,
                                      "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 400)

    def test_assign_without_job_reads_media_and_out_dir_from_state(self):
        """检查点没有 speakers.json 时也要写到真实输出路径，不能拿 work 目录名当片名。"""
        media = self.root / "番剧" / "S01E07-这是一个很长的片名.mkv"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"")
        out = self.root / "custom_out"
        (self.workdir / "state.json").write_text(json.dumps(
            {"media": str(media), "out_dir": str(out)}, ensure_ascii=False),
            encoding="utf-8")
        self.assertFalse((self.workdir / "speakers.json").exists())

        obj, code = webui.api_assign({"work": "EP01",
                                      "names": {"SPEAKER_00": "艾丽丝"}})

        self.assertEqual(code, 200)
        expect = out / f"{media.stem}.ja.srt"
        self.assertEqual(obj["srt"], str(expect))
        self.assertTrue(expect.is_file())
        # 顺手把 media 写进检查点，下次不必再猜
        self.assertEqual(sl.load_labels(self.workdir).get("media"), str(media))

    def test_assign_with_corrupt_segments_is_409(self):
        self.make_job()
        (self.workdir / "segments.jsonl").write_text(
            '{"start":null,"end":1.0,"text":"a","speaker":"SPEAKER_00"}\n',
            encoding="utf-8")
        obj, code = webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 409)

    def test_concurrent_assign_all_succeed(self):
        """HTTP 处理是多线程的：并发标注不能有任何一个抛异常或写坏状态。"""
        self.make_job()
        errors = []

        def worker(i):
            try:
                webui.api_assign({"job": "j1", "names": {"SPEAKER_00": f"角色{i}"}})
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        name = sl.cluster_name(sl.load_labels(self.workdir), "SPEAKER_00") or ""
        self.assertTrue(name.startswith("角色"), name)


class TestReassign(SpeakerApiBase):
    def test_reassign_moves_one_line(self):
        self.make_job()
        webui.api_assign({"job": "j1", "names": {"SPEAKER_00": "艾丽丝",
                                                 "SPEAKER_01": "鲁迪乌斯"}})
        obj, code = webui.api_reassign({"job": "j1", "index": 2, "to": "SPEAKER_01"})
        self.assertEqual(code, 200)
        srt = Path(obj["srt"]).read_text(encoding="utf-8")
        self.assertIn("[鲁迪乌斯] 第三句", srt)

    def test_reassign_accepts_new_cluster(self):
        self.make_job()
        webui.api_assign({"job": "j1", "names": {}})
        _, code = webui.api_reassign({"job": "j1", "index": 0, "to": "new:路人甲"})
        self.assertEqual(code, 200)
        labels = sl.load_labels(self.workdir)
        self.assertIn("路人甲", labels["clusters"])

    def test_reassign_requires_index_and_target(self):
        self.make_job()
        _, code = webui.api_reassign({"job": "j1", "index": 0, "to": ""})
        self.assertEqual(code, 400)

    def test_reassign_bad_index_is_400(self):
        self.make_job()
        _, code = webui.api_reassign({"job": "j1", "index": "abc", "to": "SPEAKER_01"})
        self.assertEqual(code, 400)

    def test_reassign_rejects_non_string_to(self):
        """传数组/对象进来不能抛 AttributeError 变成 500。"""
        self.make_job()
        obj, code = webui.api_reassign({"job": "j1", "index": 0, "to": ["SPEAKER_01"]})
        self.assertEqual(code, 400)

    def test_reassign_rejects_non_string_work_name(self):
        self.make_job()
        _, code = webui.api_reassign({"job": "j1", "index": 0,
                                      "to": "SPEAKER_01", "work_name": 7})
        self.assertEqual(code, 400)

    def test_reassign_out_of_range_index_is_400(self):
        """越界 index 不能悄悄写进 segment_overrides 还返回 200。"""
        self.make_job()
        obj, code = webui.api_reassign({"job": "j1", "index": 999, "to": "SPEAKER_01"})
        self.assertEqual(code, 400)
        self.assertEqual(sl.load_labels(self.workdir).get("segment_overrides") or {}, {})


class TestLoadSegments(unittest.TestCase):
    """segments.jsonl 是逐行 append 的，中间任何一行坏掉都不能连累后面的句子。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, lines):
        (self.work / "segments.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _line(text):
        return json.dumps({"start": 0.0, "end": 1.0, "text": text},
                          ensure_ascii=False)

    def test_bad_line_in_the_middle_keeps_later_lines(self):
        self._write([self._line("第一句"), "{ 坏行", self._line("第三句")])
        out = webui.load_segments(self.work)
        self.assertEqual([s["text"] for s in out], ["第一句", "第三句"])

    def test_bad_line_at_the_end_keeps_earlier_lines(self):
        """断电截断最后一行是常态，前面的句子必须全部保留。"""
        self._write([self._line("第一句"), self._line("第二句"),
                     '{"start":0.0,"end":1.0,"te'])
        out = webui.load_segments(self.work)
        self.assertEqual([s["text"] for s in out], ["第一句", "第二句"])

    def test_all_lines_bad_is_empty(self):
        self._write(["坏", "也坏"])
        self.assertEqual(webui.load_segments(self.work), [])

    def test_blank_lines_are_skipped(self):
        self._write([self._line("第一句"), "", "   ", self._line("第二句")])
        out = webui.load_segments(self.work)
        self.assertEqual([s["text"] for s in out], ["第一句", "第二句"])

    def test_missing_file_is_empty(self):
        self.assertEqual(webui.load_segments(self.work), [])


class TestLabelLock(unittest.TestCase):
    """_LABEL_LOCKS 的键必须归一化，否则同一目录的大小写/相对路径变体各拿一把锁。"""

    def test_same_dir_variants_share_one_lock(self):
        w = Path(tempfile.mkdtemp())
        try:
            self.assertIs(webui._label_lock(w),
                          webui._label_lock(Path(str(w).upper())))
            self.assertIs(webui._label_lock(w), webui._label_lock(str(w)))
        finally:
            shutil.rmtree(w, ignore_errors=True)

    def test_relative_and_absolute_paths_share_one_lock(self):
        w = Path(tempfile.mkdtemp())
        old = os.getcwd()
        try:
            os.chdir(w.parent)
            self.assertIs(webui._label_lock(w),
                          webui._label_lock(os.path.join(".", w.name)))
        finally:
            os.chdir(old)
            shutil.rmtree(w, ignore_errors=True)

    def test_different_dirs_get_different_locks(self):
        a = Path(tempfile.mkdtemp())
        b = Path(tempfile.mkdtemp())
        try:
            self.assertIsNot(webui._label_lock(a), webui._label_lock(b))
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_nested_acquisition_from_the_same_thread_does_not_deadlock(self):
        """F8 要求 _wipe_checkpoint 持锁后再进 _wipe_annotation，必须可重入。"""
        w = Path(tempfile.mkdtemp())
        done = threading.Event()
        errors = []

        def worker():
            try:
                with webui._label_lock(w):
                    with webui._label_lock(w):
                        pass
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))
            finally:
                done.set()

        try:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=5)
            self.assertTrue(done.is_set(), "同一线程嵌套获取 _label_lock 卡死了")
            self.assertEqual(errors, [])
        finally:
            shutil.rmtree(w, ignore_errors=True)


class TestDiaSpeakers(unittest.TestCase):
    def test_keeps_string_speakers_only_and_sorts(self):
        """手改过的 diarization.json 里畸形条目不能把接口炸掉。"""
        self.assertEqual(webui._dia_speakers({"a": 1}), [])
        self.assertEqual(webui._dia_speakers(["x"]), [])
        self.assertEqual(webui._dia_speakers("abc"), [])
        self.assertEqual(webui._dia_speakers(
            [{"speaker": "SPEAKER_00"}, {"speaker": None}, {},
             {"speaker": 7}, {"speaker": ["a"]}, "junk"]), ["SPEAKER_00"])
        self.assertEqual(webui._dia_speakers(
            [{"speaker": "SPEAKER_01"}, {"speaker": "SPEAKER_00"}]),
            ["SPEAKER_00", "SPEAKER_01"])


class TestWorkdirs(SpeakerApiBase):
    def test_lists_checkpoints_with_diarization(self):
        (webui.WORK_DIR / "EP02").mkdir()
        (webui.WORK_DIR / "EP02" / "state.json").write_text("{}", encoding="utf-8")
        obj, code = webui.api_workdirs()
        self.assertEqual(code, 200)
        self.assertEqual([w["work"] for w in obj["items"]], ["EP01"])

    def test_empty_when_no_work_dir(self):
        shutil.rmtree(webui.WORK_DIR)
        obj, code = webui.api_workdirs()
        self.assertEqual(code, 200)
        self.assertEqual(obj["items"], [])


class TestWorkdirsOrder(SpeakerApiBase):
    """「最近优先」必须是 diarization.json 的 mtime 倒序，而不是目录名逆字典序。"""

    def _mk(self, name, mtime):
        d = webui.WORK_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        dia = d / "diarization.json"
        dia.write_text("[]", encoding="utf-8")
        os.utime(dia, (mtime, mtime))
        return d

    def test_sorted_by_diarization_mtime_desc(self):
        """名字序与时间序刻意相反：最久没碰过的 EP01 反而名字最小，
        纯按目录名排序会把它排到最后。"""
        shutil.rmtree(webui.WORK_DIR)          # 去掉 setUp 造的 EP01
        self._mk("EP01", 1_700_000_003)        # 最近标注
        self._mk("EP03", 1_700_000_002)
        self._mk("EP02", 1_700_000_001)        # 最久没碰
        obj, code = webui.api_workdirs()
        self.assertEqual(code, 200)
        self.assertEqual([w["work"] for w in obj["items"]],
                         ["EP01", "EP03", "EP02"])

    def test_equal_mtime_falls_back_to_name_for_a_stable_order(self):
        shutil.rmtree(webui.WORK_DIR)
        t = 1_700_000_000
        self._mk("BBB", t)
        self._mk("AAA", t)
        obj, _ = webui.api_workdirs()
        self.assertEqual([w["work"] for w in obj["items"]], ["AAA", "BBB"])

    def test_still_only_lists_dirs_with_diarization(self):
        (webui.WORK_DIR / "EP02").mkdir()
        (webui.WORK_DIR / "EP02" / "state.json").write_text("{}", encoding="utf-8")
        obj, _ = webui.api_workdirs()
        self.assertEqual([w["work"] for w in obj["items"]], ["EP01"])

    def test_unreadable_diarization_does_not_break_the_listing(self):
        """mtime 读不到（目录被删）不能把整个列表变成 500。"""
        shutil.rmtree(webui.WORK_DIR)
        d = self._mk("AAA", 1_700_000_001)
        (d / "diarization.json").unlink()
        obj, code = webui.api_workdirs()
        self.assertEqual(code, 200)
        self.assertEqual(obj["items"], [])


class TestClipPath(SpeakerApiBase):
    def test_missing_clip_returns_none(self):
        self.make_job()
        self.assertIsNone(webui.api_clip_path("SPEAKER_00", 1, "j1"))

    def test_existing_clip_returns_path(self):
        self.make_job()
        p = webui.speaker_clips.clip_path(self.workdir, "SPEAKER_00", 1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF")
        self.assertEqual(webui.api_clip_path("SPEAKER_00", 1, "j1"), p)


class TestRunJobIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._work_dir = webui.WORK_DIR
        webui.WORK_DIR = self.root / "work"
        self.work = webui.WORK_DIR / "EP01"
        self.work.mkdir(parents=True)

    def tearDown(self):
        webui.WORK_DIR = self._work_dir
        self._tmp.cleanup()

    def test_stale_checkpoint_cleanup_list_includes_labels_and_clips(self):
        """参数变化时必须连标注状态和片段缓存一起清掉。"""
        (self.work / "speakers.json").write_text('{"version":1,"clusters":{}}',
                                                 encoding="utf-8")
        (self.work / "clips").mkdir()
        (self.work / "clips" / "SPEAKER_00_1.wav").write_bytes(b"RIFF")
        (self.work / "state.json").write_text("{}", encoding="utf-8")

        webui._wipe_checkpoint(self.work)

        self.assertFalse((self.work / "speakers.json").exists())
        self.assertFalse((self.work / "clips").exists())
        self.assertFalse((self.work / "state.json").exists())

    def test_wipe_annotation_clears_labels_and_clips_only(self):
        """重新分割后只作废标注与片段缓存，识别结果必须留着。"""
        (self.work / "speakers.json").write_text('{"version":1,"clusters":{}}',
                                                 encoding="utf-8")
        (self.work / "clips").mkdir()
        (self.work / "clips" / "SPEAKER_00_1.wav").write_bytes(b"RIFF")
        (self.work / "segments.jsonl").write_text("{}\n", encoding="utf-8")
        (self.work / "state.json").write_text("{}", encoding="utf-8")

        webui._wipe_annotation(self.work)

        self.assertFalse((self.work / "speakers.json").exists())
        self.assertFalse((self.work / "clips").exists())
        self.assertTrue((self.work / "segments.jsonl").is_file())
        self.assertTrue((self.work / "state.json").is_file())


class TestWipeUnderLabelLock(unittest.TestCase):
    """清检查点必须和 api_assign 的「读-改-写」互斥（F8）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name) / "EP01"
        self.work.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self):
        (self.work / "speakers.json").write_text('{"version":1,"clusters":{}}',
                                                 encoding="utf-8")
        (self.work / "clips").mkdir(exist_ok=True)
        (self.work / "clips" / "SPEAKER_00_1.wav").write_bytes(b"RIFF")
        (self.work / "state.json").write_text("{}", encoding="utf-8")

    def test_wipe_checkpoint_while_holding_the_same_lock_does_not_deadlock(self):
        self._seed()
        done = threading.Event()
        errors = []

        def worker():
            try:
                with webui._label_lock(self.work):
                    webui._wipe_checkpoint(self.work)
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertTrue(done.is_set(), "持锁调 _wipe_checkpoint 卡死了")
        self.assertEqual(errors, [])
        self.assertFalse((self.work / "speakers.json").exists())
        self.assertFalse((self.work / "clips").exists())
        self.assertFalse((self.work / "state.json").exists())

    def test_wipe_annotation_while_holding_the_same_lock_does_not_deadlock(self):
        self._seed()
        done = threading.Event()

        def worker():
            try:
                with webui._label_lock(self.work):
                    webui._wipe_annotation(self.work)
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertTrue(done.is_set(), "持锁调 _wipe_annotation 卡死了")
        self.assertFalse((self.work / "speakers.json").exists())

    def test_wipe_waits_for_another_thread_holding_the_lock(self):
        """线程 A 持锁期间线程 B 的 _wipe_checkpoint 必须等；A 释放后 B 才动手。

        用锁获取计数做确定性断言，不用 sleep 猜时序。
        """
        self._seed()
        b_entered = threading.Event()
        b_done = threading.Event()
        a_holding = threading.Event()

        def worker_b():
            # 打一个探针：真正进 wipe 之前置位，用来证明它没有越过锁
            orig = webui.shutil.rmtree

            def probe(*a, **k):
                b_entered.set()
                return orig(*a, **k)

            webui.shutil.rmtree = probe
            try:
                webui._wipe_checkpoint(self.work)
            finally:
                webui.shutil.rmtree = orig
                b_done.set()

        lock = webui._label_lock(self.work)
        tb = threading.Thread(target=worker_b, daemon=True)
        with lock:
            a_holding.set()
            tb.start()
            # A 持锁期间 B 一定进不去临界区（rmtree 探针没被调用）
            self.assertFalse(b_done.wait(timeout=0.5),
                             "B 在 A 仍持锁时就完成了——说明没有互斥")
            self.assertFalse(b_entered.is_set(),
                             "B 在 A 仍持锁时就越过锁动了 clips/")
            self.assertTrue((self.work / "clips").exists())
        self.assertTrue(b_done.wait(timeout=5), "A 释放锁后 B 仍未完成")
        self.assertEqual(webui._label_lock(self.work), lock)
        self.assertFalse((self.work / "clips").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
