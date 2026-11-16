#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""「放弃任务」相关逻辑的单元测试（不启动 HTTP 服务）。

放弃 = 停止 + 删除检查点，且必须能从「卡住/已停止」的任务里脱身。
"""

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import webui  # noqa: E402


class TestAbandonFlag(unittest.TestCase):
    def test_default_false(self):
        self.assertFalse(webui.abandon_requested({}))

    def test_set_true(self):
        self.assertTrue(webui.abandon_requested({"abandon": True}))

    def test_stop_is_independent_of_abandon(self):
        self.assertTrue(webui.stop_requested({"stop": True}))
        self.assertFalse(webui.abandon_requested({"stop": True}))


class TestDiscardJob(unittest.TestCase):
    def setUp(self):
        self._orig = webui.WORK_DIR
        self._tmp = tempfile.TemporaryDirectory()
        webui.WORK_DIR = Path(self._tmp.name)

    def tearDown(self):
        webui.WORK_DIR = self._orig
        self._tmp.cleanup()

    def test_removes_checkpoint_dir(self):
        media = Path(r"C:\sample-project\uploads\EP01.mkv")
        work = webui.WORK_DIR / webui.safe_name(media.stem)
        (work / "sep").mkdir(parents=True)
        (work / "state.json").write_text("{}", encoding="utf-8")
        (work / "segments.jsonl").write_text("{}\n", encoding="utf-8")

        webui._discard_job({"media": str(media)})

        self.assertFalse(work.exists())
        # 整个 work 根目录还在（别的任务还要用）
        self.assertTrue(webui.WORK_DIR.exists())

    def test_missing_dir_is_noop(self):
        webui._discard_job({"media": r"C:\sample-project\uploads\NOPE.mkv"})

    def test_empty_job_is_noop(self):
        webui._discard_job({})

    def test_only_touches_that_job(self):
        keep = Path(r"C:\sample-project\uploads\EP02.mkv")
        drop = Path(r"C:\sample-project\uploads\EP01.mkv")
        for m in (keep, drop):
            d = webui.WORK_DIR / webui.safe_name(m.stem)
            d.mkdir(parents=True)
            (d / "state.json").write_text("{}", encoding="utf-8")

        webui._discard_job({"media": str(drop)})

        self.assertFalse((webui.WORK_DIR / webui.safe_name(drop.stem)).exists())
        self.assertTrue((webui.WORK_DIR / webui.safe_name(keep.stem)).exists())

    def test_export_job_never_deletes_a_checkpoint(self):
        """P2：导出任务不拥有检查点，放弃它**绝不能**删别人的检查点。

        导出任务的 `media` 是它引用的那个 ASR 任务的原片，而 `_discard_job`
        是按 `media` 推检查点目录的 —— 不挡住的话，「放弃导出」会把用户命名好的
        `speakers.json` / `segments.jsonl` / `diarization.json` 整份删掉。
        """
        media = Path(r"C:\sample-project\uploads\EP01.mkv")
        work = webui.WORK_DIR / webui.safe_name(media.stem)
        (work / "sep").mkdir(parents=True)
        for name in ("speakers.json", "segments.jsonl", "diarization.json"):
            (work / name).write_text("{}", encoding="utf-8")

        webui._discard_job({"id": "e1", "kind": "export", "media": str(media),
                            "work": str(work)})

        self.assertTrue(work.is_dir(), "放弃导出把检查点目录删了")
        for name in ("speakers.json", "segments.jsonl", "diarization.json"):
            self.assertTrue((work / name).is_file(), "%s 被删了" % name)

    def test_unknown_kinds_still_follow_the_old_semantics(self):
        """回归：只有 `kind == "export"` 例外，别的（含没有 kind 的旧任务）照旧删。"""
        media = Path(r"C:\sample-project\uploads\EP03.mkv")
        work = webui.WORK_DIR / webui.safe_name(media.stem)
        work.mkdir(parents=True)
        (work / "state.json").write_text("{}", encoding="utf-8")

        webui._discard_job({"id": "j3", "media": str(media)})

        self.assertFalse(work.exists())


class TestEnqueueDefaults(unittest.TestCase):
    def test_new_job_is_not_abandoned(self):
        jid = webui._enqueue(Path(r"C:\sample-project\uploads\a.mkv"),
                             {"mode": "asr"}, False)
        job = webui.JOBS[jid]
        self.assertFalse(job.get("abandon"))
        self.assertFalse(job.get("stop"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
