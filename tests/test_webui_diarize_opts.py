#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""WebUI 侧说话人分割参数接线的单元测试（不启动 HTTP 服务）。

运行：
    venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import webui  # noqa: E402


def _req(**kw):
    base = {
        "asr_model": None, "separate": False, "sep_model": "",
        "vad": True, "beam": 5, "glossary": "",
        "diarize": False, "diarize_model": "", "num_speakers": None,
        "diarize_min_overlap": None, "mode": "asr",
    }
    base.update(kw)
    return base


class TestOptsKey(unittest.TestCase):
    def test_diarize_toggle_changes_key(self):
        a = webui._opts_key({"diarize": False})
        b = webui._opts_key({"diarize": True})
        self.assertNotEqual(a, b)

    def test_model_and_speaker_count_change_key(self):
        base = {"diarize": True, "diarize_model": "m1", "num_speakers": None}
        k1 = webui._opts_key(base)
        k2 = webui._opts_key({**base, "diarize_model": "m2"})
        k3 = webui._opts_key({**base, "num_speakers": 2})
        k4 = webui._opts_key({**base, "diarize_min_overlap": 0.5})
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertNotEqual(k1, k4)

    def test_irrelevant_field_does_not_change_key(self):
        k1 = webui._opts_key({"diarize": True})
        k2 = webui._opts_key({"diarize": True, "output_dir": "elsewhere"})
        self.assertEqual(k1, k2)


class TestEnqueueOpts(unittest.TestCase):
    def test_diarize_defaults(self):
        jid = webui._enqueue(Path(r"C:\sample-project\uploads\x.mp4"),
                             _req(diarize=True), False)
        opts = webui.JOBS[jid]["opts"]
        self.assertTrue(opts["diarize"])
        self.assertEqual(opts["diarize_model"], webui.A.DEFAULT_DIARIZE_MODEL)
        self.assertIsNone(opts["num_speakers"])
        self.assertEqual(opts["diarize_min_overlap"], 0.3)

    def test_diarize_off_by_default(self):
        jid = webui._enqueue(Path(r"C:\sample-project\uploads\y.mp4"),
                             _req(), False)
        opts = webui.JOBS[jid]["opts"]
        self.assertFalse(opts["diarize"])

    def test_explicit_values_kept(self):
        jid = webui._enqueue(
            Path(r"C:\sample-project\uploads\z.mp4"),
            _req(diarize=True, diarize_model="custom/model",
                 num_speakers=3, diarize_min_overlap=0.5),
            False)
        opts = webui.JOBS[jid]["opts"]
        self.assertEqual(opts["diarize_model"], "custom/model")
        self.assertEqual(opts["num_speakers"], 3)
        self.assertEqual(opts["diarize_min_overlap"], 0.5)


class TestJobView(unittest.TestCase):
    def test_job_view_carries_opts(self):
        jid = webui._enqueue(Path(r"C:\sample-project\uploads\w.mp4"),
                             _req(diarize=True), False)
        view = webui._job_view(webui.JOBS[jid])
        self.assertTrue(view["opts"]["diarize"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
