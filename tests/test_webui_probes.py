#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""F14：`_detect_diarization` 的就绪探测必须真的探测。

旧实现只检查 `DIARIZE_PYTHON` / `DIARIZE_WORKER` 两个文件在不在，
torch 与 pyannote 一次都不导入 —— 环境坏掉时面板照样显示「已就绪」，
用户白跑一整趟才发现分割被跳过、字幕里一个说话人标签都没有。
提示还是写死的「未设置 HF_TOKEN —— 首次使用需要 token」，
而本机 `models\pyannote\` 三个子模型齐全时**根本不需要 token**。

这里用 monkeypatch 把子进程替换掉，所以这几条测试不需要真的装 pyannote，
也不需要 GPU：它们只验证「探测怎么判、提示怎么说、命令走没走兼容层」。
"""

import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jisheng as A  # noqa: E402
import webui  # noqa: E402

WEBUI_SRC = (HERE / "webui.py").read_text(encoding="utf-8")

# 兼容层自检脚本的真实输出长这样（Lead 实测）
OK_STDOUT = "pyannote 3.4.0\ntorch    2.13.0+cu132 cuda True\n"


class ProbeBase(unittest.TestCase):
    def setUp(self):
        self._py = A.DIARIZE_PYTHON
        self._wk = A.DIARIZE_WORKER
        self._run = webui.subprocess.run
        self._local = A._local_pyannote_pipeline
        self._ready = webui.DIARIZE_READY
        self._note = webui.DIARIZE_NOTE
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        A.DIARIZE_PYTHON = root / "python.exe"
        A.DIARIZE_PYTHON.write_bytes(b"")
        A.DIARIZE_WORKER = root / "diarize_worker.py"
        A.DIARIZE_WORKER.write_bytes(b"")
        self.calls = []

    def tearDown(self):
        A.DIARIZE_PYTHON = self._py
        A.DIARIZE_WORKER = self._wk
        A._local_pyannote_pipeline = self._local
        webui.subprocess.run = self._run
        webui.DIARIZE_READY = self._ready
        webui.DIARIZE_NOTE = self._note
        self._tmp.cleanup()

    def fake_probe(self, returncode, stdout="", stderr=""):
        def run(cmd, **kw):
            self.calls.append({"cmd": list(cmd), "kw": dict(kw)})
            return types.SimpleNamespace(returncode=returncode,
                                         stdout=stdout, stderr=stderr)
        return run

    def set_local_models(self, present):
        A._local_pyannote_pipeline = (lambda: Path("D:/fake/local-pipeline/config.yaml")
                                      if present else None)


class TestDiarizationProbe(ProbeBase):
    def test_failed_probe_is_not_ready_and_surfaces_the_error(self):
        webui.subprocess.run = self.fake_probe(
            1, stderr="AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'")
        ready = webui._detect_diarization()
        self.assertFalse(ready, "探测失败却报『就绪』——环境坏了面板会骗人")
        self.assertFalse(webui.DIARIZE_READY)
        self.assertIn("torchaudio", webui.DIARIZE_NOTE,
                      "note 里要能看见失败原因，否则用户无从下手")

    def test_probe_actually_imports_pyannote_through_the_compat_shim(self):
        """命令里必须带 pyannote_compat.py。

        裸 `import pyannote.audio` 会失败（torchaudio 2.11 删了 AudioMetaData），
        只有那个脚本的 __main__ 里先 apply_all() 再 import 才成功。
        不带它就等于「探测一个必然失败的东西」，环境永远是红的。
        """
        webui.subprocess.run = self.fake_probe(0, stdout=OK_STDOUT)
        self.set_local_models(True)
        webui._detect_diarization()
        self.assertEqual(len(self.calls), 1, "应该只跑一次探测子进程")
        cmd = self.calls[0]["cmd"]
        self.assertEqual(cmd[0], str(A.DIARIZE_PYTHON))
        self.assertTrue(any(c.endswith("pyannote_compat.py") for c in cmd),
                        f"探测命令没走兼容层：{cmd}")
        self.assertEqual(self.calls[0]["kw"].get("timeout"), 240)

    def test_local_models_note_does_not_ask_for_a_token(self):
        webui.subprocess.run = self.fake_probe(0, stdout=OK_STDOUT)
        self.set_local_models(True)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            self.assertTrue(webui._detect_diarization())
        self.assertTrue(webui.DIARIZE_READY)
        self.assertNotIn("HF_TOKEN", webui.DIARIZE_NOTE,
                         "本地模型齐全时提 token 是把『不需要做的事』写成前提条件")
        self.assertIn("本地模型", webui.DIARIZE_NOTE)

    def test_online_model_note_does_mention_the_token(self):
        webui.subprocess.run = self.fake_probe(0, stdout=OK_STDOUT)
        self.set_local_models(False)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            self.assertTrue(webui._detect_diarization())
        self.assertIn("HF_TOKEN", webui.DIARIZE_NOTE,
                      "没有本地模型时确实需要 token，提示必须说")

    def test_missing_environment_files_still_fail_fast(self):
        A.DIARIZE_PYTHON = Path(self._tmp.name) / "不存在.exe"
        webui.subprocess.run = self.fake_probe(0, stdout=OK_STDOUT)
        self.assertFalse(webui._detect_diarization())
        self.assertEqual(self.calls, [], "文件都不在就不该起子进程")


class TestProbesAreWiredConcurrently(unittest.TestCase):
    """两个探测串行跑会让启动白等两者之和（实测 2.5 秒 + 10.3 秒）。"""

    def test_main_calls_the_concurrent_helper(self):
        """静态断言：main() 必须调 _detect_environments()。

        防的是「函数写好了但没接上」——那正是本次修复踩过的坑：
        _detect_environments 一度是死代码，启动还是串行。
        """
        self.assertIn("def _detect_environments(", WEBUI_SRC)
        m = re.search(r"\ndef main\(\)[^\n]*:\n", WEBUI_SRC)
        self.assertIsNotNone(m, "webui.py 里找不到 main()")
        rest = WEBUI_SRC[m.end():]
        # main() 的函数体 = 到下一个顶格非空行为止
        nxt = re.search(r"\n(?=[^\s#])", rest)
        body = rest[:nxt.start()] if nxt else rest
        self.assertIn("_detect_environments()", body,
                      "main() 没有调用 _detect_environments()（探测又变成串行了）")
        self.assertNotIn("_detect_separation()", body,
                         "main() 仍在直接调 _detect_separation()，会与并发版重复跑")
        self.assertNotIn("_detect_diarization()", body,
                         "main() 仍在直接调 _detect_diarization()，会与并发版重复跑")

    def test_both_probes_overlap_in_time(self):
        """行为断言：两个探测必须真的并发，而不是写了个没人用的函数。"""
        state = {"running": 0, "max": 0}
        guard = threading.Lock()
        delay = 0.35

        def make_probe():
            def probe():
                with guard:
                    state["running"] += 1
                    state["max"] = max(state["max"], state["running"])
                time.sleep(delay)
                with guard:
                    state["running"] -= 1
            return probe

        with mock.patch.object(webui, "_detect_separation", make_probe()), \
             mock.patch.object(webui, "_detect_diarization", make_probe()):
            t0 = time.time()
            webui._detect_environments()
            elapsed = time.time() - t0

        self.assertEqual(state["max"], 2,
                         "两个探测没有重叠：_detect_environments 还是串行的")
        self.assertLess(elapsed, delay * 1.9,
                        f"并发跑却花了 {elapsed:.2f}s（串行应为 {delay*2:.2f}s）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
