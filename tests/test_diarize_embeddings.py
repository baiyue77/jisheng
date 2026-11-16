#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""diarize_worker 产出聚类质心 + WebUI 写进标注状态 / 命名入库。

规则来源：
- 计划 Task 5 / 设计 §6.1：分割完把每个聚类的质心写进 `speakers.json` 的
  `clusters[<id>]["embedding"]`，命名时 `speaker_registry.enroll(...)` 入库，
  匹配结果写进 `clusters[<id>]["match_at_enroll"]`（Task 7 从这个名字改过来：
  它与 `GET /api/speakers` 实时算出来的 `match` 同名不同形 —— 那份有 `band`、
  是当前建议；这份是入库那一刻的审计快照，多一个 `vector_stored`）。
- Lead 复核过的 pyannote 3.4 源码事实（`speaker_diarization.py`）：
  `pipeline(audio, return_embeddings=True)` 返回 `(diarization, centroids)`；
  634-636 在 labels 多于质心行数时用**全零行**补齐；622-627 `centroids is None`
  是 OracleClustering 的正常路径。

**stdout 协议不许破**：调用方只认 `RESULT=` 那一行，`EMBEDDINGS=` 只能是**追加**行。

真跑 pyannote 的对齐测试要 `JISHENG_RUN_INTEGRATION=1`（会让 skipped 变多，这是预期的）。
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import types
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jisheng as A  # noqa: E402
import speaker_labels as sl  # noqa: E402
import speaker_registry as sr  # noqa: E402

MEDIA = HERE / "compare" / "片段_16m00s_19m00s.mp4"
# 只读素材（不复制、不改动）：Task 5 真跑用的 16 kHz 单声道
RAW16K = HERE / "work" / "片段_16m00s_19m00s" / "raw16k.wav"

# 本模块的隔离根。理由同 tests/test_webui_speaker_api.py 的 setUpModule：
# 库根是**全局**状态，只要有一个用例忘了指回临时根，`enroll` 就会写进
# 用户真实的 `speakers_registry\`（那种污染不报错、只悄悄改数据）。
_GUARD = {}


def setUpModule():
    _GUARD["root"] = tempfile.TemporaryDirectory(prefix="jisheng_reg_guard_t5_")
    _GUARD["prev"] = sr.default_root()
    sr.set_default_root(_GUARD["root"].name)


def tearDownModule():
    """恢复到「进入本模块时的根」而不是 None。

    写 None 看着像复位，实际是**指回真实库**——那让 setUpModule 的保证只覆盖
    半个模块（第一个用例之后守卫就失效了）。恢复原值才是真的隔离。
    """
    sr.set_default_root(_GUARD.pop("prev", None))
    tmp = _GUARD.pop("root", None)
    if tmp is not None:
        tmp.cleanup()


def _enabled() -> bool:
    return os.environ.get("JISHENG_RUN_INTEGRATION") == "1"


# ---------------------------------------------------------------------------
# 把 diarize_worker 塞进 ASR 环境（本环境没有 torch / pyannote）
# ---------------------------------------------------------------------------

class _Seg:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeDiarization:
    """够用的替身：只要 `itertracks(yield_label=True)` 与 `labels()`。

    `labels()` 是质心逐行对齐的**另一半**（worker 靠它给 sidecar 里的行贴名字），
    所以替身必须照真 pyannote 的样子实现：按出现顺序去重。
    """

    def __init__(self, turns):
        self._turns = list(turns)

    def itertracks(self, yield_label=False):  # noqa: ARG002 - 签名要与 pyannote 一致
        for start, end, label in self._turns:
            yield _Seg(start, end), None, label

    def labels(self):
        out = []
        for _s, _e, label in self._turns:
            if label not in out:
                out.append(label)
        return out


class _FakePipeline:
    """记录调用参数，返回 (diarization, centroids)。"""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append({"audio": audio, "kwargs": dict(kwargs)})
        return self._result


def _install_fake_modules():
    """往 sys.modules 里放假的 torch / pyannote.audio，让 worker 的 import 成立。

    返回复原函数。`pyannote.audio.Pipeline` 是 `diarize_worker` 唯一用到的名字，
    注入之后 `main()` 里那段 `from pyannote.audio import Pipeline` 也会拿到它 ——
    所以离线测试可以覆盖 main() 的完整参数解析与 stdout 行为。
    """
    saved = {name: sys.modules.get(name, None)
             for name in ("torch", "pyannote", "pyannote.audio")}

    class _FakePipelineFactory:
        from_pretrained = staticmethod(lambda *a, **k: _RejectPipeline())

    class _RejectPipeline:
        def to(self, *a, **k):        # noqa: ARG002
            return self

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.device = lambda *a: "cpu"
    audio_mod = types.ModuleType("pyannote.audio")
    audio_mod.Pipeline = _FakePipelineFactory
    pkg_mod = types.ModuleType("pyannote")
    pkg_mod.audio = audio_mod

    sys.modules["torch"] = torch_mod
    sys.modules["pyannote"] = pkg_mod
    sys.modules["pyannote.audio"] = audio_mod

    def restore():
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    return restore


def _load_worker():
    """按这里注入的假依赖加载 diarize_worker（每个测试删掉重建，避免串味）。"""
    spec = importlib.util.spec_from_file_location(
        "diarize_worker", HERE / "diarize_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _turns(n=2):
    return [(float(i), float(i) + 0.5, "SPEAKER_%02d" % i) for i in range(n)]


def _read_sidecar(path):
    with np.load(str(path)) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# diarize_worker.py
# ---------------------------------------------------------------------------

class WorkerCase(unittest.TestCase):
    def setUp(self):
        self._restore = _install_fake_modules()
        self.addCleanup(self._restore)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "a.wav"
        self.src.write_bytes(b"RIFF")
        self.loader = lambda _p: {"waveform": None, "sample_rate": 16000,
                                  "uri": "a"}

    def new_worker(self):
        mod = _load_worker()
        mod._load_wav = self.loader
        return mod

    def run_worker(self, worker, result, emb=None, out_json=None):
        """跑一次 run_pipeline（pipeline 是替身，调用参数会被记录）。"""
        out_json = out_json or (self.tmp / "dia.json")
        pipe = _FakePipeline(result)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = worker.run_pipeline(
                pipe, {"waveform": None, "sample_rate": 16000},
                {"num_speakers": None}, out_json, emb)
        return code, out.getvalue(), pipe, out_json


class TestWorkerRequestsEmbeddings(WorkerCase):
    def test_worker_requests_embeddings(self):
        """`return_embeddings=True` 必须真传给 pipeline（不能只写在文档里）。"""
        worker = self.new_worker()
        emb = self.tmp / "out.embeddings.npz"
        code, _text, pipe, out_json = self.run_worker(
            worker, (_FakeDiarization(_turns()),
                     np.ones((2, 256), dtype=np.float32)), emb)

        self.assertEqual(code, 0)
        self.assertEqual(len(pipe.calls), 1)
        self.assertIs(pipe.calls[0]["kwargs"].get("return_embeddings"), True)
        self.assertTrue(out_json.is_file())
        self.assertTrue(emb.is_file(), "拿到质心就必须落 sidecar")

    def test_worker_writes_labels_and_centroids_into_sidecar(self):
        """sidecar 要自带 labels：调用方靠它才知道第 i 行是哪个聚类。"""
        worker = self.new_worker()
        cents = (np.arange(2 * 256, dtype=np.float32).reshape(2, 256) + 1.0)
        emb = self.tmp / "out.embeddings.npz"
        self.run_worker(worker, (_FakeDiarization(_turns()), cents), emb)

        data = _read_sidecar(emb)
        self.assertEqual([str(x) for x in data["labels"]],
                         ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(data["centroids"].shape, (2, 256))
        np.testing.assert_allclose(data["centroids"], cents, rtol=1e-6)

    def test_worker_prints_embeddings_line_after_result(self):
        """EMBEDDINGS= 只能是 RESULT= 之后的**追加**行。"""
        worker = self.new_worker()
        emb = self.tmp / "out.embeddings.npz"
        code, text, _pipe, _out = self.run_worker(
            worker, (_FakeDiarization(_turns()),
                     np.ones((2, 256), dtype=np.float32)), emb)
        lines = [ln for ln in text.splitlines() if ln]
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith("RESULT="), lines)
        self.assertEqual(lines[1], "EMBEDDINGS=%s" % emb)


class TestWorkerDegradesGracefully(WorkerCase):
    def test_worker_handles_none_centroids(self):
        """OracleClustering 正常路径：仍然写 RESULT、不打印 EMBEDDINGS、不抛。"""
        worker = self.new_worker()
        emb = self.tmp / "out.embeddings.npz"
        code, text, _pipe, out_json = self.run_worker(
            worker, (_FakeDiarization(_turns()), None), emb)

        self.assertEqual(code, 0)
        self.assertIn("RESULT=", text)
        self.assertNotIn("EMBEDDINGS=", text)
        self.assertFalse(emb.exists(), "拿不到质心就不能留半个 sidecar")
        rows = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 2, "分割结果该有还是有")

    def test_worker_handles_short_centroid_array(self):
        """labels 多于质心行数（pyannote 补全零行）→ 不写 sidecar，分割照旧。"""
        worker = self.new_worker()
        # 3 个标签、2 行质心 —— 源码 634-636 的补位场景
        turns = _turns(3)
        emb = self.tmp / "out.embeddings.npz"
        code, text, _pipe, out_json = self.run_worker(
            worker, (_FakeDiarization(turns),
                     np.ones((2, 256), dtype=np.float32)), emb)

        self.assertEqual(code, 0)
        self.assertIn("RESULT=", text)
        self.assertNotIn("EMBEDDINGS=", text)
        self.assertFalse(emb.exists())
        self.assertEqual(len(json.loads(out_json.read_text(encoding="utf-8"))), 3)

    def test_worker_handles_empty_centroid_array(self):
        """静音输入 → (0, 256)；空矩阵同样不写 sidecar。"""
        worker = self.new_worker()
        emb = self.tmp / "out.embeddings.npz"
        code, text, _pipe, _out = self.run_worker(
            worker, (_FakeDiarization([]), np.zeros((0, 256), dtype=np.float32)),
            emb)
        self.assertEqual(code, 0)
        self.assertIn("RESULT=", text)
        self.assertNotIn("EMBEDDINGS=", text)
        self.assertFalse(emb.exists())

    def test_worker_handles_1d_centroid_array(self):
        """形状异常（一维）也不许抛、不许写 sidecar。"""
        worker = self.new_worker()
        emb = self.tmp / "out.embeddings.npz"
        code, text, _pipe, _out = self.run_worker(
            worker, (_FakeDiarization(_turns()), np.ones(256, dtype=np.float32)),
            emb)
        self.assertEqual(code, 0)
        self.assertIn("RESULT=", text)
        self.assertNotIn("EMBEDDINGS=", text)
        self.assertFalse(emb.exists())

    def test_worker_handles_nan_centroid_rows(self):
        """NaN 行不许写进 sidecar（会污染余弦）。"""
        worker = self.new_worker()
        cents = np.ones((2, 256), dtype=np.float32)
        cents[1, 0] = np.nan
        emb = self.tmp / "out.embeddings.npz"
        code, text, _pipe, _out = self.run_worker(
            worker, (_FakeDiarization(_turns()), cents), emb)
        self.assertEqual(code, 0)
        self.assertIn("RESULT=", text)
        self.assertNotIn("EMBEDDINGS=", text)
        self.assertFalse(emb.exists())

    def test_worker_without_embeddings_out_writes_nothing_extra(self):
        """老调用方不传 --embeddings-out：行为与改造前一致（只有 RESULT=）。"""
        worker = self.new_worker()
        code, text, pipe, _out = self.run_worker(
            worker, (_FakeDiarization(_turns()),
                     np.ones((2, 256), dtype=np.float32)), None)
        self.assertEqual(code, 0)
        self.assertIn("RESULT=", text)
        self.assertNotIn("EMBEDDINGS=", text)
        self.assertEqual(list(self.tmp.glob("*.npz")), [])
        # 但 return_embeddings 仍然要开：调用方可能只是不要 sidecar
        self.assertIs(pipe.calls[0]["kwargs"].get("return_embeddings"), True)


class TestWriteEmbeddingsUnit(WorkerCase):
    """write_embeddings 的直接行为（不经过 pipeline）。"""

    def test_returns_none_when_labels_longer_than_rows(self):
        worker = self.new_worker()
        got = worker.write_embeddings(self.tmp / "x.npz", ["A", "B", "C"],
                                      np.ones((2, 256), dtype=np.float32))
        self.assertIsNone(got)
        self.assertFalse((self.tmp / "x.npz").exists())

    def test_more_rows_than_labels_is_rejected(self):
        """行数多于标签数同样整份不写：契约是**严格相等**。

        曾经的实现放行这一支并原样写盘（理由：「多出来的行按标签顺序取，
        丢掉多余的就行」）。那个理由不成立：pyannote 3.4 在返回前用
        `inverse_mapping` 对质心做的是**置换**（speaker_diarization.py:641-644），
        不是「前 N 行还是前 N 行」。把行数多当成安全，等于假设一个并不存在的
        对齐关系，而读端（webui.load_cluster_embeddings）对这个形状是整份不信的
        —— 写方说合格、读方说不要，同一份文件两个答案。两边统一成严格相等。
        """
        worker = self.new_worker()
        p = self.tmp / "x.npz"
        self.assertIsNone(worker.write_embeddings(
            p, ["A"], np.ones((3, 256), dtype=np.float32)))
        self.assertFalse(p.exists())

    def test_empty_labels_returns_none(self):
        worker = self.new_worker()
        self.assertIsNone(worker.write_embeddings(
            self.tmp / "x.npz", [], np.ones((1, 256), dtype=np.float32)))

    def test_none_centroids_returns_none(self):
        worker = self.new_worker()
        self.assertIsNone(worker.write_embeddings(self.tmp / "x.npz",
                                                  ["A"], None))
        self.assertFalse((self.tmp / "x.npz").exists())

    def test_write_embeddings_hard_fails_on_the_oracle_path(self):
        """Oracle 路径（`centroids is None`）**一条都不许写**，连全零行也不行。

        这是上一条的强版本：只断言「不抛」是不够的 —— 把 `None` 偷换成
        `[[], [], ...]`（形状看起来像那么回事的补位数据）也必须被拦住，
        否则声纹库里会混进一批零向量。
        """
        worker = self.new_worker()
        for bogus in (None, [[]], [[], []], [0.0, 0.0], "x", 5):
            p = self.tmp / ("oracle_%s.npz" % abs(hash(repr(bogus))))
            self.assertIsNone(worker.write_embeddings(p, ["A", "B"], bogus),
                              "形状异常的质心被写进去了：%r" % (bogus,))
            self.assertFalse(p.exists(), "不该落盘：%r" % (bogus,))

    def test_write_failure_returns_none_and_prints_no_embeddings_line(self):
        """写盘失败时不许返回一个并不存在的路径。

        旧实现失败后仍然 `return dst`，`_run_and_write` 就照协议打印
        `EMBEDDINGS=<那个不存在的文件>` —— 调用方以为拿到了 sidecar，
        读回来是空的，而且没有任何报错。
        """
        worker = self.new_worker()
        blocked = self.tmp / "blocked"
        blocked.write_text("这是个文件，不是目录", encoding="utf-8")
        dst = blocked / "e.npz"          # 父路径是文件 -> 建目录/写盘必失败
        got = worker.write_embeddings(dst, ["A"], np.ones((1, 256)))
        self.assertIsNone(got)
        self.assertFalse(dst.exists())

        out_json = self.tmp / "d.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = worker._run_and_write(
                _FakeDiarization(_turns(1)), np.ones((1, 256)), out_json,
                blocked / "e.npz")
        text = buf.getvalue()
        self.assertEqual(code, 0, "sidecar 写不出来不该让分割失败")
        self.assertIn("RESULT=", text, "RESULT= 必须照常打印")
        self.assertNotIn("EMBEDDINGS=", text,
                         "为一个没写出来的文件打印了 EMBEDDINGS=：%r" % text)

    def test_no_leftover_tmp_files(self):
        worker = self.new_worker()
        worker.write_embeddings(self.tmp / "x.npz", ["A"],
                                np.ones((1, 256), dtype=np.float32))
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["a.wav", "x.npz"])

    def test_does_not_overwrite_good_sidecar_with_a_bad_one(self):
        """失败的一次不许把上一份好 sidecar 抹掉（宁可留着旧的）。"""
        worker = self.new_worker()
        good = self.tmp / "x.npz"
        worker.write_embeddings(good, ["A"], np.ones((1, 256), dtype=np.float32))
        before = good.read_bytes()
        self.assertIsNone(worker.write_embeddings(good, ["A", "B"],
                                                 np.ones((1, 256), dtype=np.float32)))
        self.assertEqual(good.read_bytes(), before)


class TestRowLabelContractIsTheSameOnBothSides(WorkerCase):
    r"""写方（worker）与读方（webui）对「行数 vs 标签数」必须**同一个规则**。

    这个类存在的理由：同一份 sidecar，worker 写出来了（返回路径、打印
    `EMBEDDINGS=`），webui 却整份不信（返回 `{}`）—— 两边规则不同时，
    质心在盘上躺着却永远进不了标注状态，而且**全程没有一句日志**。

    严格相等是两边唯一的公共规则（源码事实见 `write_embeddings` 的注释）。
    """

    def test_sidecar_written_by_worker_is_accepted_by_webui(self):
        import webui
        worker = self.new_worker()
        side = self.tmp / webui.EMBEDDINGS_FILE
        labels = ["SPEAKER_00", "SPEAKER_01"]
        cents = np.stack([unit(1.0, 0.0), unit(0.0, 1.0)]).astype(np.float32)
        self.assertEqual(worker.write_embeddings(side, labels, cents), side)
        self.assertEqual(sorted(webui.load_cluster_embeddings(self.tmp)), labels)

    def test_every_row_count_mismatch_is_rejected_by_both_sides(self):
        """行数 != 标签数：写方不写、读方不信 —— 两个方向都要同规则。"""
        import webui
        worker = self.new_worker()
        labels = ["SPEAKER_00", "SPEAKER_01"]
        for rows in (1, 3, 5):
            with self.subTest(rows=rows):
                side = self.tmp / ("m%d.npz" % rows)
                cents = np.tile(unit(1.0, 0.0), (rows, 1)).astype(np.float32)
                self.assertIsNone(worker.write_embeddings(side, labels, cents),
                                  "worker 放行了 %d 行 / %d 标签" % (rows, len(labels)))
                self.assertFalse(side.exists())
                # 手写一份同形状的文件，模拟「盘上被改坏」：读方也必须整份不信
                np.savez(str(self.tmp / webui.EMBEDDINGS_FILE),
                         labels=np.asarray(labels, dtype="<U32"), centroids=cents)
                self.assertEqual(webui.load_cluster_embeddings(self.tmp), {},
                                 "webui 接受了 %d 行 / %d 标签" % (rows, len(labels)))

    def test_zero_padded_rows_are_equal_count_and_dropped_by_the_reader(self):
        """行数相等、补位那几行全零 —— 这才是实际会出现的退化形态。

        读端（`speaker_registry.extract_centroids`）逐行丢零行；写端**照写**
        （行数对得上，它没有理由拒绝）。这条测试同时钉住两件事：
        零行不会被当成质心用，好行也不会被连累丢掉。
        """
        import webui
        worker = self.new_worker()
        side = self.tmp / webui.EMBEDDINGS_FILE
        labels = ["SPEAKER_00", "SPEAKER_01"]
        cents = np.stack([unit(1.0, 0.0), np.zeros(256)]).astype(np.float32)
        self.assertEqual(worker.write_embeddings(side, labels, cents), side,
                         "行数相等时写端应当照写（零行由读端丢）")
        got = webui.load_cluster_embeddings(self.tmp)
        self.assertEqual(sorted(got), ["SPEAKER_00"])
        self.assertAlmostEqual(float(np.linalg.norm(got["SPEAKER_00"])), 1.0,
                               places=5)


class TestWorkerStdoutProtocol(unittest.TestCase):
    """`main()` 的完整路径：既有 stdout 解析方式必须原样能取到结果。"""

    def setUp(self):
        self._restore = _install_fake_modules()
        self.addCleanup(self._restore)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _run_main(self, extra_argv, result):
        worker = _load_worker()
        worker._load_wav = lambda _p: {"waveform": None, "sample_rate": 16000,
                                       "uri": "a"}
        # 只换「加载模型」这一步，别动 main() 的其它真实逻辑
        worker._load_pipeline = lambda model, device: _FakePipeline(result)

        src = self.tmp / "a.wav"
        src.write_bytes(b"RIFF")
        out = self.tmp / "d.json"
        argv = ["diarize_worker.py", str(src), str(out)] + extra_argv
        saved = sys.argv
        sys.argv = argv
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = worker.main()
        finally:
            sys.argv = saved
        return code, buf.getvalue(), out

    def test_stdout_protocol_unchanged(self):
        code, text, out = self._run_main([], (_FakeDiarization(_turns()), None))
        self.assertEqual(code, 0)
        # 既有解析方式：逐行找 RESULT= 前缀
        got = [ln for ln in text.splitlines() if ln.startswith("RESULT=")]
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0], "RESULT=%s" % out)
        self.assertTrue(out.is_file())
        self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))), 2)

    def test_cli_flag_enables_sidecar_and_appends_a_line(self):
        emb = self.tmp / "e.npz"
        code, text, out = self._run_main(
            ["--embeddings-out", str(emb)],
            (_FakeDiarization(_turns()), np.ones((2, 256), dtype=np.float32)))
        self.assertEqual(code, 0)
        lines = [ln for ln in text.splitlines()
                 if ln.startswith("RESULT=") or ln.startswith("EMBEDDINGS=")]
        self.assertEqual(len(lines), 2, lines)
        self.assertEqual(lines[0], "RESULT=%s" % out)
        self.assertEqual(lines[1], "EMBEDDINGS=%s" % emb)
        self.assertTrue(emb.is_file())


# ---------------------------------------------------------------------------
# jisheng.run_diarization 透传
# ---------------------------------------------------------------------------

class _CapturingProc:
    """替身 Popen：按 target 写结果 JSON，并把 argv 记下来。"""

    def __init__(self, argv, returncode=0, stderr_text=""):
        self.argv = list(argv)
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

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode


class TestRunDiarizationPassthrough(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.wav = self.tmp / "a.wav"
        self.wav.write_bytes(b"RIFF")
        self.seen = []

    def _call(self, out_json, **kwargs):
        """在补丁生效期间调一次 run_diarization，**无论如何**都还原。"""
        self.seen = []
        orig = (A.subprocess.Popen, A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                A.resolve_diarize_model)

        def fake_popen(argv, **kw):  # noqa: ARG001
            proc = _CapturingProc(argv)
            self.seen.append(proc)
            return proc

        A.subprocess.Popen = fake_popen
        A.DIARIZE_PYTHON = Path(sys.executable)
        A.DIARIZE_WORKER = Path(sys.executable)
        A.resolve_diarize_model = lambda m: "stub-model"   # noqa: ARG005
        try:
            return A.run_diarization(self.wav, self._args(),
                                     out_json=out_json, **kwargs)
        finally:
            (A.subprocess.Popen, A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
             A.resolve_diarize_model) = orig

    @staticmethod
    def _args():
        return types.SimpleNamespace(diarize_model=None, device="cpu",
                                     num_speakers=None)

    def test_run_diarization_keeps_returning_a_list(self):
        """老调用方式（不传 out_embeddings）不许破：仍然返回 list|None。"""
        payload = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        out = self.tmp / "dia.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        got = self._call(out)
        self.assertIsInstance(got, list)
        self.assertEqual(got, payload)
        self.assertNotIn("--embeddings-out", self.seen[0].argv)

    def test_out_embeddings_is_passed_to_the_worker(self):
        payload = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        out = self.tmp / "dia.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        emb = self.tmp / "dia.embeddings.npz"
        got = self._call(out, out_embeddings=emb)
        self.assertEqual(got, payload)
        argv = self.seen[0].argv
        self.assertIn("--embeddings-out", argv)
        self.assertEqual(argv[argv.index("--embeddings-out") + 1], str(emb))

    def test_out_embeddings_none_does_not_add_the_flag(self):
        out = self.tmp / "dia.json"
        out.write_text("[]", encoding="utf-8")
        self._call(out, out_embeddings=None)
        self.assertNotIn("--embeddings-out", self.seen[0].argv)


# ---------------------------------------------------------------------------
# WebUI：sidecar -> speakers.json
# ---------------------------------------------------------------------------

DIA = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
       {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
       {"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"}]

SEGS = [{"start": 0.0, "end": 4.0, "text": "第一句", "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 15.0, "text": "第二句", "speaker": "SPEAKER_01"},
        {"start": 20.0, "end": 26.0, "text": "第三句", "speaker": "SPEAKER_00"}]


def unit(*comps, dim=256):
    v = np.zeros(dim, dtype=np.float64)
    for i, c in enumerate(comps):
        v[i] = float(c)
    return v / float(np.linalg.norm(v))


class WebuiCase(unittest.TestCase):
    """每个测试都有**自己的**临时角色库根 —— 绝不碰项目里的真实库。"""

    def setUp(self):
        import webui
        self.webui = webui
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.workdir = self.root / "work" / "EP01"
        self.workdir.mkdir(parents=True)
        (self.workdir / "diarization.json").write_text(
            json.dumps(DIA, ensure_ascii=False), encoding="utf-8")
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in SEGS) + "\n",
            encoding="utf-8")

        self._work_dir = webui.WORK_DIR
        self._out_root = webui.OUTPUT_ROOT
        self._jobs = webui.JOBS
        webui.WORK_DIR = self.root / "work"
        webui.OUTPUT_ROOT = self.root / "output"
        webui.JOBS = {}
        self.addCleanup(self._restore_webui)

        self.reg_root = self.root / "reg"
        # 用例期间指到临时根；跑完恢复到**本模块的**隔离根（不是 None ——
        # None 会把后面不继承本类的用例指回真实库，见 setUpModule）。
        self._reg_prev = sr.default_root()
        sr.set_default_root(str(self.reg_root))
        self.addCleanup(sr.set_default_root, self._reg_prev)

    def _restore_webui(self):
        self.webui.WORK_DIR = self._work_dir
        self.webui.OUTPUT_ROOT = self._out_root
        self.webui.JOBS = self._jobs

    def make_job(self, jid="j1"):
        out = self.root / "output"
        out.mkdir(parents=True, exist_ok=True)
        self.webui.JOBS[jid] = {
            "id": jid, "media": str(self.root / "EP01.mkv"),
            "work": str(self.workdir), "out_dir": str(out),
            "logs": [], "status": "done", "stage": "完成"}
        return self.webui.JOBS[jid]

    def write_sidecar(self, labels=("SPEAKER_00", "SPEAKER_01"), cents=None,
                      name="embeddings.npz"):
        cents = (np.stack([unit(1.0, 0.0), unit(0.0, 1.0)])
                 if cents is None else np.asarray(cents, dtype=np.float32))
        p = self.workdir / name
        np.savez(str(p), labels=np.asarray(list(labels), dtype="<U32"),
                 centroids=cents.astype(np.float32))
        return p


class TestWebuiStoresEmbedding(WebuiCase):
    def test_webui_stores_embedding_in_speakers_json(self):
        self.make_job()
        self.write_sidecar()
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00", "SPEAKER_01"])
        sl.save_labels(self.workdir, labels)

        got = self.webui._store_cluster_embeddings(self.workdir, labels)

        self.assertEqual(sorted(got), ["SPEAKER_00", "SPEAKER_01"])
        back = sl.load_labels(self.workdir)
        for cluster in ("SPEAKER_00", "SPEAKER_01"):
            vec = back["clusters"][cluster]["embedding"]
            self.assertEqual(len(vec), 256)
            # speaker_registry 的契约是「存进去的一律是单位向量」（pyannote
            # 原始输出范数 2.2~2.6）。写进 speakers.json 的也必须已经归一化，
            # 否则面板拿到的是一个量纲不对的「相似度」。
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_zero_and_nan_rows_are_not_stored(self):
        """补位零行 / NaN 行不许进 speakers.json（那里放的是可比的方向）。"""
        self.make_job()
        zeros = np.zeros((2, 256), dtype=np.float32)
        self.write_sidecar(labels=("SPEAKER_00", "SPEAKER_01"), cents=zeros)
        self.assertEqual(self.webui.load_cluster_embeddings(self.workdir), {})
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00", "SPEAKER_01"])
        sl.save_labels(self.workdir, labels)
        self.assertEqual(self.webui._store_cluster_embeddings(self.workdir, labels), {})

        bad = np.ones((2, 256), dtype=np.float32)
        bad[1, 0] = np.nan
        self.write_sidecar(labels=("SPEAKER_00", "SPEAKER_01"), cents=bad)
        # 只有坏行被丢，好行照常可用（一人坏不能连累另一人）
        self.assertEqual(sorted(self.webui.load_cluster_embeddings(self.workdir)),
                         ["SPEAKER_00"])

    def test_missing_sidecar_leaves_labels_alone(self):
        self.make_job()
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00"])
        sl.save_labels(self.workdir, labels)
        got = self.webui._store_cluster_embeddings(self.workdir, labels)
        self.assertEqual(got, {})
        back = sl.load_labels(self.workdir)
        self.assertNotIn("embedding", back["clusters"]["SPEAKER_00"])

    def test_labels_longer_than_centroids_are_skipped(self):
        """补位零行场景：对齐不可靠 → 一个都不写（宁可纯手动命名）。

        两道闸都要断言：`extract_centroids` 丢掉补位行是一道，`load_cluster_embeddings`
        的「行数 != 标签数 → 整份不信」是另一道。只测后者能过的话，
        前者被改成「截断照用」就没人拦得住了。
        """
        self.make_job()
        self.write_sidecar(labels=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"),
                           cents=np.stack([unit(1.0, 0.0), unit(0.0, 1.0)]))
        self.assertEqual(self.webui.load_cluster_embeddings(self.workdir), {})
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        sl.save_labels(self.workdir, labels)
        got = self.webui._store_cluster_embeddings(self.workdir, labels)
        self.assertEqual(got, {})

    def test_row_count_mismatch_rejects_the_whole_sidecar(self):
        """行数与标签数不等 → 整份不信。

        `speaker_registry.extract_centroids` 对「行数多于标签数」是**截断**（多出来的
        行丢掉），那在 worker 刚写完时是安全的。但盘上的文件可能被改坏/截断，
        而被截断过的文件看起来正好像「行数少了」—— 一旦接受，`labels[i]` 与第 i 行
        就不再对应，声纹会配到另一个人的名字上。所以这里一律要求严格相等。
        """
        self.make_job()
        self.write_sidecar(labels=("SPEAKER_00",),
                           cents=np.stack([unit(1.0), unit(0.0, 1.0)]))
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00"])
        got = self.webui._store_cluster_embeddings(self.workdir, labels)
        self.assertEqual(got, {})
        self.assertEqual(self.webui.load_cluster_embeddings(self.workdir), {})

    def test_corrupt_sidecar_does_not_raise(self):
        self.make_job()
        (self.workdir / "embeddings.npz").write_bytes(b"not an npz")
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00"])
        self.assertEqual(
            self.webui._store_cluster_embeddings(self.workdir, labels), {})

    def test_worker_stdout_lines_are_read_back_by_run_diarization(self):
        """端到端（打桩子进程）：worker 的 EMBEDDINGS= 行指向的 sidecar 能被读回。"""
        self.make_job()
        emb_path = self.workdir / self.webui.EMBEDDINGS_FILE
        cents = np.stack([unit(1.0, 0.0), unit(0.0, 1.0)]).astype(np.float32)
        labels = ("SPEAKER_00", "SPEAKER_01")

        proc = _StubWorkerProc(
            [sys.executable, "diarize_worker.py", self.workdir / "dia.json"],
            DIA, emb_path, labels, cents)
        # 全程 try/finally 就地还原：addCleanup 在测试方法结束后才跑，
        # 中间任何异常都会把 Popen 的补丁泄漏给后面所有测试模块
        # （症状是别的模块报 "'_StubWorkerProc' object does not support the
        #  context manager protocol"）。补丁点越少越好、还原越早越好。
        orig = (A.subprocess.Popen, A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
                A.resolve_diarize_model)
        A.subprocess.Popen = lambda argv, **kw: proc   # noqa: ARG005
        A.DIARIZE_PYTHON = Path(sys.executable)
        A.DIARIZE_WORKER = Path(sys.executable)
        A.resolve_diarize_model = lambda m: "stub"      # noqa: ARG005
        try:
            got = A.run_diarization(
                self.workdir / "raw16k.wav",
                types.SimpleNamespace(diarize_model=None, device="cpu",
                                      num_speakers=None),
                out_json=self.workdir / "diarization.json",
                out_embeddings=emb_path)
        finally:
            (A.subprocess.Popen, A.DIARIZE_PYTHON, A.DIARIZE_WORKER,
             A.resolve_diarize_model) = orig

        self.assertEqual(got, DIA)
        self.assertTrue(emb_path.is_file())
        stdout = proc.stdout.getvalue()
        self.assertIn("RESULT=", stdout)
        self.assertIn("EMBEDDINGS=", stdout)
        # 再走 WebUI 那一步：sidecar -> speakers.json
        back = sl.load_labels(self.workdir)
        sl.ensure_clusters(back, list(labels))
        stored = self.webui._store_cluster_embeddings(self.workdir, back)
        self.assertEqual(sorted(stored), list(labels))


class _StubWorkerProc:
    """替身 Popen：把结果与 sidecar 写到目标路径，并按新协议打印两行。"""

    def __init__(self, argv, dia, emb_path, labels, cents):
        argv = [str(a) for a in argv]
        self.argv = argv
        out = Path(argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dia, ensure_ascii=False), encoding="utf-8")
        np.savez(str(emb_path), labels=np.asarray(list(labels), dtype="<U32"),
                 centroids=np.asarray(cents, dtype=np.float32))
        self.stdout = _NoCloseStringIO(
            "RESULT=%s\nEMBEDDINGS=%s\n" % (out, emb_path))
        self.stderr = io.StringIO("")
        self.returncode = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode


class _NoCloseStringIO(io.StringIO):
    """`run_diarization` 在 finally 里会 close 管道；测试还要回头读它记下的行。"""

    closed = False

    def close(self):
        pass


class TestNamingEnrolls(WebuiCase):
    def test_naming_enrolls_into_the_registry(self):
        """命名 = 确认：库里出现该角色，且 speakers.json 记下匹配结果。

        这里刻意**不**先跑 api_speakers：真实面板打开顺序不保证，而 api_assign
        只有 `_api_open` 那条路（它不落盘聚类记录）。命名前状态表里可能一个
        聚类都没有，入库这一步必须自己把记录建出来。
        """
        self.make_job()
        self.write_sidecar()
        obj, code = self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                                           "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200)

        rows = sr.list_registry(str(self.reg_root))
        self.assertEqual([(r["work"], r["page"]) for r in rows],
                         [("无职转生", "艾丽丝")])
        self.assertEqual(rows[0]["n_enroll"], 1)
        self.assertGreaterEqual(rows[0]["n_vectors"], 1)
        vecs = sr.load_vectors(str(self.reg_root))
        self.assertIn(sr.page_centroid_key("无职转生", "艾丽丝"), vecs)

        labels = sl.load_labels(self.workdir)
        self.assertEqual(labels["clusters"]["SPEAKER_00"]["name"], "艾丽丝")
        self.assertIn("match_at_enroll", labels["clusters"]["SPEAKER_00"])

    def test_naming_without_work_name_does_not_touch_the_registry(self):
        """没有作品名 = 库不知道这是哪部作品：不许凭空造一个作品条目。"""
        self.make_job()
        self.write_sidecar()
        obj, code = self.webui.api_assign({"job": "j1",
                                           "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200)
        self.assertEqual(sr.list_registry(str(self.reg_root)), [])
        self.assertFalse((self.reg_root / sr.REGISTRY_FILE).exists())
        # 也不许在标注状态里留一份 `match_at_enroll`：那会让人以为已经入库了
        # （面板上就是「已入库，自动匹配：艾丽丝」）。判据不能只看库 ——
        # `enroll(work="")` 自己会拒绝写库，所以库是空的并不说明没走入库分支。
        rec = sl.load_labels(self.workdir)["clusters"]["SPEAKER_00"]
        self.assertEqual(rec.get("name"), "艾丽丝")
        self.assertNotIn("match_at_enroll", rec)

    def test_naming_without_embedding_still_saves_the_name(self):
        """拿不到质心（无 sidecar）→ 名字照存，库里只登记角色、不写向量。"""
        self.make_job()
        obj, code = self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                                           "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200)
        rows = sr.list_registry(str(self.reg_root))
        self.assertEqual([(r["work"], r["page"]) for r in rows],
                         [("无职转生", "艾丽丝")])
        self.assertEqual(rows[0]["n_vectors"], 0)
        self.assertEqual(sl.cluster_name(sl.load_labels(self.workdir),
                                         "SPEAKER_00"), "艾丽丝")

    def test_two_clusters_enroll_under_their_own_names(self):
        """两个聚类各自按**自己的**质心入库 —— 串了就是「声纹配到别人名字上」。"""
        self.make_job()
        # SPEAKER_00 -> e0，SPEAKER_01 -> e1，两个方向正交，串了必然看得出来
        self.write_sidecar(labels=("SPEAKER_00", "SPEAKER_01"),
                           cents=np.stack([unit(1.0, 0.0), unit(0.0, 1.0)]))
        obj, code = self.webui.api_assign(
            {"job": "j1", "work_name": "无职转生",
             "names": {"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪"}})
        self.assertEqual(code, 200)

        rows = {(r["work"], r["page"]): r for r in sr.list_registry(str(self.reg_root))}
        self.assertEqual(set(rows),
                         {("无职转生", "鲁迪"), ("无职转生", "艾丽丝")})
        for rec in rows.values():
            self.assertEqual(rec["n_enroll"], 1)
            # 一次入库落两条：`__centroid`（70/30 滚动）+ `__seg000`（本次质心）
            self.assertEqual(rec["n_vectors"], 2)

        vecs = sr.load_vectors(str(self.reg_root))
        alice = vecs[sr.page_centroid_key("无职转生", "艾丽丝")]
        rudy = vecs[sr.page_centroid_key("无职转生", "鲁迪")]
        self.assertAlmostEqual(float(alice[0]), 1.0, places=5)
        self.assertAlmostEqual(float(alice[1]), 0.0, places=5)
        self.assertAlmostEqual(float(rudy[0]), 0.0, places=5)
        self.assertAlmostEqual(float(rudy[1]), 1.0, places=5)
        # 两个角色的向量必须真的不同（不是同一份被写了两遍）
        self.assertLess(abs(sr.cosine(alice, rudy)), 1e-6)

    def test_renaming_enrolls_the_new_name(self):
        self.make_job()
        self.write_sidecar()
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_00": "艾丽丝"}})
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_00": "艾莉丝"}})
        rows = sr.list_registry(str(self.reg_root))
        self.assertIn(("无职转生", "艾莉丝"), [(r["work"], r["page"]) for r in rows])

    def test_reassign_does_not_enroll(self):
        """单句改判不是「确认整条聚类」，不该往库里写向量。"""
        self.make_job()
        self.write_sidecar()
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_00": "艾丽丝"}})
        before = sr.list_registry(str(self.reg_root))[0]["n_enroll"]
        self.webui.api_reassign({"job": "j1", "work_name": "无职转生",
                                 "index": 1, "to": "SPEAKER_00"})
        after = sr.list_registry(str(self.reg_root))[0]["n_enroll"]
        self.assertEqual(before, after)

    def test_assign_still_writes_subtitles(self):
        """入库这一步不能把「重写字幕」这条主路径挤掉。"""
        self.make_job()
        self.write_sidecar()
        obj, code = self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                                           "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200)
        self.assertTrue(obj.get("rewritten"))
        srt = Path(obj["srt"]).read_text(encoding="utf-8")
        self.assertIn("[艾丽丝] 第一句", srt)


class TestStoreEmbeddingsKeepsConcurrentNaming(WebuiCase):
    """写质心这一步不许把并发命名回退掉（lost update）。"""

    def test_stale_snapshot_does_not_roll_back_a_saved_name(self):
        """run_job 拿旧快照写 embedding 时，期间落盘的名字必须还在。

        真实交错：`run_job` 先 bootstrap（读-改-写）拿到一份快照 → 用户此刻
        在面板上命名（api_assign 持锁读-改-写、字幕已按新名字重写、库里已入库）
        → run_job 继续用**那份旧快照**写 embedding。
        旧实现直接 `save_labels(old_dict)`，用户刚命的名当场被回退，于是
        speakers.json / 字幕 / 声纹库三处互相矛盾。
        """
        self.make_job()
        self.write_sidecar()
        snapshot = sl.bootstrap(self.workdir, "j1",
                                str(self.root / "EP01.mkv"),
                                ["SPEAKER_00", "SPEAKER_01"])
        obj, code = self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                                           "names": {"SPEAKER_01": "鲁迪"}})
        self.assertEqual(code, 200)
        self.assertIn("[鲁迪]", Path(obj["srt"]).read_text(encoding="utf-8"))

        stored = self.webui._store_cluster_embeddings(self.workdir, snapshot)

        self.assertEqual(sorted(stored), ["SPEAKER_00", "SPEAKER_01"])
        back = sl.load_labels(self.workdir)
        self.assertEqual(sl.cluster_name(back, "SPEAKER_01"), "鲁迪",
                         "并发命名被旧快照覆盖了")
        self.assertEqual(sl.get_work_name(back), "无职转生")
        # 旧快照里没有的键也不能被合并过程弄丢
        self.assertIn("SPEAKER_00", back["clusters"])

    def test_store_embeddings_waits_for_the_same_label_lock(self):
        """写质心必须持该 work 的标注锁：锁被别人持着时它必须等。

        用真的 `webui._label_lock`（api_assign / _wipe_annotation 用的就是它）
        做确定性交错：主线程先持锁，再从另一个线程调 `_store_cluster_embeddings`。
        不持锁的实现会当场把「读 sidecar」走完 —— 那一刻的交错正是 lost update。
        """
        self.make_job()
        self.write_sidecar()
        labels = sl.load_labels(self.workdir)

        lock = self.webui._label_lock(self.workdir)
        entered = threading.Event()
        done = threading.Event()
        errors = []
        orig = self.webui.load_cluster_embeddings

        def probe(work):
            entered.set()
            return orig(work)

        self.addCleanup(setattr, self.webui, "load_cluster_embeddings", orig)
        self.webui.load_cluster_embeddings = probe

        def store():
            try:
                self.webui._store_cluster_embeddings(self.workdir, labels)
            except Exception as exc:      # noqa: BLE001 - 就是要证明它不抛
                errors.append(repr(exc))
            finally:
                done.set()

        t = threading.Thread(target=store, daemon=True)
        with lock:
            t.start()
            self.assertFalse(entered.wait(timeout=1.0),
                             "主线程还持着锁，写质心就已经读到 sidecar 了——没有持锁")
            self.assertFalse(done.is_set())
        self.assertTrue(done.wait(timeout=5), "释放锁后写质心仍未完成")
        self.assertTrue(entered.is_set())
        self.assertEqual(errors, [])
        self.assertIn("embedding",
                      sl.load_labels(self.workdir)["clusters"]["SPEAKER_00"])


class TestBadEmbeddingDoesNotPoisonTheRegistry(WebuiCase):
    """state 是给人手改的：坏 embedding 不许改库级 dim，也不许吞掉角色登记。"""

    def _seed_bad_embedding(self, bad):
        (self.workdir / "speakers.json").write_text(json.dumps(
            {"version": 1, "job": "j1", "media": "", "work_name": "",
             "source_kind": "separated",
             "clusters": {"SPEAKER_00": {"name": "", "embedding": bad},
                          "SPEAKER_01": {"name": ""}},
             "segment_overrides": {}, "updated_at": ""}, ensure_ascii=False),
            encoding="utf-8")

    def test_short_vector_does_not_change_the_library_dim(self):
        """3 维向量：只登记角色、不写向量，库级 dim 必须还是 256。

        旧行为：`enroll` 从向量量出 dim=3 并写进库级 `dim`，于是**全部** 256 维
        向量当场不再参与匹配（`_page_vectors` 按全局 dim 过滤）—— 一次手改
        就让整个库失声，而且没有任何报错。
        """
        self.make_job()
        # 先让库里有一个 256 维的既有角色，dim 才有「被改坏」的余地
        good = np.zeros(256, dtype=np.float64)
        good[0] = 1.0
        sr.enroll(str(self.reg_root), work="别的作品", page="艾丽丝",
                  vector=good, model=self.webui.EMBEDDING_MODEL_TAG)
        self.assertEqual(sr.load_registry(str(self.reg_root))["dim"], 256)

        self._seed_bad_embedding([0.1, 0.2, 0.3])
        obj, code = self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                                           "names": {"SPEAKER_00": "艾丽丝"}})

        self.assertEqual(code, 200)
        self.assertEqual(sl.cluster_name(sl.load_labels(self.workdir), "SPEAKER_00"),
                         "艾丽丝")
        self.assertEqual(sr.load_registry(str(self.reg_root))["dim"], 256,
                         "坏向量把库级 dim 改掉了——全库向量一起失声")
        pages = [(r["work"], r["page"], r["n_vectors"])
                 for r in sr.list_registry(str(self.reg_root))]
        self.assertIn(("无职转生", "艾丽丝", 0), pages,
                      "坏向量把角色登记也一起吞了")
        self.assertFalse(sr.load_vectors(str(self.reg_root)).get(
            sr.page_centroid_key("无职转生", "艾丽丝")) is not None)

    def test_non_numeric_embedding_still_registers_the_role(self):
        """字符串 / None / NaN / 字典：优雅退化 = 200、名字照存、角色照登记、无向量。"""
        for bad in ("not-a-vector", None, [float("nan")] * 256, {"a": 1}):
            with self.subTest(bad=repr(bad)[:24]):
                # 每个形态都要一份干净的状态与库，否则上一个形态的登记会串进来
                shutil.rmtree(self.workdir, ignore_errors=True)
                self.workdir.mkdir(parents=True)
                (self.workdir / "diarization.json").write_text(
                    json.dumps(DIA, ensure_ascii=False), encoding="utf-8")
                (self.workdir / "segments.jsonl").write_text(
                    "\n".join(json.dumps(s, ensure_ascii=False) for s in SEGS) + "\n",
                    encoding="utf-8")
                self.make_job()
                sr.set_default_root(str(self.root / ("reg_" + str(abs(hash(repr(bad)))))))
                self._seed_bad_embedding(bad)
                try:
                    obj, code = self.webui.api_assign(
                        {"job": "j1", "work_name": "无职转生",
                         "names": {"SPEAKER_00": "艾丽丝"}})
                    self.assertEqual(code, 200)
                    self.assertNotIn("error", obj)
                    self.assertEqual(
                        sl.cluster_name(sl.load_labels(self.workdir), "SPEAKER_00"),
                        "艾丽丝")
                    pages = [(r["work"], r["page"], r["n_vectors"])
                             for r in sr.list_registry(str(sr.default_root()))]
                    self.assertIn(("无职转生", "艾丽丝", 0), pages)
                finally:
                    sr.set_default_root(str(self.reg_root))


class TestRediarizeKeepsFreshEmbeddings(WebuiCase):
    """在检查点上重新分割：新质心必须活下来，旧质心必须死透。"""

    DIA2 = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
            {"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"}]

    def _run_redo(self, opts=None):
        """跑真实的分割收尾代码，语音模型与分割子进程使用替身。

        前置状态刻意做成「上一次分割失败留下的检查点」：音频已抽好、state.json
        记着同一套参数，但**没有** diarization.json —— 于是这次 `same=True`
        且 `dia_path` 不存在，走「在检查点上重新分割」那条路。
        """
        media = self.root / "EP01.mkv"
        media.write_bytes(b"fake media")
        (self.workdir / "raw16k.wav").write_bytes(b"RIFF")
        # WebuiCase.setUp 会先落一份 diarization.json（别的用例要用它）。
        # 这里要的正是「还没有分割结果」的前置状态：删掉它，才会走重新分割那条路。
        (self.workdir / "diarization.json").unlink(missing_ok=True)
        opts = opts or {"diarize": True, "separate": False, "mode": "asr"}
        self.webui.save_state(self.workdir, {
            "media": str(media), "media_size": media.stat().st_size,
            "media_mtime": media.stat().st_mtime,
            "opts_key": self.webui._opts_key(opts), "opts": opts,
            "duration": 30.0, "status": "done", "total_chunks": 1})
        job = {"id": "j1", "media": str(media),
               "out_dir": str(self.root / "output"),
               "logs": [], "status": "queued", "stage": "", "opts": opts}

        seen = {"saw_stale_sidecar": None}
        def stop_before_recognition(*args):
            job["stop"] = True

        def fake_diarization(wav, args, out_json=None, on_proc=None,  # noqa: ARG001
                             should_stop=None, timeout=1800,
                             out_embeddings=None):
            side = Path(out_embeddings)
            seen["saw_stale_sidecar"] = side.is_file()
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(self.DIA2, ensure_ascii=False),
                                      encoding="utf-8")
            # 本环境没有 torch / pyannote，worker 模块 import 不动；这里直接按
            # 它的 sidecar 契约落盘（数组名与形状由
            # TestRowLabelContractIsTheSameOnBothSides 单独钉住）。
            side.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(side), labels=np.asarray(["SPEAKER_00", "SPEAKER_01"],
                                                  dtype="<U64"),
                     centroids=np.stack([np.ones(256),
                                         np.full(256, 0.5)]).astype(np.float32))
            return list(self.DIA2)

        with (patch.object(A, "run_diarization", side_effect=fake_diarization),
              patch.object(self.webui, "_get_model", side_effect=stop_before_recognition)):
            self.webui.run_job(job)
        return job, seen

    def test_first_time_redo_writes_the_fresh_centroids(self):
        """第一次分割失败后按同一套参数重跑：质心必须写进标注状态。"""
        job, seen = self._run_redo()

        labels = sl.load_labels(self.workdir)
        got = {c: rec.get("embedding") for c, rec in labels["clusters"].items()
               if isinstance(rec, dict) and rec.get("embedding") is not None}
        side = self.workdir / self.webui.EMBEDDINGS_FILE
        self.assertEqual(sorted(got), ["SPEAKER_00", "SPEAKER_01"],
                         "重新分割后质心没进标注状态（sidecar 在=%s）" % side.is_file())
        for vec in got.values():
            self.assertEqual(len(vec), 256)
        self.assertTrue(side.is_file())
        self.assertTrue([ln for ln in job["logs"] if "质心" in ln],
                        "一句日志都没有：%r" % (job["logs"],))

    def test_stale_sidecar_is_gone_before_the_new_one_is_written(self):
        """旧质心必须在**新分割写入之前**就被清掉。

        顺序是判据，不只是「最后有没有文件」：留在新质心写盘之后的 wipe 会把
        刚写好的 sidecar 一起删掉（而且静默）；留到新分割之后的旧 sidecar 则会
        把声纹配到另一个人的名字上。两种都不能接受。
        """
        _job, seen = self._run_redo()
        self.assertIs(seen["saw_stale_sidecar"], False,
                      "新质心写盘时旧 sidecar 还在：wipe 没安排在写盘之前")

    def test_reused_checkpoint_with_diarization_json_keeps_the_sidecar(self):
        """复用已有 diarization.json（没重新分割）时，sidecar 与那些聚类仍然对应。

        这一条与上一条互为边界：wipe 只能挂在「即将重新分割」这个分支上。
        挂成无条件的「分割阶段收尾就清」，复用检查点时会把一份**仍然有效**的
        sidecar 删掉；反过来把 wipe 挪到写新 sidecar 之后，上一条就会红。
        """
        media = self.root / "EP01.mkv"
        media.write_bytes(b"fake media")
        (self.workdir / "raw16k.wav").write_bytes(b"RIFF")
        (self.workdir / "diarization.json").write_text(
            json.dumps(self.DIA2, ensure_ascii=False), encoding="utf-8")
        self.write_sidecar(name=self.webui.EMBEDDINGS_FILE)
        self.assertTrue((self.workdir / self.webui.EMBEDDINGS_FILE).is_file())
        opts = {"diarize": True, "separate": False, "mode": "asr"}
        self.webui.save_state(self.workdir, {
            "media": str(media), "media_size": media.stat().st_size,
            "media_mtime": media.stat().st_mtime,
            "opts_key": self.webui._opts_key(opts), "opts": opts,
            "duration": 30.0, "status": "done", "total_chunks": 1})
        job = {"id": "j1", "media": str(media), "out_dir": str(self.root / "output"),
               "logs": [], "status": "queued", "stage": "", "opts": opts}

        orig = self.webui.stop_requested
        calls = {"n": 0}

        def fake_stop(_job):
            calls["n"] += 1
            return calls["n"] > 2          # 走完分割阶段即收工

        self.webui.stop_requested = fake_stop
        try:
            self.webui.run_job(job)
        finally:
            self.webui.stop_requested = orig

        self.assertTrue((self.workdir / self.webui.EMBEDDINGS_FILE).is_file(),
                        "复用旧分割时把仍然有效的质心删掉了")
        self.assertIn("复用已完成的说话人分割结果",
                      "\n".join(job["logs"]))


class TestEnrollOrderingAndIdempotence(WebuiCase):
    """三条「破坏了也没人发现」的规则，逐条钉住。

    这些断言来自变异实验（把规则破坏掉，全量测试仍然 0 红）—— 说明规则本身
    没有测试守着。每条断言都对应一个能被单独破坏的行为，注释写明**为什么**
    这个行为重要。
    """

    def test_match_is_computed_before_enrolling_not_after(self):
        """`match` 必须是「入库前」的印象，不能是入库后的自己匹配自己。

        入库之后这个聚类自己就成了库里的角色：它必然以 1.0 的分数匹配到自己，
        `match` 就成了一句永远成立、毫无信息的废话（面板会显示
        「自动匹配：艾丽丝 100%」，而那正是用户**刚手工输入**的名字）。
        判据用 `auto`：它是匹配到的那个人；入库后匹配时它恒等于当前聚类的新名字。
        """
        self.make_job()
        # 先给**同一个作品的另一个角色**入库，让库里有一个真能匹配上的对象
        # （匹配只在同一作品内比：别的作品的向量不该拿来配这一集）
        prior = np.zeros(256, dtype=np.float64)
        prior[0] = 1.0
        sr.enroll(str(self.reg_root), work="无职转生", page="艾丽丝",
                  vector=prior, model=self.webui.EMBEDDING_MODEL_TAG)
        self.write_sidecar(cents=np.stack([unit(1.0, 0.0), unit(0.0, 1.0)]))

        obj, code = self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                                           "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200)
        rec = sl.load_labels(self.workdir)["clusters"]["SPEAKER_00"]
        match = rec.get("match_at_enroll") or {}
        self.assertAlmostEqual(float(match.get("score")), 1.0, places=4)
        self.assertEqual(match.get("auto"), "艾丽丝")

        # 用一个**库里没有**的新名字再走一遍：match 必须老实说「没匹配上」。
        # 入库后才匹配的实现会在这里返回 auto=<刚输入的名字> / score=0.0
        # （0.0 正是 `enroll` 给「首次入库、没有可比对象」记的分）。
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_01": "全新的角色"}})
        fresh = sl.load_labels(self.workdir)["clusters"]["SPEAKER_01"]
        fresh_match = fresh.get("match_at_enroll") or {}
        self.assertIsNone(fresh_match.get("auto"),
                          "首入库的聚类匹配到了 %r —— 入库后才匹配，自己是自己"
                          % (fresh_match.get("auto"),))
        self.assertIsNone(fresh_match.get("top"),
                          "入库前的匹配结果不该有 top：%r" % (fresh_match,))

    def test_second_cluster_in_the_same_batch_does_not_self_match(self):
        """同一批里第二个聚类也不能自己匹配到自己。"""
        self.make_job()
        self.write_sidecar()
        obj, code = self.webui.api_assign(
            {"job": "j1", "work_name": "无职转生",
             "names": {"SPEAKER_00": "艾丽丝", "SPEAKER_01": "鲁迪"}})
        self.assertEqual(code, 200)
        clusters = sl.load_labels(self.workdir)["clusters"]
        for cluster in ("SPEAKER_00", "SPEAKER_01"):
            match = clusters[cluster].get("match_at_enroll") or {}
            self.assertIsNone(match.get("auto"),
                              "%s 的 match_at_enroll 是自己匹配自己：%r" % (cluster, match))
            self.assertIsNone(match.get("score"),
                              "%s 的 match_at_enroll 是自己匹配自己：%r" % (cluster, match))

    def test_saving_the_same_names_twice_does_not_enroll_twice(self):
        """名字没变就不是一次新的确认：不许重复入库。

        重复入库会把 70/30 滚动质心往同一批向量上反复加权（越滚越偏），
        并把 `n_enroll` 刷成假数字；面板上「已入库 N 次」也就不再有意义。
        """
        self.make_job()
        self.write_sidecar()
        names = {"SPEAKER_00": "艾丽丝"}
        self.webui.api_assign({"job": "j1", "work_name": "无职转生", "names": names})
        first = self._row("无职转生", "艾丽丝")
        self.assertEqual(first["n_enroll"], 1)

        self.webui.api_assign({"job": "j1", "work_name": "无职转生", "names": names})
        second = self._row("无职转生", "艾丽丝")

        self.assertEqual(second["n_enroll"], first["n_enroll"],
                         "同名再存了一次却被当成新确认：n_enroll %d -> %d"
                         % (first["n_enroll"], second["n_enroll"]))
        self.assertEqual(second["n_vectors"], first["n_vectors"],
                         "同名再存了一次却多写了向量：n_vectors %d -> %d"
                         % (first["n_vectors"], second["n_vectors"]))

    def test_renaming_the_same_cluster_does_enroll_again(self):
        """改名是**新的**确认：必须入库（上一条只管「名字没变」）。"""
        self.make_job()
        self.write_sidecar()
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_00": "艾丽丝"}})
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_00": "艾莉丝"}})
        self.assertEqual(self._row("无职转生", "艾莉丝")["n_enroll"], 1)

    def test_old_embedding_is_cleared_when_a_new_sidecar_has_no_vector(self):
        """换一份 sidecar（或重新分割）后，上一轮的质心不许留在聚类上。

        留着的后果不是「显示不准」：`clusters[<id>]["embedding"]` 是入库与匹配
        的原料，一个上一轮的向量会被当成这次聚类的声纹拿去匹配（声纹配到
        另一个人的名字上），而面板完全看不出来。
        """
        self.make_job()
        labels = sl.load_labels(self.workdir)
        sl.ensure_clusters(labels, ["SPEAKER_00", "SPEAKER_01"])
        self.write_sidecar(cents=np.stack([unit(1.0, 0.0), unit(0.0, 1.0)]))
        first = self.webui._store_cluster_embeddings(self.workdir, labels)
        self.assertEqual(sorted(first), ["SPEAKER_00", "SPEAKER_01"])

        # 第二次分割只认出一个说话人：SPEAKER_01 这一轮没有质心
        self.write_sidecar(labels=("SPEAKER_00",), cents=np.stack([unit(0.0, 1.0)]))
        self.webui._store_cluster_embeddings(self.workdir, sl.load_labels(self.workdir))

        back = sl.load_labels(self.workdir)["clusters"]
        self.assertNotIn("embedding", back.get("SPEAKER_01", {}),
                         "上一轮的质心留在新聚类上了")
        self.assertIn("embedding", back["SPEAKER_00"])

    def test_old_match_is_cleared_when_a_new_sidecar_has_no_vector(self):
        """`match_at_enroll` 与 `embedding` 同生共死：没有新质心就一起清掉。"""
        self.make_job()
        self.write_sidecar()
        self.webui.api_assign({"job": "j1", "work_name": "无职转生",
                               "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertIn("match_at_enroll",
                      sl.load_labels(self.workdir)["clusters"]["SPEAKER_00"])

        self.write_sidecar(labels=("SPEAKER_01",), cents=np.stack([unit(0.0, 1.0)]))
        self.webui._store_cluster_embeddings(self.workdir, sl.load_labels(self.workdir))

        back = sl.load_labels(self.workdir)["clusters"]
        self.assertNotIn("match_at_enroll", back["SPEAKER_00"],
                         "上一轮的 match_at_enroll 留在聚类上了")
        self.assertNotIn("embedding", back["SPEAKER_00"])

    def _row(self, work, page):
        for r in sr.list_registry(str(self.reg_root)):
            if (r["work"], r["page"]) == (work, page):
                return r
        self.fail("库里没有 %s / %s：%r"
                  % (work, page, [(r["work"], r["page"])
                                  for r in sr.list_registry(str(self.reg_root))]))


class TestWipeClearsEmbeddings(WebuiCase):
    def test_wipe_checkpoint_removes_the_sidecar(self):
        """重新分割作废旧标注：旧质心跟新聚类不是一个含义，必须一起清。"""
        self.make_job()
        side = self.write_sidecar()
        self.assertTrue(side.is_file())
        self.webui._wipe_checkpoint(self.workdir)
        self.assertFalse(side.is_file())
        self.assertFalse((self.workdir / "speakers.json").exists())


# ---------------------------------------------------------------------------
# 真跑 pyannote：对齐关系与范数（门控）
# ---------------------------------------------------------------------------

def _real_audio():
    r"""真跑用素材。**只读**，不复制、不改动。

    优先 `work\片段_16m00s_19m00s\raw16k.wav`（16 kHz 单声道，worker 直接吃得下）；
    没有就用 `compare\...mp4` 现场抽一条（要 ffmpeg，抽到系统临时目录）。
    """
    if RAW16K.is_file():
        return RAW16K
    return None


@unittest.skipUnless(_enabled(), "需要 JISHENG_RUN_INTEGRATION=1 才运行")
@unittest.skipUnless(A.DIARIZE_PYTHON.exists(), f"缺少 {A.DIARIZE_PYTHON}")
@unittest.skipUnless(RAW16K.is_file(), f"缺少 16k 测试素材 {RAW16K}")
@unittest.skipUnless(A.resolve_diarize_model(None) != A.DEFAULT_DIARIZE_MODEL,
                     "缺少本地 pyannote 模型（models\\pyannote）")
class TestRealPyannoteAlignment(unittest.TestCase):
    """真跑 pyannote，实测「labels 与 centroids 逐行对齐」+ 每条范数 > 0。

    这条测试存在的原因：对齐关系是 Task 4/5 全部匹配逻辑的地基，
    只看文档不算数（计划里那句话就是这么写的）。

    必须用 `venv-diarize\\Scripts\\python.exe` 跑（本环境没有 pyannote）：
        set JISHENG_RUN_INTEGRATION=1
        venv-diarize\\Scripts\\python.exe -m unittest tests.test_diarize_embeddings -v
    """

    def test_labels_match_centroid_rows_and_norms_are_positive(self):
        # 不要直接 `import pyannote.audio`：它在 torchaudio 新版上会先炸
        # （AudioMetaData），兼容层必须由 worker 先应用。所以只用 find_spec 探「装没装」。
        if importlib.util.find_spec("pyannote.audio") is None:
            self.skipTest("本环境没有 pyannote.audio（要用 venv-diarize 跑）")
        import numpy as np2

        worker = _load_worker()          # 真依赖，只有 venv-diarize 里才 import 得动
        worker._apply_compat()           # torchaudio / torch.load 的兼容层，必须在 import pyannote 之前
        src = _real_audio()
        audio = worker._load_wav(src)
        print("[集成] 素材 %s  样本数 %d  采样率 %s"
              % (src, int(audio["waveform"].shape[-1]), audio["sample_rate"]))

        model = A.resolve_diarize_model(None)
        pipeline = worker._load_pipeline(model, "cuda")
        diarization, centroids = pipeline(audio, return_embeddings=True)

        labels = list(diarization.labels())
        print("[集成] labels(%d): %r" % (len(labels), labels))
        print("[集成] centroids: shape=%r dtype=%r"
              % (getattr(centroids, "shape", None),
                 getattr(centroids, "dtype", None)))

        self.assertIsNotNone(centroids, "return_embeddings=True 却没拿到质心")
        rows = int(centroids.shape[0])
        self.assertEqual(len(labels), rows,
                         "labels 数 != 质心行数：逐行对齐的前提不成立")
        norms = [float(np2.linalg.norm(centroids[i])) for i in range(rows)]
        print("[集成] 每行 L2 范数: %r" % ([round(n, 4) for n in norms],))
        for i, n in enumerate(norms):
            self.assertGreater(n, 0.0, "第 %d 行是零向量（补位行）" % i)

    def test_real_worker_writes_sidecar_that_extract_centroids_accepts(self):
        """真跑 worker 子进程：RESULT= + EMBEDDINGS= 两行都在，sidecar 能被库接受。"""
        src = _real_audio()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "dia.json"
            emb = tmp / "dia.embeddings.npz"
            cmd = [str(A.DIARIZE_PYTHON), str(A.DIARIZE_WORKER),
                   str(src), str(out),
                   "--model", A.resolve_diarize_model(None),
                   "--device", "cuda",
                   "--embeddings-out", str(emb)]
            import subprocess
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=1800)
            print("[集成] worker 退出码 %s" % proc.returncode)
            tail = [ln for ln in (proc.stdout or "").splitlines()
                    if ln.startswith("RESULT=") or ln.startswith("EMBEDDINGS=")]
            print("[集成] worker stdout 关键行: %r" % (tail,))
            if proc.returncode != 0:
                print("[集成] worker stderr 尾部: %s"
                      % (proc.stderr or "")[-800:])
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(len(tail), 2, tail)
            self.assertTrue(tail[0].startswith("RESULT="))
            self.assertTrue(emb.is_file())
            with np.load(str(emb)) as z:
                labels = [str(x) for x in z["labels"]]
                cents = z["centroids"]
            norms = [round(float(np.linalg.norm(cents[i])), 4)
                     for i in range(cents.shape[0])]
            print("[集成] sidecar labels=%r shape=%r 每行范数=%r"
                  % (labels, cents.shape, norms))
            self.assertEqual(len(labels), int(cents.shape[0]))
            self.assertTrue(all(n > 0.0 for n in norms), norms)
            got = sr.extract_centroids(cents, labels)
            self.assertEqual(sorted(got), sorted(labels))


if __name__ == "__main__":
    unittest.main(verbosity=2)
