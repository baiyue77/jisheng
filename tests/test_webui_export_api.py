#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Task 6：导出训练集的接口（`POST /api/export`）与后台导出任务。

未命名聚类导出成什么样子、`.list` 怎么写，已经在 `test_train_export_io.py` 里
验过了；这里只验 **WebUI 这一层**：

1. 目标定位与校验（没有分割结果 / 未知 work / 采样率不合法）
2. `source_kind` 的警告只在**确实降级**时出现
3. 导出**走既有任务队列**（不在请求线程里同步跑），失败/停止/进度都复用既有机制
4. 磁盘预检真的能拦住导出，且拒绝时**一个文件都不写**
5. `_job_view` 把结果透给前端（否则面板拿不到 `.list` 路径与计数）

测试使用系统临时目录。
"""

import json
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_labels as sl            # noqa: E402
import speaker_registry as sr          # noqa: E402
import train_export as te              # noqa: E402
import webui                           # noqa: E402


TMP_ROOT = Path(tempfile.gettempdir())

DIA = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
       {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"}]

#: 10 秒静音正弦：概率上不会触发任何 ffmpeg 报错，文件又小
SEGS = [{"start": 0.0, "end": 4.0, "text": "第一句", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 9.0, "text": "第二句", "speaker": "SPEAKER_01"}]

LABELS = {"clusters": {"SPEAKER_00": {"name": "艾丽丝"},
                       "SPEAKER_01": {"name": "鲁迪"}},
          "segment_overrides": {},
          "work_name": "无职转生"}


def write_wav(path, seconds=10.0, sr=16000, gap=None):
    """写一段真有声音的 WAV（全静音会被 trim 整条吃掉，断言就没意义了）。

    `gap=(起, 止)` 在中间挖一段 0 采样（真静音），给 silencedetect 用。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    one = int(sr / 200.0)                       # 200 Hz 方波，峰值满刻度
    frames = bytearray()
    for i in range(n):
        if gap and gap[0] * sr <= i < gap[1] * sr:
            v = 0
        else:
            v = 12000 if (i // one) % 2 == 0 else -12000
        frames += int(v).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(bytes(frames))
    return path


class ExportApiBase(unittest.TestCase):
    """把 `WORK_DIR` / `OUTPUT_ROOT` / `JOBS` / 声纹库根全指到临时目录。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="jisheng_export_", dir=str(TMP_ROOT))
        self.root = Path(self._tmp.name)
        self._work_dir = webui.WORK_DIR
        self._out_root = webui.OUTPUT_ROOT
        self._jobs = webui.JOBS
        self._queue = webui.TASK_QUEUE
        webui.WORK_DIR = self.root / "work"
        webui.OUTPUT_ROOT = self.root / "output"
        webui.JOBS = {}
        # `TASK_QUEUE` 是**模块级全局**：`_enqueue`（别的测试文件会调）把任务留在
        # 里面没人消费，本文件再起 `worker_loop()` 就会把别人留下的任务一起跑掉 ——
        # 于是 `run_job` 被多调用几次，`assert_not_called()` 假红
        # （实测：单独跑绿、全量跑红）。每个用例换一条自己的队列。
        webui.TASK_QUEUE = queue.Queue()
        # 命名会 enroll 到全局声纹库：根必须隔离，绝不写用户真实的 speakers_registry\
        self._reg_prev = sr.default_root()
        sr.set_default_root(str(self.root / "registry"))

        self.workdir = webui.WORK_DIR / "EP01"
        self.workdir.mkdir(parents=True)
        (self.workdir / "diarization.json").write_text(
            json.dumps(DIA, ensure_ascii=False), encoding="utf-8")
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in SEGS) + "\n",
            encoding="utf-8")
        (self.workdir / "speakers.json").write_text(
            json.dumps(LABELS, ensure_ascii=False), encoding="utf-8")

        self.media = write_wav(self.root / "media" / "EP01.wav")
        self.src = write_wav(self.root / "src" / "vocals.wav")
        self.export_root = webui.OUTPUT_ROOT / webui.TRAIN_EXPORT_DIRNAME

    def tearDown(self):
        webui.WORK_DIR = self._work_dir
        webui.OUTPUT_ROOT = self._out_root
        webui.JOBS = self._jobs
        webui.TASK_QUEUE = self._queue
        sr.set_default_root(getattr(self, "_reg_prev", None))
        self._tmp.cleanup()

    # ---- 辅助 ----

    def make_job(self, jid="j1"):
        webui.JOBS[jid] = {"id": jid, "media": str(self.media),
                           "work": str(self.workdir),
                           "out_dir": str(webui.OUTPUT_ROOT),
                           "logs": [], "status": "done", "stage": "完成"}
        return webui.JOBS[jid]

    def fake_picker(self, kind="separated_hq", mode_seen=None):
        """假 `pick_source`：不碰 ffmpeg，但记录拿到的 mode。"""
        def picker(work, media, mode, **kw):
            if mode_seen is not None:
                mode_seen.append(mode)
            return Path(self.src), kind
        return picker

    def prepare_export(self, sr=32000, source="hq", **over):
        """只入队、不开跑：返回 (job, 假 pick_source)。"""
        payload = {"job": "j1", "source": source, "sr": sr, "drop_short": False}
        payload.update(over)
        self.make_job()
        obj, code = webui.api_export(payload)
        self.assertEqual(code, 200, obj)
        job = webui.JOBS[obj["id"]]
        # 假静音探测：真 ffmpeg 在假音频上跑一轮纯属浪费时间，且结果无关
        job["_export_silence_finder"] = lambda a, b: []
        return job

    def export_job(self, kind="separated_hq", mode_seen=None, sr=32000, **over):
        """真正跑一遍导出任务（同步调用 worker 侧的 run_export_job）。"""
        job = self.prepare_export(sr=sr, **over)
        webui.run_export_job(job, source_picker=self.fake_picker(kind, mode_seen))
        return job

    def list_lines(self, path):
        raw = Path(path).read_bytes()
        return raw.decode("utf-8").strip("\n").split("\n"), raw

    def run_worker_once(self, job):
        """只让 worker 处理 `job` 这一件：先清空本用例的队列，再补哨兵结束循环。

        `TASK_QUEUE` 在 setUp 里换成了本用例自己的队列（模块级全局会被别的
        测试文件留下任务），这里再倒一次，保证 worker 只看到我们放的那一件。
        """
        while True:
            try:
                webui.TASK_QUEUE.get_nowait()
            except queue.Empty:
                break
        webui.TASK_QUEUE.put(job)
        webui.TASK_QUEUE.put(None)
        webui.worker_loop()


class TestExportValidation(ExportApiBase):
    """Steps 1-2：目标定位与采样率校验。"""

    def test_export_unknown_work_returns_404(self):
        obj, code = webui.api_export({"work": "不存在的作品", "source": "hq"})
        self.assertEqual(code, 404)

    def test_export_rejects_invalid_format_before_queueing(self):
        self.make_job()
        for kind in ('mp3', [], None, False):
            with self.subTest(kind=kind):
                result, status = webui.api_export({'job': 'j1', 'format': kind})
                self.assertEqual(status, 400, result)
                self.assertNotIn('id', result)

    def test_export_respects_state_language_and_format(self):
        webui.save_state(self.workdir, {'opts': {'language': 'en'}})
        job = self.export_job()
        self.assertEqual(job['status'], 'done', job.get('error'))
        self.assertEqual(self.list_lines(job['result']['list'])[0][0].split('|')[2], 'EN')

        webui.save_state(self.workdir, {'opts': {'language': 'auto'}})
        _, status = webui.api_export({'job': 'j1'})
        self.assertEqual(status, 400)
        job = self.export_job(format='txt')
        self.assertEqual(job['status'], 'done', job.get('error'))
        self.assertIn('\tund\t', Path(job['result']['manifest_path']).read_text(encoding='utf-8'))

    def test_export_unknown_job_returns_404(self):
        obj, code = webui.api_export({"job": "nope"})
        self.assertEqual(code, 404)

    def test_export_without_segments_returns_409(self):
        self.make_job()
        (self.workdir / "segments.jsonl").unlink()
        obj, code = webui.api_export({"job": "j1"})
        self.assertEqual(code, 409)

    def test_export_without_diarization_returns_409(self):
        self.make_job()
        (self.workdir / "diarization.json").unlink()
        obj, code = webui.api_export({"job": "j1"})
        self.assertEqual(code, 409)

    def test_export_rejects_bad_sr(self):
        for bad in (16000, 22050, 48000, 0, -1, "abc", None, True):
            with self.subTest(sr=bad):
                obj, code = webui.api_export({"job": "j1", "sr": bad})
                # 具体错误码可能是 404（连任务都没有）—— 但绝不能是「接受」
                self.assertIn(code, (400, 404), "%r：%s" % (bad, obj))
                self.assertNotIn("id", obj, "%r 被接受了：%s" % (bad, obj))

    def test_bad_sr_error_names_the_sample_rate(self):
        self.make_job()
        obj, code = webui.api_export({"job": "j1", "sr": 16000})
        self.assertEqual(code, 400, obj)
        self.assertIn("采样率", obj["error"])
        self.assertNotIn("id", obj)

    def test_bad_source_is_rejected(self):
        self.make_job()
        for bad in ("ultra", "HQ", 1):
            with self.subTest(source=bad):
                obj, code = webui.api_export({"job": "j1", "source": bad})
                self.assertEqual(code, 400, obj)
                self.assertIn("音源", obj["error"])

    def test_export_accepts_both_allowed_sr(self):
        for ok in (32000, 44100, "32000"):
            with self.subTest(sr=ok):
                self.make_job()
                obj, code = webui.api_export({"job": "j1", "sr": ok})
                self.assertEqual(code, 200, obj)
                self.assertEqual(webui.JOBS[obj["id"]]["sr"], int(ok))

    def test_export_rejects_empty_target(self):
        obj, code = webui.api_export({})
        self.assertIn(code, (400, 404))

    def test_bad_sr_does_not_enqueue_anything(self):
        self.make_job()
        webui.api_export({"job": "j1", "sr": 12345})
        self.assertEqual(len(webui.JOBS), 1, "坏 sr 却建了一个导出任务")

    def test_export_defaults_to_hq_and_32000(self):
        self.make_job()
        obj, _ = webui.api_export({"job": "j1"})
        job = webui.JOBS[obj["id"]]
        self.assertEqual(job["source"], "hq")
        self.assertEqual(job["sr"], te.DEFAULT_SAMPLE_RATE)

    def test_export_uses_work_name_input(self):
        """面板上填的作品名必须**落进 speakers.json**，不只是跟着任务跑。

        不然：导出用的是新名字，面板下次打开（`/api/speakers` 读 speakers.json）
        却回填旧名字 —— 用户会以为没生效。
        """
        self.make_job()
        obj, _ = webui.api_export({"job": "j1", "work_name": "自定义作品"})
        self.assertEqual(webui.JOBS[obj["id"]]["work_name"], "自定义作品")
        self.assertEqual(sl.get_work_name(sl.load_labels(self.workdir)), "自定义作品")

    def test_export_keeps_the_stored_work_name_when_the_payload_omits_it(self):
        self.make_job()
        obj, _ = webui.api_export({"job": "j1"})
        self.assertEqual(webui.JOBS[obj["id"]]["work_name"], "")
        self.assertEqual(sl.get_work_name(sl.load_labels(self.workdir)), "无职转生")


class TestExportEnqueue(ExportApiBase):
    """Step 4：导出必须**入队**，请求线程不许同步跑。"""

    def test_api_export_enqueues_and_does_not_touch_disk(self):
        self.make_job()
        obj, code = webui.api_export({"job": "j1", "source": "hq", "sr": 32000})
        self.assertEqual(code, 200, obj)
        self.assertIn("id", obj)
        self.assertFalse(self.export_root.exists(),
                         "请求线程里就把产物写出来了 = 没走队列")
        self.assertEqual(webui.JOBS[obj["id"]]["status"], "wait")

    def test_queued_job_is_tagged_as_export(self):
        self.make_job()
        obj, _ = webui.api_export({"job": "j1"})
        self.assertEqual(webui.JOBS[obj["id"]].get("kind"), "export")

    def test_export_does_not_need_a_source_file_at_request_time(self):
        """真正的音源准备（可能是一整趟 HQ 分离）必须留给 worker。"""
        self.make_job()
        obj, code = webui.api_export({"job": "j1"})
        self.assertEqual(code, 200, obj)
        self.assertFalse(self.export_root.exists())

    def test_run_export_job_writes_list_and_result(self):
        job = self.export_job()
        self.assertEqual(job["status"], "done", job.get("error"))
        res = job["result"]
        self.assertTrue(Path(res["list"]).is_file(), res["list"])
        self.assertTrue(Path(res["list"]).is_relative_to(self.export_root))
        self.assertEqual(res["n_clips"], 2)
        self.assertEqual(res["folders"], {"艾丽丝": 1, "鲁迪": 1})

    def test_worker_loop_dispatches_export_job(self):
        """走 `kind` 分发，且**只有**导出分支能跑成。

        这里故意给一个不存在的 media：如果 dispatch 落到 `run_job`（识别任务），
        它会立刻以「读不到输入文件」失败 —— 于是「status==done」这一条就抓住了
        「导出任务被当识别任务跑」这种分发错误。
        """
        job = self.prepare_export()
        job["media"] = str(self.root / "不存在的媒体.mkv")
        with mock.patch.object(webui.ts, "pick_source",
                               side_effect=self.fake_picker()):
            self.run_worker_once(job)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["stage"], "完成")
        self.assertIn("result", job)
        self.assertTrue(Path(job["result"]["list"]).is_file())

    def test_worker_loop_still_runs_plain_jobs(self):
        """回归：`kind` 分发不能把既有（没有 kind 的）任务弄丢。"""
        self.make_job()
        webui.JOBS["j1"]["opts"] = {}
        with mock.patch.object(webui, "run_job") as fake:
            self.run_worker_once(webui.JOBS["j1"])
        fake.assert_called_once()
        self.assertIsNone(webui.JOBS["j1"].get("kind"))

    def test_unknown_kind_is_an_error_not_a_silent_asr_run(self):
        """未知 kind 绝对不能当成 ASR 任务跑掉（那会白跑一整趟识别）。"""
        job = {"id": "jX", "kind": "打错了", "media": "x", "logs": [],
               "status": "wait", "stage": "", "opts": {}}
        with mock.patch.object(webui, "run_job") as fake:
            self.run_worker_once(job)
        fake.assert_not_called()
        self.assertEqual(job["status"], "error")
        self.assertIn("未知的任务类型", job["error"])

    def test_job_view_carries_kind_and_result_but_not_internals(self):
        job = self.export_job()
        view = webui._job_view(job)
        self.assertEqual(view["result"]["list"], job["result"]["list"])
        # kind 必须透给前端：面板靠它认出「这是我的导出任务」，
        # 也是唯一能让前端不给导出任务画「说话人标注」按钮的依据。
        self.assertEqual(view["kind"], "export")
        self.assertEqual(webui._job_view(self.make_job())["kind"], "asr")
        self.assertNotIn("_export_source", view)
        self.assertNotIn("_last_progress", view)
        self.assertNotIn("_export_silence_finder", view)

    def test_failure_sets_error_status(self):
        job = self.prepare_export()
        job["_export_source"] = (str(self.root / "没有这个文件.wav"), "separated_hq")
        webui.run_export_job(job)
        self.assertEqual(job["status"], "error")
        self.assertTrue(job["error"])

    def test_source_prepare_failure_is_reported_not_crashed(self):
        job = self.prepare_export()
        with mock.patch.object(webui.ts, "pick_source",
                               side_effect=webui.ts.SourceError("分离环境不存在")):
            webui.run_export_job(job)
        self.assertEqual(job["status"], "error")
        self.assertIn("分离环境不存在", job["error"])


class TestExportSourceKind(ExportApiBase):
    """Step 3：warning 只在该出现时出现。"""

    #: 每种降级音源各自的**语义锚点**（写死在测试里，不从 train_export 反推）。
    #:
    #: 为什么钉锚点而不是整句：整句的逐字钉在 `test_train_export_io.py`
    #: （那边是文案的归属地）。这里要抓的是「warning 根本不是这句」——
    #: 审查者的变异 W14 把 `separated_16k` 的文案换成「自造文案：这个源不行」，
    #: 老断言 `assertEqual(result.warning, te.source_warning(kind))` 两边同源、
    #: 是同义反复，617 条全绿。锚点都取自「这句话在说什么」而不是措辞：
    #: 16 kHz 轨是**人声轨**、raw16k 是**中间产物**、media 是**原片**；
    #: 三个词互不出现于彼此的文案里，所以改文案改错对象当场红，
    #: 而单纯调语气（加字、换标点）不会误报。
    ANCHORS = {"separated_16k": "人声轨",
               "raw16k": "中间产物",
               "media": "原片"}

    def test_warns_when_source_is_band_limited(self):
        for kind, anchor in self.ANCHORS.items():
            with self.subTest(kind=kind):
                job = self.export_job(kind=kind, source="fast")
                self.assertEqual(job["status"], "done", job.get("error"))
                warning = job["result"]["warning"]
                self.assertTrue(warning, "%s 的降级音源没有任何 warning" % kind)
                self.assertIn(anchor, warning,
                              "%s 的 warning 不是那句话：%r" % (kind, warning))
                for other, token in self.ANCHORS.items():
                    if other != kind:
                        self.assertNotIn(token, warning,
                                         "%s 的 warning 串到了 %s 的文案" % (kind, other))

    def test_no_warning_for_hq_source(self):
        job = self.export_job()
        self.assertIsNone(job["result"]["warning"])
        self.assertEqual(job["result"]["source_kind"], "separated_hq")

    def test_fast_mode_is_passed_to_pick_source(self):
        seen = []
        job = self.export_job(kind="separated_16k", source="fast",
                              mode_seen=seen)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(seen, ["fast"])
        self.assertEqual(job["result"]["source_kind"], "separated_16k")

    def test_hq_mode_is_passed_to_pick_source(self):
        seen = []
        job = self.export_job(mode_seen=seen)
        self.assertEqual(seen, ["hq"])

    def test_progress_reaches_the_job(self):
        """进度与阶段必须**逐次**写进 job（复用既有 `_set`），不是另起一套。

        这里刻意钉住**完整序列**，而不是 `max(seen)` / `seen[-1]`。
        实测（变异性实验）：把「磁盘预检」0.55 或「切分落盘」0.60 那整条 `_set`
        删掉，`max(seen)` 仍是最终的 1.0、`seen[-1]` 也仍是 1.0 ——
        用「最大值 >= 0.7」这种断言去测中间那几步，**一次都抓不到**。
        序列是确定性的：准备音源 0.05 -> 磁盘预检 0.55 -> 切分 0.60
        -> 整理 0.95 -> 完成 1.00。
        """
        seen = []
        stages = []
        job = self.prepare_export()
        real = webui._set

        def spy(j, **kw):
            if "progress" in kw:
                seen.append(kw["progress"])
            if "stage" in kw:
                stages.append(kw["stage"])
            return real(j, **kw)

        with mock.patch.object(webui, "_set", side_effect=spy):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(seen, [0.05, 0.55, 0.60, 0.95, 1.0],
                         "进度序列不对（少了哪一步？）：%r" % (seen,))
        self.assertEqual(stages, ["准备音源", "磁盘预检", "导出中", "导出中", "完成"],
                         "阶段序列不对（面板会看着像卡住）：%r" % (stages,))
        self.assertEqual(job["result"]["n_clips"], 2)

    def test_disk_precheck_stage_is_reported_before_writing(self):
        """「磁盘预检」必须作为一个**阶段**出现，且在落盘之前。

        预检**确实**可能很慢：它现在拿到的是与落盘共用的那个**真** ffmpeg
        静音探测器（P4 之前预检拿的是 None，所以那句话在生产路径下并不成立），
        没有内部静音信息的音源全流程要在这里跑一遍 `silencedetect`。
        面板上看不见这个阶段就等于卡住 —— 实测这条断言能抓住「预检 stage 被删掉」。
        """
        stages = []
        job = self.prepare_export()
        real = webui._set

        def spy(j, **kw):
            if "stage" in kw:
                stages.append(kw["stage"])
            return real(j, **kw)

        with mock.patch.object(webui, "_set", side_effect=spy):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertIn("磁盘预检", stages)
        self.assertLess(stages.index("磁盘预检"), stages.index("导出中"))


class TestExportDiskPrecheck(ExportApiBase):
    """设计 §12：导出前按估算体积预检磁盘；不够就**一条都不写**。"""

    def test_insufficient_space_blocks_export(self):
        job = self.prepare_export()
        with mock.patch.object(webui.te, "disk_free_bytes",
                               return_value=1024 * 1024):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "error")
        self.assertIn("磁盘空间不足", job["error"])
        # 关键：拒绝发生在写任何东西之前
        self.assertFalse(self.export_root.exists(),
                         "磁盘预检失败却已经建了导出目录")

    def test_real_disk_check_blocks_export(self):
        """磁盘预检要**真的拦住导出**，而且拦在 `export_work` 之前。

        不能把 `disk_free_bytes` 钉成小值（那只是重放「预检不通过」），
        也不能指望 `export_work` 自己去拦：它内部有同一套检查，所以
        「删掉预检」在**最终状态**上是观察不到的（两边都 error）。
        能观察到的是**它有没有被调用**：这里把预检换成会抛的版本，
        于是 `export_work` 必须一次都不被调用。
        """
        job = self.prepare_export()
        job["_export_source"] = (str(self.src), "separated_hq")
        real = te.export_work
        with mock.patch.object(webui.te, "precheck_work",
                               side_effect=te.InsufficientSpace("空间不足（测试）")), \
             mock.patch.object(webui.te, "export_work", side_effect=real) as spy:
            webui.run_export_job(job)
        self.assertEqual(job["status"], "error", job.get("error"))
        self.assertIn("InsufficientSpace", job["error"])
        self.assertIn("空间不足（测试）", job["error"])
        spy.assert_not_called()
        self.assertFalse(self.export_root.exists(),
                         "磁盘预检失败却已经建了导出目录")

    def test_disk_precheck_is_actually_called(self):
        """预检必须被**调用**（不是摆设），且用同一个 `export_work` 再落盘。

        故意把 `estimate_bytes` 压成 1 字节，这样 `export_work` 内部的同一套
        检查也一定放行 —— 于是「预检被删掉」就只剩「`precheck_work` 没被调用」
        这一个可观察差异。
        """
        job = self.prepare_export()
        job["_export_source"] = (str(self.src), "separated_hq")
        real = te.precheck_work
        with mock.patch.object(webui.te, "estimate_bytes", return_value=1), \
             mock.patch.object(webui.te, "precheck_work",
                               side_effect=real) as spy:
            webui.run_export_job(job)
        self.assertEqual(job["status"], "done", job.get("error"))
        spy.assert_called_once()
        # 预检拿到的就是选定的那条音源（`export_work` 也拿它落盘，不能是两条）
        self.assertEqual(str(spy.call_args[0][2]), str(self.src))

    def test_enough_space_lets_it_through(self):
        job = self.prepare_export()
        with mock.patch.object(webui.te, "disk_free_bytes", return_value=1 << 40):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "done", job.get("error"))

    def test_precheck_uses_the_planned_clip_seconds(self):
        """预检必须真的按计划体积算，不是拍一个常数。"""
        need = te.plan_export_space(8.0, 32000)
        self.assertEqual(need, int(te.estimate_bytes(8.0, 32000) * te.DISK_SAFETY)
                         + te.DISK_RESERVE_BYTES)
        self.assertEqual(need, int(8.0 * 32000 * 2 * te.DISK_SAFETY)
                         + te.DISK_RESERVE_BYTES)
        self.assertGreater(te.plan_export_space(100.0, 32000),
                           te.plan_export_space(8.0, 32000))

    def test_precheck_only_does_not_write(self):
        self.export_root.parent.mkdir(parents=True, exist_ok=True)
        out = te.precheck_work([dict(SEGS[0])], dict(LABELS), str(self.src),
                               self.export_root,
                               work_name="预检作品", sr=32000,
                               silence_finder=lambda a, b: [],
                               source_kind="separated_hq")
        self.assertTrue(out["ok"])
        self.assertGreater(out["needed_bytes"], 0)
        self.assertFalse(self.export_root.exists())


class TestExportStopAndAbandon(ExportApiBase):
    """停止 / 放弃必须能中断导出，并且走既有机制（不是导出专属的分支）。"""

    def test_stop_during_prepare_marks_stopped(self):
        job = self.prepare_export()
        with mock.patch.object(webui.ts, "pick_source",
                               side_effect=webui.ts.SourceCancelled("已被停止")):
            webui.run_export_job(job)
        self.assertEqual(job["status"], "stopped")
        self.assertEqual(job["stage"], "已停止")
        self.assertFalse(self.export_root.exists())

    def test_abandon_during_prepare_marks_abandoned(self):
        job = self.prepare_export()
        job["abandon"] = True
        with mock.patch.object(webui.ts, "pick_source",
                               side_effect=webui.ts.SourceCancelled("已被停止")):
            webui.run_export_job(job)
        self.assertEqual(job["status"], "abandoned")

    def test_stop_is_honoured_inside_the_export_callback(self):
        """`should_stop` 真的传给了 train_export，而且是**活的**。

        只看「有没有传一个 callable」不够：传一个永远返回 False 的假回调也能过。
        所以这里先要求它开跑时为 False，再改动停止标志，同一个回调必须立刻为真。
        用 `stop_calls` 钉住调用顺序：`_run_export_job_locked` 在调 `export_work`
        之前查过一次停止，所以第一个值必然是 False。
        """
        job = self.prepare_export()
        seen = []
        stop_calls = []
        real = te.export_work

        def spy(*a, **kw):
            seen.append(kw.get("should_stop"))
            return real(*a, **kw)

        def stop_probe(*a, **kw):
            value = bool(job.get("stop"))
            stop_calls.append(value)
            return value

        with mock.patch.object(webui.te, "export_work", side_effect=spy):
            with mock.patch.object(webui, "stop_requested",
                                   side_effect=stop_probe):
                webui.run_export_job(job, source_picker=self.fake_picker())

        self.assertTrue(seen, "export_work 没被调用")
        self.assertTrue(callable(seen[0]),
                        "should_stop 没传给 export_work，导出中途停不下来")
        self.assertEqual(stop_calls[:1], [False], "刚开跑就报要停")
        # 用户点停止之后，同一个回调必须立刻为真 —— 否则 ffmpeg 收不到中断
        job["stop"] = True
        self.assertTrue(seen[0](), "停止标志改了，回调还是 False")

    def test_real_abort_midway_marks_stopped(self):
        """不做任何 mock：在 `cut_segment` 里点停止，导出必须真的停下来。"""
        job = self.prepare_export()
        calls = []
        real = te.cut_segment

        def spy(*a, **kw):
            calls.append(1)
            job["stop"] = True
            return real(*a, **kw)

        with mock.patch.object(webui.te, "cut_segment", side_effect=spy):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "stopped")
        self.assertTrue(calls)
        self.assertNotIn("result", job)

    def test_export_cleanup_frees_intermediates(self):
        """`cleanup_intermediates` 必须被调用，释放量写进日志。"""
        job = self.prepare_export()
        with mock.patch.object(webui.ts, "cleanup_intermediates",
                               return_value=12.5) as fake:
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "done", job.get("error"))
        fake.assert_called_once()
        self.assertTrue(any("12.5" in ln for ln in job["logs"]),
                        "释放的空间没写进日志：%r" % (job["logs"][-3:],))

    def test_cleanup_is_called_once_after_a_successful_export(self):
        job = self.prepare_export()
        with mock.patch.object(webui.ts, "cleanup_intermediates") as fake:
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "done", job.get("error"))
        fake.assert_called_once()

    def test_precheck_gets_a_live_stop_hook(self):
        """磁盘预检那一步也必须拿到**活的** `should_stop`。

        预检**确实**会跑真 ffmpeg 静音探测（P4 之后它与落盘共用同一个探测器，
        全流程只跑一次），所以它可能很慢。它要是拿不到停止回调，用户在预检期间
        点停止就得干等 —— 实测把这里的 `should_stop=cancelled`
        换成 None，其它用例一条都不红，只有这条能抓到。
        """
        job = self.prepare_export()
        seen = []

        def fake_precheck(segments, labels, src, out_root, **kw):
            seen.append(kw.get("should_stop"))
            return {"ok": True, "needed_bytes": 0, "free_bytes": 1 << 62,
                    "estimated_bytes": 0, "planned_seconds": 0.0,
                    "work": "W", "work_dir": str(self.export_root / "W"),
                    "list_path": str(self.export_root / "W.list"),
                    "report_path": str(self.export_root / "W.json"),
                    "source_kind": "separated_hq", "n_total": 1, "n_planned": 1}

        with mock.patch.object(webui.te, "precheck_work",
                               side_effect=fake_precheck):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertTrue(seen, "precheck_work 没被调用")
        self.assertTrue(callable(seen[0]), "预检没拿到 should_stop")
        self.assertFalse(seen[0](), "刚开跑就报要停")
        job["stop"] = True
        self.assertTrue(seen[0](), "停止标志改了，预检的回调还是 False")

    def test_cleanup_not_called_on_cancel(self):
        job = self.prepare_export()
        with mock.patch.object(webui.ts, "cleanup_intermediates") as fake:
            webui.run_export_job(
                job, source_picker=mock.Mock(
                    side_effect=webui.ts.SourceCancelled("已被停止")))
        self.assertEqual(job["status"], "stopped")
        fake.assert_not_called()

    def test_cleanup_not_called_when_the_export_itself_fails(self):
        """导出失败（源不存在）时半成品不该被清掉 —— 用户还要查原因。"""
        job = self.prepare_export()
        job["_export_source"] = (str(self.root / "缺.wav"), "separated_hq")
        with mock.patch.object(webui.ts, "cleanup_intermediates") as fake:
            webui.run_export_job(job)
        self.assertEqual(job["status"], "error")
        fake.assert_not_called()


class TestExportListContract(ExportApiBase):
    """`.list` 的三条源码级硬约束（GPT-SoVITS）。"""

    def test_each_line_has_exactly_four_fields(self):
        job = self.export_job()
        lines, _ = self.list_lines(job["result"]["list"])
        self.assertEqual(len(lines), 2)
        for ln in lines:
            self.assertEqual(len(ln.split("|")), 4, ln)

    def test_text_field_never_contains_a_raw_pipe(self):
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in [
                {"start": 0.0, "end": 4.0, "text": "甲|乙", "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 9.0, "text": "丙", "speaker": "SPEAKER_01"}])
            + "\n", encoding="utf-8")
        job = self.export_job()
        lines, _ = self.list_lines(job["result"]["list"])
        for ln in lines:
            self.assertEqual(len(ln.split("|")), 4, ln)
        self.assertIn("甲", lines[0])

    def test_pipe_in_a_role_name_does_not_break_the_line(self):
        """角色名也可能带 `|`（人手打进去的），它同样必须被转义。

        转义结果与目录名**同源**（P5）：两处用同一个净化结果，所以字段里是
        半角下划线 `_` 而不是全角竖线 —— 目录名与字段必须能对上，
        见下面那条 `test_dir_name_equals_the_speaker_field`。
        """
        (self.workdir / "speakers.json").write_text(
            json.dumps({"clusters": {"SPEAKER_00": {"name": "甲|乙"},
                                     "SPEAKER_01": {"name": "丙"}},
                        "segment_overrides": {}, "work_name": "无职转生"},
                       ensure_ascii=False), encoding="utf-8")
        job = self.export_job()
        self.assertEqual(job["status"], "done", job.get("error"))
        lines, _ = self.list_lines(job["result"]["list"])
        for ln in lines:
            self.assertEqual(len(ln.split("|")), 4, ln)
        self.assertIn("甲", lines[0])
        self.assertEqual(lines[0].split("|")[1], "甲_乙")

    def test_dir_name_equals_the_speaker_field(self):
        """`.list` 第 2 字段必须**等于**这条 wav 所在的那级角色目录名。

        面板/文档承诺的产物形状是 `<导出根>\\<作品名>\\<角色名>\\*.wav`，而
        `.list` 的说话人字段就是「角色名」。GPT-SoVITS 按第 1 字段找音频、
        按第 2 字段认说话人；两边不是同一个字符串时（角色名里带 `|`：
        目录曾是 `甲_乙`、字段曾是 `甲｜乙`），按说话人切片就会找不到文件。
        """
        for name in ("艾丽丝", "甲|乙", "甲<乙>", "甲:乙", "甲/乙", "甲?乙*丙", "甲."):
            with self.subTest(role=name):
                (self.workdir / "speakers.json").write_text(
                    json.dumps({"clusters": {"SPEAKER_00": {"name": name},
                                             "SPEAKER_01": {"name": "丙"}},
                                "segment_overrides": {}, "work_name": "无职转生"},
                               ensure_ascii=False), encoding="utf-8")
                job = self.export_job()
                self.assertEqual(job["status"], "done", job.get("error"))
                lines, _ = self.list_lines(job["result"]["list"])
                self.assertTrue(lines)
                for ln in lines:
                    parts = ln.split("|")
                    self.assertEqual(len(parts), 4, ln)
                    wav, speaker = Path(parts[0]), parts[1]
                    self.assertEqual(wav.parent.name, speaker,
                                     "目录名与 .list 说话人字段不一致：%r != %r"
                                     % (wav.parent.name, speaker))

    def test_dir_name_is_not_empty_even_for_a_fully_illegal_role(self):
        """角色名全是非法字符时目录名不能净化成空串（那会落到上一级目录）。"""
        (self.workdir / "speakers.json").write_text(
            json.dumps({"clusters": {"SPEAKER_00": {"name": "..."},
                                     "SPEAKER_01": {"name": "丙"}},
                        "segment_overrides": {}, "work_name": "无职转生"},
                       ensure_ascii=False), encoding="utf-8")
        job = self.export_job()
        self.assertEqual(job["status"], "done", job.get("error"))
        lines, _ = self.list_lines(job["result"]["list"])
        wav, speaker = Path(lines[0].split("|")[0]), lines[0].split("|")[1]
        self.assertTrue(speaker)
        self.assertEqual(wav.parent.name, speaker)
        self.assertEqual(wav.parent.parent.name, "无职转生")

    def test_list_is_lf_and_language_is_ja(self):
        job = self.export_job()
        lines, raw = self.list_lines(job["result"]["list"])
        self.assertNotIn(b"\r", raw, "`.list` 里出现了 CR")
        self.assertTrue(raw.endswith(b"\n"))
        for ln in lines:
            self.assertEqual(ln.split("|")[2], "JA")

    def test_sr_choice_reaches_the_written_wavs(self):
        """面板上选 44100 必须真的写出 44.1 kHz 的 wav，不是只改了返回字段。"""
        job = self.export_job(sr=44100)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["result"]["sample_rate"], 44100)
        lines, _ = self.list_lines(job["result"]["list"])
        self.assertTrue(lines)
        with wave.open(lines[0].split("|")[0], "rb") as fh:
            self.assertEqual(fh.getframerate(), 44100)

    def test_default_sr_writes_32k_wavs(self):
        job = self.export_job()
        lines, _ = self.list_lines(job["result"]["list"])
        with wave.open(lines[0].split("|")[0], "rb") as fh:
            self.assertEqual(fh.getframerate(), 32000)

    def test_paths_point_at_existing_wavs_under_the_export_root(self):
        job = self.export_job()
        lines, _ = self.list_lines(job["result"]["list"])
        for ln in lines:
            wav = Path(ln.split("|")[0])
            self.assertTrue(wav.is_file(), wav)
            self.assertTrue(wav.is_relative_to(self.export_root), wav)

    def test_work_name_reaches_the_list_path(self):
        job = self.prepare_export(work_name="无职转生")
        webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(Path(job["result"]["list"]).name, "无职转生.list")
        self.assertTrue(Path(job["result"]["list"]).is_relative_to(self.export_root))
        self.assertEqual(job["result"]["report"]["work"], "无职转生")

    def test_work_name_from_speakers_json_is_used(self):
        job = self.export_job()
        self.assertEqual(Path(job["result"]["list"]).name, "无职转生.list")


class TestExportUnknownBucket(ExportApiBase):
    """没起名的聚类不进 `.list`，但必须落 `_未命名\\` 并计数。"""

    def test_unnamed_cluster_is_counted_and_not_in_list(self):
        (self.workdir / "speakers.json").write_text(
            json.dumps({"clusters": {"SPEAKER_00": {"name": ""},
                                     "SPEAKER_01": {"name": "鲁迪"}},
                        "segment_overrides": {}, "work_name": "无职转生"},
                       ensure_ascii=False), encoding="utf-8")
        job = self.export_job()
        res = job["result"]
        self.assertEqual(res["n_unknown"], 1)
        self.assertEqual(res["n_clips"], 2)          # 未命名的也落盘了
        lines, _ = self.list_lines(res["list"])
        self.assertEqual(len(lines), 1)
        self.assertIn("鲁迪", lines[0])


class TestExportDropShort(ExportApiBase):
    """`drop_short=True` 时短句不落盘，计数仍然在。"""

    def setUp(self):
        super().setUp()
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in [
                {"start": 0.0, "end": 0.8, "text": "嗯", "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 9.0, "text": "第二句", "speaker": "SPEAKER_01"}])
            + "\n", encoding="utf-8")

    def test_short_is_written_by_default(self):
        job = self.export_job()
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["result"]["n_short"], 1)
        self.assertTrue((self.export_root / "无职转生" / te.SHORT_BUCKET).is_dir())

    def test_drop_short_skips_the_file_but_keeps_the_count(self):
        job = self.export_job(drop_short=True)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["result"]["n_short"], 1)
        self.assertFalse((self.export_root / "无职转生" / te.SHORT_BUCKET).exists())
        self.assertEqual(job["result"]["report"]["short_skipped"], 1)
        lines, _ = self.list_lines(job["result"]["list"])
        self.assertEqual(len(lines), 1)


class TestExportRootIsPinned(ExportApiBase):
    """导出根必须叫 `output\\说话人\\`（T5）。

    以前所有断言都从 `webui.TRAIN_EXPORT_DIRNAME` 反推期望路径，于是把常量
    改成 `"speakers"`（变异 W27）**617 条全绿** —— 目录名是用户看得见的产物
    形状，也是 ui.html 帮助文案里向用户承诺的东西，必须逐字钉住。
    """

    def test_dirname_constant_is_pinned(self):
        self.assertEqual(webui.TRAIN_EXPORT_DIRNAME, "说话人")

    def test_export_root_path_is_pinned(self):
        self.assertEqual(webui.training_export_root(),
                         webui.OUTPUT_ROOT / "说话人")

    def test_repeated_exports_keep_existing_wavs_and_manifests(self):
        first = self.export_job()
        self.assertEqual(first['status'], 'done', first.get('error'))
        first_manifest = Path(first['result']['manifest_path'])
        _, first_bytes = self.list_lines(first_manifest)
        other = self.export_job(format='indextts')
        again = self.export_job()
        for job in (other, again):
            self.assertEqual(job['status'], 'done', job.get('error'))
            manifest = Path(job['result']['manifest_path'])
            self.assertNotEqual(manifest.parent, first_manifest.parent)
            self.assertTrue(manifest.is_file())
            self.assertEqual(job['files'][0]['path'], str(manifest))
        self.assertEqual(first_manifest.read_bytes(), first_bytes)
        for line in self.list_lines(first_manifest)[0]:
            self.assertTrue(Path(line.split('|')[0]).is_file())

    def test_written_list_and_wavs_use_the_pinned_root(self):
        job = self.export_job()
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(Path(job["result"]["list"]),
                         self.root / "output" / "说话人" / "无职转生.list")
        self.assertEqual(Path(job["result"]["work_dir"]),
                         self.root / "output" / "说话人" / "无职转生")
        lines, _ = self.list_lines(job["result"]["list"])
        self.assertTrue(lines)
        for ln in lines:
            wav = Path(ln.split("|")[0])
            # <导出根>\<作品名>\<角色名>\<文件>.wav
            self.assertEqual(wav.parent.parent.parent,
                             self.root / "output" / "说话人", str(wav))
            self.assertEqual(wav.parent.parent.name, "无职转生", str(wav))
            self.assertTrue(wav.is_file(), str(wav))

class TestPrecheckAndExportShareOnePlan(ExportApiBase):
    """P4：预检与落盘必须是**同一条计划**，不是各算一遍。

    出事的样子：`_run_export_job_locked` 给 `precheck_work` 传
    `silence_finder=None`（= 没有静音信息 = 不拆长），`export_work` 自己又新建
    一个真 ffmpeg 探测器 —— 于是预检报 2 条片段、落盘写出 3 条，
    预检的磁盘数字也不是真正要写的量。
    """

    def test_one_silence_finder_is_shared_by_both_steps(self):
        job = self.prepare_export()
        job.pop("_export_silence_finder")        # 走生产路径：真的去建探测器
        seen = {}
        real_pre, real_exp = te.precheck_work, te.export_work

        def pre_spy(*a, **kw):
            seen["pre"] = kw.get("silence_finder")
            return real_pre(*a, **kw)

        def exp_spy(*a, **kw):
            seen["exp"] = kw.get("silence_finder")
            seen["plan"] = kw.get("plan")
            return real_exp(*a, **kw)

        with mock.patch.object(webui.te, "plan_silence_finder",
                               wraps=te.plan_silence_finder) as maker, \
             mock.patch.object(webui.te, "precheck_work", side_effect=pre_spy), \
             mock.patch.object(webui.te, "export_work", side_effect=exp_spy):
            webui.run_export_job(job, source_picker=self.fake_picker())

        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(maker.call_count, 1, "静音探测器建了不止一个")
        self.assertIsNotNone(seen.get("pre"), "预检没拿到静音探测器（= 不拆长）")
        self.assertIs(seen["pre"], seen["exp"],
                      "预检与落盘用的不是同一个探测器")
        self.assertIsNotNone(seen.get("plan"), "落盘没有复用预检算出来的计划")

    def test_the_plan_is_computed_exactly_once(self):
        """`_plan_export` 只跑一次（预检那次），落盘复用它。"""
        job = self.prepare_export()
        real = te._plan_export
        calls = []

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        with mock.patch.object(webui.te, "_plan_export", side_effect=counting):
            webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(len(calls), 1,
                         "预检与落盘各算了一遍计划（预检的数字可能不等于落盘）")

    def test_precheck_numbers_equal_the_landed_product(self):
        """真 ffmpeg 探测器 + 一条 30 s 长句：预检报的条数就是落盘的条数。

        这条不 mock 计划本身，只截获预检的返回值 —— 所以它抓的是端到端结果：
        预检说 N 条，报告里就必须是 N 条（长句在有静音信息时会被拆开）。
        """
        (self.workdir / "segments.jsonl").write_text(
            json.dumps({"start": 0.0, "end": 30.0, "text": "一" * 30,
                        "speaker": "SPEAKER_00"}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        src = write_wav(self.root / "src" / "long.wav", 40.0, sr=16000,
                        gap=(15.0, 15.6))
        job = self.prepare_export()
        job.pop("_export_silence_finder")
        job["_export_source"] = (str(src), "separated_hq")
        seen = {}
        real_pre = te.precheck_work

        def spy(*a, **kw):
            out = real_pre(*a, **kw)
            seen.update(out)
            return out

        with mock.patch.object(webui.te, "precheck_work", side_effect=spy):
            webui.run_export_job(job)

        self.assertEqual(job["status"], "done", job.get("error"))
        report = job["result"]["report"]
        self.assertGreater(report["total_clips"], 1,
                           "30 s 的长句没有被拆开（静音探测没跑？）")
        self.assertEqual(seen["n_total"], report["total_clips"],
                         "预检报的片段数与落盘写出的不是一回事")
        self.assertEqual(seen["n_planned"], job["result"]["n_total"])
        self.assertEqual(seen["planned_seconds"], report["planned_seconds"])
        self.assertEqual(seen["estimated_bytes"], report["estimated_bytes"])


class TestAbandonExportKeepsOtherJobsCheckpoint(unittest.TestCase):
    """P2：放弃**排队中**的导出任务，绝不能删掉别的任务的检查点。

    走真 HTTP（`ThreadingHTTPServer` + `webui.Handler`），因为出事的那一行在
    请求处理器里：`/api/abandon` 对 `wait/stopped/error` 的任务直接调
    `_discard_job`，而 `_discard_job` 是按 `job["media"]` 推检查点目录的 ——
    导出任务的 media 就是它引用的那个 ASR 任务的原片，于是「放弃导出」把
    用户命名好的 `speakers.json` / `segments.jsonl` / `diarization.json`
    整份删掉（用户数据丢失，不可恢复）。单元级调 `_discard_job` 覆盖不到
    「处理器到底调没调它」。
    """

    LABELS = {"clusters": {"SPEAKER_00": {"name": "艾丽丝"}},
              "segment_overrides": {}, "work_name": "无职转生"}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="jisheng_abandon_",
                                                dir=str(TMP_ROOT))
        self.root = Path(self._tmp.name)
        self._work_dir = webui.WORK_DIR
        self._out_root = webui.OUTPUT_ROOT
        self._jobs = webui.JOBS
        self._queue = webui.TASK_QUEUE
        webui.WORK_DIR = self.root / "work"
        webui.OUTPUT_ROOT = self.root / "output"
        webui.JOBS = {}
        webui.TASK_QUEUE = queue.Queue()

        self.media = write_wav(self.root / "media" / "EP01.wav", 12.0)
        self.ckpt = webui.WORK_DIR / "EP01"
        self.ckpt.mkdir(parents=True)
        (self.ckpt / "diarization.json").write_text(json.dumps(
            [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
            ensure_ascii=False), encoding="utf-8")
        (self.ckpt / "segments.jsonl").write_text(json.dumps(
            {"start": 0.0, "end": 4.0, "text": "台词", "speaker": "SPEAKER_00"},
            ensure_ascii=False) + "\n", encoding="utf-8")
        (self.ckpt / "speakers.json").write_text(
            json.dumps(self.LABELS, ensure_ascii=False), encoding="utf-8")

        # 一个**已经跑完**的 ASR 任务：检查点就是上面那个，导出任务只是引用它
        webui.JOBS["asr1"] = {"id": "asr1", "media": str(self.media),
                              "work": str(self.ckpt), "logs": [],
                              "status": "done", "stage": "完成"}

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), webui.Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self):
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:                                   # noqa: BLE001
            pass
        webui.WORK_DIR = self._work_dir
        webui.OUTPUT_ROOT = self._out_root
        webui.JOBS = self._jobs
        webui.TASK_QUEUE = self._queue
        self._tmp.cleanup()

    # ---- HTTP 小工具（用真处理器，不重写一遍逻辑）----

    def post(self, path, payload):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def get(self, path):
        with urllib.request.urlopen(
                "http://127.0.0.1:%d%s" % (self.port, path), timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def assert_checkpoint_alive(self, why):
        self.assertTrue(self.ckpt.is_dir(), "检查点目录被删了（%s）" % why)
        for name in ("speakers.json", "segments.jsonl", "diarization.json"):
            self.assertTrue((self.ckpt / name).is_file(),
                            "%s 被删了（%s）" % (name, why))
        self.assertEqual(json.loads((self.ckpt / "speakers.json")
                                    .read_text(encoding="utf-8")), self.LABELS)

    def test_abandoning_a_queued_export_keeps_the_checkpoint(self):
        queued = self.post("/api/export", {"job": "asr1", "work_name": "无职转生"})
        jid = queued["id"]
        self.assertEqual(self.get("/api/status?id=%s" % jid)["status"], "wait")

        self.post("/api/abandon", {"ids": [jid]})

        self.assert_checkpoint_alive("放弃排队中的导出任务")
        self.assertNotIn(jid, webui.JOBS, "放弃的导出任务还留在任务表里")
        # 被引用的那个 ASR 任务本身也不能受牵连
        self.assertIn("asr1", webui.JOBS)
        self.assertEqual(webui.JOBS["asr1"]["status"], "done")

    def test_worker_seeing_an_abandoned_export_keeps_the_checkpoint(self):
        """worker 那条路也不能删（`worker_loop` 对排队中就被放弃的任务同样调
        `_discard_job`）。"""
        queued = self.post("/api/export", {"job": "asr1", "work_name": "无职转生"})
        job = webui.JOBS[queued["id"]]
        job["abandon"] = True
        while True:
            try:
                webui.TASK_QUEUE.get_nowait()
            except queue.Empty:
                break
        webui.TASK_QUEUE.put(job)
        webui.TASK_QUEUE.put(None)
        webui.worker_loop()

        self.assertEqual(job["status"], "abandoned")
        self.assert_checkpoint_alive("worker 处理被放弃的导出任务")

    def test_abandoning_a_real_asr_task_still_deletes_its_checkpoint(self):
        """回归：既有语义（ASR 任务放弃 = 删检查点）一个字都不能改。"""
        self.post("/api/abandon", {"ids": ["asr1"]})
        self.assertFalse(self.ckpt.exists(),
                         "ASR 任务放弃时不再删检查点了（既有语义被破坏）")


class TestAbandonedExportLeavesNothingBehind(ExportApiBase):
    """P6：日志说事实，并且能安全清掉**自己**的半截产物。

    以前中断时日志写「已作废」就完事，而 `<导出根>\\<作品名>\\` 下的半截 wav
    还在盘上。现在：能安全清的清掉并报告，清不掉的把原因写出来。
    """

    def test_abandon_midway_removes_the_half_written_dir(self):
        job = self.prepare_export()
        real = te.cut_segment
        calls = []

        def spy(*a, **kw):
            calls.append(1)
            job["abandon"] = True            # 切完第一条就点「放弃」
            return real(*a, **kw)

        work_dir = self.export_root / "无职转生"
        with mock.patch.object(webui.te, "cut_segment", side_effect=spy):
            webui.run_export_job(job, source_picker=self.fake_picker())

        self.assertEqual(job["status"], "abandoned")
        self.assertTrue(calls, "一条片段都没切？（那这条用例没验到东西）")
        self.assertFalse(work_dir.exists(),
                         "放弃之后半截目录还在：%s" % work_dir)
        self.assertTrue(any("已删除本次导出的半截产物" in ln for ln in job["logs"]),
                        "日志没说清到底删没删：%r" % (job["logs"][-3:],))

    def test_abandon_before_anything_starts_says_nothing_was_written(self):
        """开始前就放弃：连作品名都还没解析，日志必须说「还没开始写盘」。"""
        job = self.prepare_export()
        job["abandon"] = True
        webui.run_export_job(job, source_picker=self.fake_picker())
        self.assertEqual(job["status"], "abandoned")
        self.assertTrue(any("还没开始写盘" in ln for ln in job["logs"]),
                        "日志没有如实说明盘上什么都没有：%r" % (job["logs"][-3:],))

    def test_abandon_during_source_prepare_says_nothing_was_written(self):
        """准备音源时放弃：作品名已解析，但导出根里确实一个文件都没有。"""
        job = self.prepare_export()

        def picker(work, media, mode, **kw):
            job["abandon"] = True                # 用户在准备音源时点了放弃
            raise webui.ts.SourceCancelled("已被停止")

        webui.run_export_job(job, source_picker=picker)
        self.assertEqual(job["status"], "abandoned")
        self.assertTrue(any("没有在导出根里留下产物" in ln for ln in job["logs"]),
                        "日志没有如实说明盘上什么都没有：%r" % (job["logs"][-3:],))

    def test_abandon_keeps_a_previous_successful_export(self):
        """中断发生在「删旧 `.list`」之前时，上一次的产物必须**原封不动**。"""
        first = self.export_job()
        self.assertEqual(first["status"], "done", first.get("error"))
        lines_before, raw_before = self.list_lines(first["result"]["list"])

        job = self.prepare_export()
        job["abandon"] = True
        with mock.patch.object(webui.ts, "pick_source",
                               side_effect=webui.ts.SourceCancelled("已被停止")):
            webui.run_export_job(job)

        self.assertTrue(Path(first["result"]["list"]).is_file(),
                        "上一次的 .list 被删了")
        lines_after, raw_after = self.list_lines(first["result"]["list"])
        self.assertEqual(lines_after, lines_before)
        self.assertEqual(raw_after, raw_before)
        for ln in lines_after:
            self.assertTrue(Path(ln.split("|")[0]).is_file(),
                            "上一次 .list 指向的 wav 被删了：%s" % ln)

    def test_stopped_export_says_the_truth_about_leftovers(self):
        """「停止」（不是放弃）不走清理：日志必须如实说片片段段不完整。"""
        job = self.prepare_export()
        real = te.cut_segment

        def spy(*a, **kw):
            job["stop"] = True
            return real(*a, **kw)

        with mock.patch.object(webui.te, "cut_segment", side_effect=spy):
            webui.run_export_job(job, source_picker=self.fake_picker())

        self.assertEqual(job["status"], "stopped")
        self.assertTrue(any("不完整" in ln and ".list" in ln for ln in job["logs"]),
                        "日志没有如实说明半截产物：%r" % (job["logs"][-3:],))


if __name__ == "__main__":
    unittest.main(verbosity=2)
