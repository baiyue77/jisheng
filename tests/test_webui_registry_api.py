#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Task 7：角色库接口（`GET /api/registry`、rename/forget）、`/api/speakers` 的
`match` 建议，以及 D7「嵌入模型标识从**实际使用的**分割模型推导」。

三条硬不变量（都有用例钉住）：
1. `GET /api/speakers` 是**只读**路径：调完 `speakers.json` 必须逐字节不变
   （不许把建议悄悄写成名字、也不许顺手写库）。
2. 改名 / 删除**只动目标 `(作品, 角色)`**：同作品其它角色、别的作品的向量键一个都不许碰。
3. 换分割模型后，旧模型的向量**不得**参与匹配（`_work_vectors_usable` 判 False）。

库根隔离同 `test_webui_speaker_api.py`：`setUpModule` 指到临时目录，
`tearDownModule` 恢复**进入本模块时**的根 —— 绝不 `set_default_root(None)`
（那是把根指回真实库，Task 5 的 D5 就是踩了这个）。
"""

import json
import math
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jisheng as A          # noqa: E402
import speaker_labels as sl    # noqa: E402
import speaker_registry as sr  # noqa: E402
import webui                   # noqa: E402

_GUARD_ROOT = None
_GUARD_PREV = None


def setUpModule():
    global _GUARD_ROOT, _GUARD_PREV
    _GUARD_ROOT = tempfile.TemporaryDirectory(prefix="jisheng_reg7_guard_")
    _GUARD_PREV = sr.default_root()
    sr.set_default_root(_GUARD_ROOT.name)


def tearDownModule():
    global _GUARD_ROOT
    sr.set_default_root(_GUARD_PREV)
    if _GUARD_ROOT is not None:
        _GUARD_ROOT.cleanup()
        _GUARD_ROOT = None


DIA = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
       {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
       {"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"}]

SEGS = [{"start": 0.0, "end": 4.0, "text": "第一句", "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 15.0, "text": "第二句", "speaker": "SPEAKER_01"},
        {"start": 20.0, "end": 26.0, "text": "第三句", "speaker": "SPEAKER_00"}]

DIM = sr.DEFAULT_DIM
WORK = "无职转生"


def basis(index, dim=DIM):
    v = np.zeros(dim, dtype=np.float64)
    v[index] = 1.0
    return v


def mix(cos, dim=DIM):
    """与 `basis(0)` 余弦**恰为** cos 的单位向量（只在第 0/1 维上混）。"""
    cos = max(-1.0, min(1.0, float(cos)))
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))
    v = np.zeros(dim, dtype=np.float64)
    v[0] = cos
    v[1] = sin
    return v


def near_basis(index, cos, other=0, dim=DIM):
    """与 `basis(index)` 余弦**恰为** cos 的单位向量（剩下的分量放在 `other` 维）。"""
    v = np.zeros(dim, dtype=np.float64)
    v[index] = cos
    v[other] = math.sqrt(max(0.0, 1.0 - cos * cos))
    return v


PIPELINE_CFG = """version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {embedding}
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {segmentation}
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""


def write_pipeline_config(path: Path, embedding: str, version: str = "3.1.0",
                          segmentation: str = "pyannote/segmentation-3.0") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PIPELINE_CFG.format(embedding=embedding,
                                        segmentation=segmentation,
                                        version=version).replace(
        "version: 3.1.0", "version: %s" % version),
        encoding="utf-8")
    return path


class RegistryCase(unittest.TestCase):
    """公共夹具：临时 work / output / JOBS / 库根。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="jisheng_reg7_")
        self.root = Path(self._tmp.name)
        self._work_dir = webui.WORK_DIR
        self._out_root = webui.OUTPUT_ROOT
        self._jobs = webui.JOBS
        webui.WORK_DIR = self.root / "work"
        webui.OUTPUT_ROOT = self.root / "output"
        webui.JOBS = {}
        self._reg_prev = sr.default_root()
        sr.set_default_root(str(self.root / "registry"))
        self.reg_root = str(self.root / "registry")
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
        # 恢复进入本用例前的根，**绝不**写 None（那是指回真实库）
        sr.set_default_root(getattr(self, "_reg_prev", None))
        self._tmp.cleanup()

    # ---- 夹具工具 ----
    def make_job(self, jid="j1", out_dir=None):
        out = Path(out_dir) if out_dir else self.root / "output"
        out.mkdir(parents=True, exist_ok=True)
        webui.JOBS[jid] = {"id": jid, "media": str(self.root / "EP01.mkv"),
                           "work": str(self.workdir), "out_dir": str(out),
                           "logs": [], "status": "done", "stage": "完成"}
        return webui.JOBS[jid]

    def write_state(self, **extra):
        st = {"media": str(self.root / "EP01.mkv"), "media_size": 3,
              "opts_key": "k", "opts": {"diarize": True,
                                        "diarize_model": extra.pop(
                                            "diarize_model",
                                            A.DEFAULT_DIARIZE_MODEL)},
              "out_dir": str(self.root / "output"), "status": "done",
              "total_chunks": len(SEGS)}
        st.update(extra)
        webui.save_state(self.workdir, st)
        return st

    def write_labels(self, clusters, work_name=WORK, job="j1"):
        data = sl.new_labels(job_id=job, media=str(self.root / "EP01.mkv"))
        data["work_name"] = work_name
        data["clusters"] = clusters
        sl.save_labels(self.workdir, data)
        return data

    def seed_page(self, work, page, vector, model=None, job=""):
        """往库里塞一个角色（走真实 `enroll`，不手写文件）。"""
        return sr.enroll(self.reg_root, work=work, page=page,
                         vector=np.asarray(vector, dtype=np.float64),
                         job=job or ("j_" + page), cluster="SPEAKER_00",
                         seconds=10.0, model=model)

    def _reset_library(self):
        """清空临时库与这一集的标注状态（subTest 之间要彻底隔离）。"""
        for path in (sr.registry_path(self.reg_root),
                     sr.vectors_path(self.reg_root),
                     self.workdir / "speakers.json"):
            try:
                Path(path).unlink()
            except OSError:
                pass

    def read_bytes(self, path):
        return Path(path).read_bytes()


# --------------------------------------------------------------------------
# GET /api/registry
# --------------------------------------------------------------------------

class TestRegistryList(RegistryCase):
    def test_empty_registry_has_a_stable_shape(self):
        obj, code = webui.api_registry()
        self.assertEqual(code, 200)
        self.assertEqual(obj["works"], [])
        for key in ("embedding_model", "dim", "thresholds",
                    "n_works", "n_pages", "n_episodes"):
            self.assertIn(key, obj)
        self.assertEqual(obj["n_works"], 0)
        self.assertEqual(obj["n_pages"], 0)
        self.assertEqual(obj["dim"], sr.DEFAULT_DIM)
        self.assertEqual(obj["thresholds"]["auto"], sr.DEFAULT_AUTO)
        self.assertEqual(obj["thresholds"]["suggest"], sr.DEFAULT_SUGGEST)

    def test_lists_works_pages_and_episode_counts(self):
        self.seed_page(WORK, "艾丽丝", basis(0), job="j1")
        self.seed_page(WORK, "艾丽丝", basis(1), job="j2")
        self.seed_page(WORK, "鲁迪乌斯", basis(2), job="j3")
        self.seed_page("别的作品", "路人", basis(3), job="j4")

        obj, code = webui.api_registry()
        self.assertEqual(code, 200)
        self.assertEqual(obj["n_works"], 2)
        self.assertEqual(obj["n_pages"], 3)
        self.assertEqual(obj["n_episodes"], 4)

        works = {w["work"]: w for w in obj["works"]}
        self.assertEqual(sorted(works), sorted(["别的作品", WORK]))
        w = works[WORK]
        self.assertEqual(w["n_pages"], 2)
        self.assertEqual(w["n_episodes"], 3, "同一作品下的集数要按作品汇总")
        pages = {p["page"]: p for p in w["pages"]}
        self.assertEqual(sorted(pages), ["艾丽丝", "鲁迪乌斯"])
        ali = pages["艾丽丝"]
        self.assertEqual(ali["n_enroll"], 2)
        self.assertEqual(ali["n_episodes"], 2, "两集各入了一次库")
        self.assertEqual(sorted(ali["jobs"]), ["j1", "j2"])
        self.assertEqual(ali["n_vectors"], 3, "centroid + seg000 + seg001")
        self.assertTrue(ali["vectors_usable"])
        self.assertAlmostEqual(ali["total_s"], 20.0, places=3)

    def test_get_registry_never_writes(self):
        """只读接口：调用前后两个库文件的字节、mtime 都必须完全一样，
        并且**一次写入口都没被调用过**。

        三层判据缺一不可（与 `/api/speakers` 那条同口径，Task 7 审查 T4：
        这里原本只有字节一层，把写入口 Mock 掉或「同一秒写回同样内容」时
        抓不到）：
        * 字节比对抓**内容**变化；
        * mtime 抓「内容一样也重写了一遍」（同一秒内 `write_bytes` 同内容
          产出的字节完全相同，Task 7 审查 L2 实测过）；
        * 写入口 Mock 抓「文件一个字节没动，但确实调了保存函数」（L3）。
        """
        self.seed_page(WORK, "艾丽丝", basis(0))
        reg_path = sr.registry_path(self.reg_root)
        vec_path = sr.vectors_path(self.reg_root)
        before = (self.read_bytes(reg_path), self.read_bytes(vec_path))
        times = (reg_path.stat().st_mtime_ns, vec_path.stat().st_mtime_ns)

        writers = [
            mock.patch.object(sr, "save_registry"),
            mock.patch.object(sr, "save_vectors"),
            mock.patch.object(sr, "enroll"),
            mock.patch.object(sr, "rename_page"),
            mock.patch.object(sr, "forget_page"),
            mock.patch.object(sr, "register_page"),
            mock.patch.object(sr, "set_work_hint"),
        ]
        mocks = [m.start() for m in writers]
        try:
            for _ in range(3):
                obj, code = webui.api_registry()
                self.assertEqual(code, 200)
        finally:
            for m in writers:
                m.stop()

        called = [m for m in mocks if m.called]
        self.assertEqual(called, [], "只读接口调用了写入口：%r"
                         % ([str(m) for m in called],))
        self.assertEqual(self.read_bytes(reg_path), before[0], "列一次库就改了元数据")
        self.assertEqual(self.read_bytes(vec_path), before[1], "列一次库就改了向量")
        self.assertEqual((reg_path.stat().st_mtime_ns, vec_path.stat().st_mtime_ns),
                         times, "只读接口重写了文件（内容一样也算写了）")

    def test_missing_registry_is_not_created_by_listing(self):
        self.assertFalse(Path(self.reg_root).exists())
        obj, code = webui.api_registry()
        self.assertEqual(code, 200)
        self.assertEqual(obj["works"], [])
        self.assertFalse(sr.registry_path(self.reg_root).exists(),
                         "列一次库就把空库写出来了")


# --------------------------------------------------------------------------
# POST /api/registry/rename
# --------------------------------------------------------------------------

class TestRegistryRename(RegistryCase):
    def test_rename_moves_all_vectors_and_metadata(self):
        self.seed_page(WORK, "爱丽丝", basis(0), job="j1")
        self.seed_page(WORK, "爱丽丝", basis(1), job="j2")

        obj, code = webui.api_registry_rename(
            {"work": WORK, "from": "爱丽丝", "to": "艾丽丝"})
        self.assertEqual(code, 200)
        self.assertTrue(obj["ok"])
        self.assertEqual(obj["moved"], 3, "centroid + seg000 + seg001 都要跟着走")

        vecs = sr.load_vectors(self.reg_root)
        self.assertIn(sr.page_centroid_key(WORK, "艾丽丝"), vecs)
        self.assertIn(sr.page_segment_key(WORK, "艾丽丝", 1), vecs)
        self.assertNotIn(sr.page_centroid_key(WORK, "爱丽丝"), vecs)

        rows = {(r["work"], r["page"]): r for r in sr.list_registry(self.reg_root)}
        self.assertIn((WORK, "艾丽丝"), rows)
        self.assertNotIn((WORK, "爱丽丝"), rows)
        self.assertEqual(rows[(WORK, "艾丽丝")]["n_enroll"], 2)

        listing, _ = webui.api_registry()
        self.assertEqual([p["page"] for p in listing["works"][0]["pages"]],
                         ["艾丽丝"])

    def test_rename_only_touches_the_target_page(self):
        """只动目标的向量键：同作品其它角色、别的作品的同名人都不许碰。"""
        self.seed_page(WORK, "爱丽丝", basis(0))
        self.seed_page(WORK, "鲁迪乌斯", basis(2))
        self.seed_page("别的作品", "爱丽丝", basis(3))

        other_keys = {
            "本作品别人": sr.page_centroid_key(WORK, "鲁迪乌斯"),
            "本作品别人seg": sr.page_segment_key(WORK, "鲁迪乌斯", 0),
            "别的作品同名": sr.page_centroid_key("别的作品", "爱丽丝"),
            "别的作品同名seg": sr.page_segment_key("别的作品", "爱丽丝", 0),
        }
        before = {tag: np.array(sr.load_vectors(self.reg_root).get(key))
                  for tag, key in other_keys.items()}

        obj, code = webui.api_registry_rename(
            {"work": WORK, "from": "爱丽丝", "to": "艾丽丝"})
        self.assertEqual(code, 200)

        after = sr.load_vectors(self.reg_root)
        for tag, key in other_keys.items():
            self.assertIn(key, after, "%s 的向量键被误动了" % tag)
            np.testing.assert_array_equal(after[key], before[tag])
        rows = {(r["work"], r["page"]) for r in sr.list_registry(self.reg_root)}
        self.assertIn((WORK, "鲁迪乌斯"), rows)
        self.assertIn(("别的作品", "爱丽丝"), rows)

    def test_rename_onto_existing_page_reports_merged(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        self.seed_page(WORK, "艾丽丝", basis(1))
        obj, code = webui.api_registry_rename(
            {"work": WORK, "from": "爱丽丝", "to": "艾丽丝"})
        self.assertEqual(code, 200)
        self.assertTrue(obj["merged"], "并入已有角色必须能看出来")
        rows = {(r["work"], r["page"]): r for r in sr.list_registry(self.reg_root)}
        self.assertEqual(rows[(WORK, "艾丽丝")]["n_enroll"], 2)

    def test_rename_unknown_page_is_404(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        obj, code = webui.api_registry_rename(
            {"work": WORK, "from": "不存在", "to": "X"})
        self.assertEqual(code, 404)
        self.assertIn("error", obj)

    def test_rename_unknown_work_is_404(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        obj, code = webui.api_registry_rename(
            {"work": "没这个作品", "from": "爱丽丝", "to": "X"})
        self.assertEqual(code, 404)
        self.assertTrue(obj.get("error"))

    def test_rename_missing_params_is_400(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        cases = [{"from": "爱丽丝", "to": "X"},
                 {"work": WORK, "to": "X"},
                 {"work": WORK, "from": "爱丽丝"},
                 {"work": "  ", "from": "爱丽丝", "to": "X"},
                 {"work": WORK, "from": "  ", "to": "X"},
                 {"work": WORK, "from": "爱丽丝", "to": "  "}]
        for payload in cases:
            with self.subTest(payload=payload):
                obj, code = webui.api_registry_rename(payload)
                self.assertEqual(code, 400, payload)
                self.assertTrue(obj.get("error"))

    def test_rename_bad_types_is_400(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        for payload in ({"work": 5, "from": "爱丽丝", "to": "X"},
                        {"work": WORK, "from": ["爱丽丝"], "to": "X"},
                        {"work": WORK, "from": "爱丽丝", "to": {"a": 1}}):
            with self.subTest(payload=payload):
                obj, code = webui.api_registry_rename(payload)
                self.assertEqual(code, 400, payload)

    def test_rename_to_the_same_name_is_400(self):
        """同名不是「成功」，别让面板显示一个什么都没发生的成功。"""
        self.seed_page(WORK, "爱丽丝", basis(0))
        obj, code = webui.api_registry_rename(
            {"work": WORK, "from": "爱丽丝", "to": "爱丽丝"})
        self.assertEqual(code, 400)
        self.assertTrue(obj.get("error"))

    def test_non_object_payload_is_400(self):
        for bad in ([], "x", 5, None):
            with self.subTest(bad=bad):
                obj, code = webui.api_registry_rename(bad)
                self.assertEqual(code, 400)
                self.assertTrue(obj.get("error"))
                obj, code = webui.api_registry_forget(bad)
                self.assertEqual(code, 400)
                self.assertTrue(obj.get("error"))


# --------------------------------------------------------------------------
# POST /api/registry/forget
# --------------------------------------------------------------------------

class TestRegistryForget(RegistryCase):
    def test_forget_removes_the_page_and_its_vectors(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        self.seed_page(WORK, "爱丽丝", basis(1))

        obj, code = webui.api_registry_forget({"work": WORK, "page": "爱丽丝"})
        self.assertEqual(code, 200)
        self.assertTrue(obj["ok"])
        self.assertEqual(obj["removed"], 3)

        rows = {(r["work"], r["page"]) for r in sr.list_registry(self.reg_root)}
        self.assertNotIn((WORK, "爱丽丝"), rows)
        vecs = sr.load_vectors(self.reg_root)
        self.assertFalse([k for k in vecs if k.startswith("page__")
                          and "爱丽丝" in k],
                         "向量没删干净：%r" % (sorted(vecs),))
        # `job__*` 是审计键（设计 §6.2），forget_page 明确保留它们
        self.assertTrue([k for k in vecs if k.startswith("job__")],
                        "审计用的 job__ 键不该被删（那不是角色数据）")

    def test_forget_keeps_other_pages_and_other_works(self):
        """不可逆动作：只许删掉那一个 (作品, 角色)。"""
        self.seed_page(WORK, "爱丽丝", basis(0))
        self.seed_page(WORK, "鲁迪乌斯", basis(1))
        self.seed_page("别的作品", "爱丽丝", basis(2))

        obj, code = webui.api_registry_forget({"work": WORK, "page": "爱丽丝"})
        self.assertEqual(code, 200)

        rows = {(r["work"], r["page"]): r for r in sr.list_registry(self.reg_root)}
        self.assertNotIn((WORK, "爱丽丝"), rows)
        self.assertIn((WORK, "鲁迪乌斯"), rows)
        self.assertIn(("别的作品", "爱丽丝"), rows)
        self.assertEqual(rows[(WORK, "鲁迪乌斯")]["n_vectors"], 2)

        vecs = sr.load_vectors(self.reg_root)
        self.assertIn(sr.page_centroid_key(WORK, "鲁迪乌斯"), vecs)
        self.assertIn(sr.page_centroid_key("别的作品", "爱丽丝"), vecs)
        self.assertNotIn(sr.page_centroid_key(WORK, "爱丽丝"), vecs)

    def test_forget_unknown_page_is_404(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        obj, code = webui.api_registry_forget({"work": WORK, "page": "不存在"})
        self.assertEqual(code, 404)
        self.assertTrue(obj.get("error"))
        rows = {(r["work"], r["page"]) for r in sr.list_registry(self.reg_root)}
        self.assertIn((WORK, "爱丽丝"), rows)

    def test_forget_unknown_work_is_404(self):
        obj, code = webui.api_registry_forget({"work": "没这个作品",
                                               "page": "爱丽丝"})
        self.assertEqual(code, 404)
        self.assertTrue(obj.get("error"))

    def test_forget_missing_params_is_400(self):
        self.seed_page(WORK, "爱丽丝", basis(0))
        for payload in ({}, {"work": WORK}, {"page": "爱丽丝"},
                        {"work": "", "page": "爱丽丝"},
                        {"work": WORK, "page": "   "},
                        {"work": 7, "page": "爱丽丝"},
                        {"work": WORK, "page": None}):
            with self.subTest(payload=payload):
                obj, code = webui.api_registry_forget(payload)
                self.assertEqual(code, 400, payload)
                self.assertTrue(obj.get("error"))
        rows = {(r["work"], r["page"]) for r in sr.list_registry(self.reg_root)}
        self.assertIn((WORK, "爱丽丝"), rows, "400 的请求居然把角色删了")


# --------------------------------------------------------------------------
# /api/speakers 的 match
# --------------------------------------------------------------------------

class TestSpeakersMatch(RegistryCase):
    def _seed_cluster(self, cluster, vector, name=""):
        labels = sl.load_labels(self.workdir)
        labels["work_name"] = WORK
        labels["clusters"].setdefault(cluster, {"name": name})
        if vector is not None:
            try:
                raw = [float(x) for x in np.asarray(vector, dtype=np.float64)]
            except (TypeError, ValueError):
                raw = vector          # 手改坏的值要能原样写进去
            labels["clusters"][cluster]["embedding"] = raw
        sl.save_labels(self.workdir, labels)

    def test_match_above_threshold_is_auto(self):
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.91))

        obj, code = webui.api_speakers("j1")
        self.assertEqual(code, 200)
        match = obj["clusters"][0]["match"]
        self.assertIsNotNone(match, "0.91 的相似度没有给出建议")
        self.assertEqual(match["work"], WORK)
        self.assertEqual(match["page"], "艾丽丝")
        self.assertEqual(match["band"], "auto")
        self.assertEqual(match["auto"], "艾丽丝", "auto 档必须能预填")
        self.assertAlmostEqual(match["score"], 0.91, places=4)

    def test_match_in_the_suggest_band_is_not_auto(self):
        """0.48（suggest 带）只能当候选，绝不能当「可预填」。"""
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.48))

        obj, _ = webui.api_speakers("j1")
        match = obj["clusters"][0]["match"]
        self.assertIsNotNone(match)
        self.assertEqual(match["band"], "suggest")
        self.assertIsNone(match["auto"], "suggest 档不许预填")
        self.assertAlmostEqual(match["score"], 0.48, places=4)

    def test_band_uses_the_library_thresholds_not_a_hardcoded_one(self):
        """阈值可配置：库里把 auto 调到 0.9 之后 0.8 就不再是 auto。"""
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        reg = sr.load_registry(self.reg_root)
        reg["thresholds"] = {"auto": 0.9, "suggest": 0.45}
        sr.save_registry(self.reg_root, reg)
        self._seed_cluster("SPEAKER_00", mix(0.8))
        obj, _ = webui.api_speakers("j1")
        self.assertEqual(obj["clusters"][0]["match"]["band"], "suggest")
        self.assertIsNone(obj["clusters"][0]["match"]["auto"])

    def test_match_below_the_suggest_band_is_reported_as_none_band(self):
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.30))
        obj, _ = webui.api_speakers("j1")
        match = obj["clusters"][0]["match"]
        self.assertIsNotNone(match, "有候选就该报出来，让面板自己按 band 决定显不显示")
        self.assertEqual(match["band"], "none")
        self.assertIsNone(match["auto"])

    def test_match_is_null_without_embedding(self):
        """这次分割没拿到质心 → match 必须是 null，不许编造。"""
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", None)
        obj, _ = webui.api_speakers("j1")
        self.assertIsNone(obj["clusters"][0]["match"])

    def test_match_is_null_when_the_cluster_vector_is_broken(self):
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        for bad in ([0.1, 0.2, 0.3], [float("nan")] * DIM, "not-a-vector", {}):
            with self.subTest(bad=repr(bad)[:20]):
                self._seed_cluster("SPEAKER_00", bad)
                obj, _ = webui.api_speakers("j1")
                self.assertIsNone(obj["clusters"][0]["match"],
                                  "坏向量也算出了建议：%r" % (bad,))

    def test_match_is_null_without_a_work_name(self):
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.99))
        labels = sl.load_labels(self.workdir)
        labels["work_name"] = ""
        sl.save_labels(self.workdir, labels)
        obj, _ = webui.api_speakers("j1")
        self.assertIsNone(obj["clusters"][0]["match"],
                          "没有作品名却给出了跨集建议")

    def test_named_cluster_is_not_matched_to_itself(self):
        """自我匹配：库里只有它自己的名字，就得老实说「没建议」。

        库里那个角色的向量与它**逐位相同**（余弦 1.0）；不过滤自己就会得到
        「建议：艾丽丝（1.00）」，而它本来就叫艾丽丝 —— 一句废话，
        而且是 Task 5 审查 M4 变异抓到过的同一类错误。
        """
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", basis(0), name="艾丽丝")
        obj, _ = webui.api_speakers("j1")
        self.assertIsNone(obj["clusters"][0]["match"],
                          "已命名为艾丽丝的聚类又建议改名成艾丽丝")

    def test_a_named_cluster_can_still_match_a_different_role(self):
        """过滤的只是「自己」：别人（哪怕是同作品的另一个角色）照样能建议。"""
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self.seed_page(WORK, "鲁迪乌斯", basis(1))
        self._seed_cluster("SPEAKER_00", basis(0), name="艾丽丝")
        obj, _ = webui.api_speakers("j1")
        match = obj["clusters"][0]["match"]
        self.assertIsNotNone(match)
        self.assertEqual(match["page"], "鲁迪乌斯",
                         "把自己过滤掉之后，库里别的角色仍必须是候选")

    def test_two_clusters_do_not_both_get_the_same_name(self):
        """一对一：同一个角色在一集里最多配一个聚类（分数高的那个）。"""
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.95))
        self._seed_cluster("SPEAKER_01", mix(0.80))

        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertEqual(matches["SPEAKER_00"]["auto"], "艾丽丝")
        self.assertIsNone(matches["SPEAKER_01"]["auto"],
                          "同一个角色被预填给了两个聚类")
        self.assertEqual(matches["SPEAKER_01"]["page"], "艾丽丝")

    def test_match_ignores_vectors_from_another_model(self):
        """库级模型已经不是这个作品入库时那个 → 它的向量不参与匹配。

        库只记一个全局 `embedding_model`（设计 §10.8），`_work_vectors_usable`
        按「作品记录 vs 当前模型」判定；这里把全局改成 v2，本作品记的是 v1。
        """
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0), model="别家模型/v1")
        self.seed_page("别的作品", "路人", basis(1), model="别家模型/v2")
        reg = sr.load_registry(self.reg_root)
        self.assertEqual(reg["embedding_model"], "别家模型/v2")
        self.assertFalse(sr._work_vectors_usable(reg["works"][WORK],
                                                 reg["embedding_model"]))
        # 面板这次的质心就是用 v2 算的（statu.json 记着），所以不是「跨空间比较」
        # 被拦下来的，而是**本作品的旧向量**被模型判定挡住了。
        self.write_state(diarize_model="别家模型/v2")
        self._seed_cluster("SPEAKER_00", basis(0), name="")

        obj, _ = webui.api_speakers("j1")
        self.assertIsNone(obj["clusters"][0]["match"],
                          "换过模型的旧向量参与了匹配")

    def test_match_is_skipped_when_the_library_is_in_another_space(self):
        """这批质心与库里的向量不是一个模型算的 → 一个建议都不许给。

        `match_vectors` 只用库里的全局模型判定**库**那一侧的可用性，它对
        「这次的质心是哪个模型算的」一无所知 —— 不在这里先比一次的话，
        A 空间的质心就会去跟 B 空间的库向量算余弦（D7 要防的静默混比）。
        """
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0), model="别家模型/v2")
        self.write_state(diarize_model="本地自制/config.yaml")
        self._seed_cluster("SPEAKER_00", basis(0))

        obj, _ = webui.api_speakers("j1")
        self.assertIsNone(obj["clusters"][0]["match"],
                          "跨模型的余弦被算出来了")

    def test_a_named_cluster_does_not_steal_the_one_to_one_slot(self):
        """一对一的名额只该给**还没命名**的聚类。

        库里同时有「艾丽丝」与「鲁迪乌斯」：
        * SPEAKER_00 已经叫艾丽丝，但它的向量与**鲁迪乌斯**的余弦是 0.90；
        * SPEAKER_01 还没名字，与鲁迪乌斯的余弦是 0.80。

        两条都是 auto 档，一对一只能给一个。名额必须给**未命名**的那个 ——
        已命名的聚类根本不会被预填（面板里名字优先于建议），名额给它就是白占。
        变异「去掉『已命名不占名额』这条规则」在这里红：SPEAKER_00 会以 0.90
        抢走鲁迪乌斯，SPEAKER_01 拿不到任何 auto。

        （旧版本的这条用例**测不到这条规则**：那里已命名的聚类唯一的候选就是它
        自己的名字，早被自我过滤删掉了，加不加 named 判断结果一样。）
        """
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self.seed_page(WORK, "鲁迪乌斯", basis(1))
        self._seed_cluster("SPEAKER_00", near_basis(1, 0.90, other=2), name="艾丽丝")
        self._seed_cluster("SPEAKER_01", near_basis(1, 0.80, other=3), name="")

        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertEqual(matches["SPEAKER_01"]["auto"], "鲁迪乌斯",
                         "未命名的聚类没拿到本该属于它的建议：%r"
                         % (matches["SPEAKER_01"],))
        self.assertEqual(matches["SPEAKER_01"]["band"], "auto")
        self.assertIsNone(matches["SPEAKER_00"]["auto"],
                          "已经有名字的聚类还在占一对一的名额：%r"
                          % (matches["SPEAKER_00"],))

    def test_one_to_one_runs_on_the_self_filtered_candidates(self):
        """一对一必须用**自我过滤之后**的候选（上一条测的是「已命名不占名额」）。

        两个规则是两件事，得分开钉：
        * SPEAKER_00 已经叫「艾丽丝」，它的自我匹配是 1.0（被过滤掉）；
        * SPEAKER_01 还没名字，与「艾丽丝」0.92 —— 它要的就是这个名额。

        配对如果拿**过滤前**的候选，SPEAKER_00 会先用自己那条 1.0 占掉艾丽丝
        （新候选语义下就是「过滤前候选 + 丢掉 named 判断」，即 Task 7 之前那版
        两段式 pool 的写法），SPEAKER_01 于是一个 auto 都拿不到。
        """
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self.seed_page(WORK, "鲁迪乌斯", basis(1))
        self._seed_cluster("SPEAKER_00", basis(0), name="艾丽丝")   # 自我匹配 1.0
        self._seed_cluster("SPEAKER_01", near_basis(0, 0.92, other=4), name="")

        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertEqual(matches["SPEAKER_01"]["auto"], "艾丽丝",
                         "已命名聚类自己的名字占掉了一对一的名额：%r"
                         % (matches["SPEAKER_01"],))
        self.assertEqual(matches["SPEAKER_01"]["page"], "艾丽丝")
        self.assertIsNone(matches["SPEAKER_00"]["auto"])

    def test_speakers_is_read_only(self):
        """硬不变量：调完面板接口，标注状态与库都必须逐字节不变。

        两道判据缺一不可：
        * 字节比对抓得住**内容**变化（建议被写成了名字、库被写了）；
        * 但 `save_labels` 在同一秒里重写同一份内容会产出**完全相同的字节**
          （updated_at 精度是秒）—— 所以还要把「写入口」整个换成替身，
          断言它一次都没被调用（mtime 也一起比，多一层兜底）。
        """
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.95))
        self._seed_cluster("SPEAKER_01", mix(0.5))

        labels_path = self.workdir / "speakers.json"
        reg_path = sr.registry_path(self.reg_root)
        vec_path = sr.vectors_path(self.reg_root)
        before = (self.read_bytes(labels_path), self.read_bytes(reg_path),
                  self.read_bytes(vec_path))
        times = (labels_path.stat().st_mtime_ns, reg_path.stat().st_mtime_ns,
                 vec_path.stat().st_mtime_ns)

        writers = [
            mock.patch.object(sl, "save_labels"),
            mock.patch.object(sr, "save_registry"),
            mock.patch.object(sr, "save_vectors"),
            mock.patch.object(sr, "enroll"),
            mock.patch.object(sr, "rename_page"),
            mock.patch.object(sr, "forget_page"),
            mock.patch.object(sr, "register_page"),
        ]
        mocks = [m.start() for m in writers]
        try:
            for _ in range(3):
                obj, code = webui.api_speakers("j1")
                self.assertEqual(code, 200)
                self.assertIsNotNone(obj["clusters"][0]["match"])
        finally:
            for m in writers:
                m.stop()

        called = [m for m in mocks if m.called]
        self.assertEqual(called, [], "只读接口调用了写入口：%r"
                         % ([str(m) for m in called],))
        after = (self.read_bytes(labels_path), self.read_bytes(reg_path),
                 self.read_bytes(vec_path))
        self.assertEqual(after[0], before[0],
                         "只读接口把 speakers.json 改了（建议被写成了名字？）")
        self.assertEqual(after[1], before[1], "只读接口写了角色库")
        self.assertEqual(after[2], before[2], "只读接口写了向量库")
        self.assertEqual((labels_path.stat().st_mtime_ns,
                          reg_path.stat().st_mtime_ns,
                          vec_path.stat().st_mtime_ns), times,
                         "只读接口重写了文件（内容一样也算写了）")

    def test_match_does_not_leak_into_other_clusters(self):
        """没向量的聚类不许因为别的聚类有向量就跟着拿到建议。"""
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))
        self._seed_cluster("SPEAKER_00", mix(0.95))
        self._seed_cluster("SPEAKER_01", None)
        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertIsNotNone(matches["SPEAKER_00"])
        self.assertIsNone(matches["SPEAKER_01"])


# --------------------------------------------------------------------------
# HTTP 路由（模块级函数全绿也可能根本没接上线）
# --------------------------------------------------------------------------

class TestRegistryHttpRoutes(RegistryCase):
    def setUp(self):
        super().setUp()
        import threading
        from http.server import ThreadingHTTPServer
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        daemon=True)
        self._thread.start()
        self.base = "http://127.0.0.1:%d" % self._srv.server_address[1]
        self.make_job()
        self.seed_page(WORK, "艾丽丝", basis(0))

    def tearDown(self):
        try:
            self._srv.shutdown()
            self._srv.server_close()
        finally:
            super().tearDown()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_get_registry_route_is_wired(self):
        status, obj = self._get("/api/registry")
        self.assertEqual(status, 200)
        self.assertEqual(obj["n_works"], 1)
        self.assertEqual(obj["works"][0]["work"], WORK)

    def test_get_speakers_route_carries_match(self):
        self.write_labels({"SPEAKER_00": {"name": "", "embedding": [
            float(x) for x in mix(0.95)]}, "SPEAKER_01": {"name": ""}})
        status, obj = self._get("/api/speakers?job=j1")
        self.assertEqual(status, 200)
        self.assertEqual(obj["clusters"][0]["match"]["page"], "艾丽丝")

    def test_rename_route_is_wired_and_validates(self):
        status, obj = self._post("/api/registry/rename",
                                 {"work": WORK, "from": "艾丽丝", "to": "爱丽丝"})
        self.assertEqual(status, 200)
        self.assertTrue(obj["ok"])
        status, obj = self._post("/api/registry/rename",
                                 {"work": WORK, "from": "爱丽丝"})
        self.assertEqual(status, 400, "缺参数必须是 4xx，不能 500")
        self.assertTrue(obj.get("error"))

    def test_forget_route_is_wired_and_validates(self):
        status, obj = self._post("/api/registry/forget",
                                 {"work": WORK, "page": "不存在"})
        self.assertEqual(status, 404)
        self.assertTrue(obj.get("error"))
        status, obj = self._post("/api/registry/forget", {"work": WORK})
        self.assertEqual(status, 400)
        status, obj = self._post("/api/registry/forget",
                                 {"work": WORK, "page": "艾丽丝"})
        self.assertEqual(status, 200)
        self.assertTrue(obj["ok"])


# --------------------------------------------------------------------------
# D7：嵌入模型标识从实际分割模型推导
# --------------------------------------------------------------------------

class TestLibraryDimSelfHealsThroughTheApi(RegistryCase):
    """Task 7 复审 D4：库级 `dim` 被污染后，走**面板真实入库路径**必须自愈。

    `TestLibraryDimSelfHeals`（tests/test_speaker_registry.py）只调
    `speaker_registry.enroll` —— 它自己按向量重量维度，**根本没走**
    `webui._expected_embedding_dim()` 这条它 docstring 里点名的路径。于是把该函数
    改成「去读库里的 dim」在全量 741 条下零红（M5）。

    真去读库时的后果：污染值 3 把 256 维向量判成非法 ⇒ `vector=None` ⇒ `enroll`
    再也量不出维度 ⇒ `dim` 永远是 3、库彻底卡死。所以这条路径必须**恒 256**。
    """

    def _pipeline(self, name, embedding_dir):
        return write_pipeline_config(
            self.root / "pipes" / name / "config.yaml",
            embedding="D:/models/%s/pytorch_model.bin" % embedding_dir)

    def test_a_256_dim_assign_repairs_a_polluted_library_dim(self):
        self.make_job()
        cfg = self._pipeline("b", "other-embedder-256")
        self.write_state(diarize_model=str(cfg))
        # 污染：手改出来的短向量曾经把库级 dim 写成 3（`_check_embedding` 就是为此）
        sr.save_registry(self.reg_root, {"embedding_model": "m", "dim": 3})
        self.assertEqual(sr.load_registry(self.reg_root)["dim"], 3)

        v = basis(0)
        self.write_labels({"SPEAKER_00": {"name": "",
                                          "embedding": [float(x) for x in v]}})
        obj, code = webui.api_assign({"job": "j1", "work_name": WORK,
                                      "names": {"SPEAKER_00": "艾丽丝"}})
        self.assertEqual(code, 200, obj)

        self.assertEqual(sr.load_registry(self.reg_root)["dim"], DIM,
                         "库级 dim 没自愈（256 维向量被当成非法？）")
        vecs = sr.load_vectors(self.reg_root)
        self.assertIn(sr.page_centroid_key(WORK, "艾丽丝"), vecs,
                      "只登记了角色、向量没落盘")
        np.testing.assert_allclose(vecs[sr.page_centroid_key(WORK, "艾丽丝")], v,
                                   rtol=0, atol=1e-6)
        # 修好之后立刻能匹配（不是「写进去但谁也用不了」）
        out = sr.match_vectors(self.reg_root, work=WORK, clusters={"K": v})
        self.assertEqual(out["K"]["top"]["page"], "艾丽丝")
        self.assertEqual(out["K"]["top"]["score"], 1.0)
        self.assertEqual(webui._expected_embedding_dim(), sr.DEFAULT_DIM)


class TestEmbeddingModelTag(unittest.TestCase):
    """`webui.embedding_model_tag()`：身份从**实际用的**分割模型推导。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="jisheng_reg7_tag_")
        self.root = Path(self._tmp.name)
        self._prev_local = A.LOCAL_PYANNOTE_DIR
        A.LOCAL_PYANNOTE_DIR = self.root / "models" / "pyannote"
        # 缓存是按 ref 键的：换过本地目录后必须别拿到别人算出来的旧值
        webui._embedding_tag_cache_clear()

    def tearDown(self):
        A.LOCAL_PYANNOTE_DIR = self._prev_local
        webui._embedding_tag_cache_clear()
        self._tmp.cleanup()

    def _make_local_models(self, embedding_dir="wespeaker-voxceleb-resnet34-LM",
                           embedding_ref=None):
        """造一份与真实布局一致的本地 pyannote 模型目录。"""
        base = A.LOCAL_PYANNOTE_DIR
        (base / "segmentation-3.0").mkdir(parents=True, exist_ok=True)
        (base / "segmentation-3.0" / "pytorch_model.bin").write_bytes(b"seg")
        (base / embedding_dir).mkdir(parents=True, exist_ok=True)
        (base / embedding_dir / "pytorch_model.bin").write_bytes(b"emb")
        src = base / "speaker-diarization-3.1" / "config.yaml"
        write_pipeline_config(
            src, embedding_ref or "pyannote/wespeaker-voxceleb-resnet34-LM")
        return base

    def test_local_pipeline_tag_equals_the_shipped_constant(self):
        """本机默认这条路：推导出来的标识必须**与旧常量逐字相同**。

        否则升级一次就让用户已有的库整体失效（旧向量会被判成别的模型算的），
        而向量空间其实一个字节都没变。
        """
        self._make_local_models()
        self.assertEqual(webui.embedding_model_tag(None),
                         webui.EMBEDDING_MODEL_TAG)

    def test_tag_uses_the_embedding_directory_name(self):
        """本地生成的 config 把 HF 名字换成了权重路径 → 身份取**目录名**。"""
        cfg = write_pipeline_config(
            self.root / "custom" / "config.yaml",
            embedding="D:/models/other-embedder-256/pytorch_model.bin")
        tag = webui.embedding_model_tag(str(cfg))
        self.assertTrue(tag.endswith("/other-embedder-256"), tag)
        self.assertNotEqual(tag, webui.EMBEDDING_MODEL_TAG)

    def test_hf_repo_id_and_weight_path_give_the_same_identity(self):
        """在线写的 `pyannote/wespeaker-...` 与本地权重路径是同一个空间。"""
        cfg = write_pipeline_config(
            self.root / "online" / "config.yaml",
            embedding="pyannote/wespeaker-voxceleb-resnet34-LM")
        self.assertEqual(webui.embedding_model_tag(str(cfg)),
                         webui.EMBEDDING_MODEL_TAG)

    def test_pipeline_version_is_part_of_the_identity(self):
        """2.x 的流水线（嵌入式模型完全不同）不许与 3.1 混成同一个身份。"""
        cfg = write_pipeline_config(
            self.root / "old" / "config.yaml",
            embedding="D:/models/old-embedder/pytorch_model.bin",
            version="2.1.0")
        tag = webui.embedding_model_tag(str(cfg))
        self.assertTrue(tag.startswith("pyannote/speaker-diarization-2.1/"), tag)
        self.assertNotEqual(tag, webui.EMBEDDING_MODEL_TAG)

    def test_other_online_pipeline_is_not_the_default_tag(self):
        """在线选择的下拉第二项（旧版流水线）必须有自己的身份。"""
        tag = webui.embedding_model_tag("pyannote/speaker-diarization")
        self.assertNotEqual(tag, webui.EMBEDDING_MODEL_TAG)
        self.assertEqual(tag, "pyannote/speaker-diarization")

    def test_default_model_without_local_files_falls_back_to_the_constant(self):
        """推导不出来也不能让入库失败：回退到常量。"""
        self.assertEqual(webui.embedding_model_tag(None),
                         webui.EMBEDDING_MODEL_TAG)
        self.assertEqual(webui.embedding_model_tag(A.DEFAULT_DIARIZE_MODEL),
                         webui.EMBEDDING_MODEL_TAG)
        self.assertEqual(webui.embedding_model_tag(""), webui.EMBEDDING_MODEL_TAG)

    def test_derivation_never_raises_and_never_returns_empty(self):
        """任何畸形输入都不许抛，也不许返回空串（空串会被当成「没记模型」=通配）。"""
        junk = [None, "", "   ", 5, ["a"], {"a": 1},
                str(self.root / "nope" / "config.yaml"),
                str(self.root), "x" * 400]
        bad_cfg = self.root / "bad" / "config.yaml"
        bad_cfg.parent.mkdir(parents=True, exist_ok=True)
        bad_cfg.write_bytes(b"\x00\x01\x02 not yaml at all")
        junk.append(str(bad_cfg))
        for ref in junk:
            with self.subTest(ref=repr(ref)[:40]):
                tag = webui.embedding_model_tag(ref)
                self.assertIsInstance(tag, str)
                self.assertTrue(tag.strip(), "回退值不能是空串：%r" % (ref,))

    def test_tag_is_cached_but_follows_the_config_content(self):
        cfg = write_pipeline_config(
            self.root / "c" / "config.yaml",
            embedding="D:/models/aaa/pytorch_model.bin")
        with mock.patch.object(webui, "_parse_pipeline_identity",
                               wraps=webui._parse_pipeline_identity) as spy:
            first = webui.embedding_model_tag(str(cfg))
            second = webui.embedding_model_tag(str(cfg))
            self.assertEqual(first, second)
            self.assertEqual(spy.call_count, 1, "每次命名都去读盘解析了一遍 config")

        write_pipeline_config(cfg,
                              embedding="D:/models/bbbb/pytorch_model.bin")
        third = webui.embedding_model_tag(str(cfg))
        self.assertNotEqual(third, first, "改了 config 还拿着旧身份（缓存没失效）")
        self.assertTrue(third.endswith("/bbbb"), third)

    def test_cache_key_is_the_model_not_the_work(self):
        """同一份 config 在多次调用间必须命中缓存（不按作品重复读盘）。"""
        cfg = write_pipeline_config(
            self.root / "d" / "config.yaml",
            embedding="D:/models/ccc/pytorch_model.bin")
        tags = {webui.embedding_model_tag(str(cfg)) for _ in range(5)}
        self.assertEqual(len(tags), 1)


class TestModelSwitchInvalidatesVectors(RegistryCase):
    """D7 的判据：换 `diarize_model` 之后旧模型的向量不得参与匹配。"""

    def _pipeline(self, name, embedding_dir):
        return write_pipeline_config(
            self.root / "pipes" / name / "config.yaml",
            embedding="D:/models/%s/pytorch_model.bin" % embedding_dir)

    def _name_cluster(self, cluster, vector, page):
        labels = sl.load_labels(self.workdir)
        labels["work_name"] = WORK
        labels["clusters"].setdefault(cluster, {"name": ""})
        labels["clusters"][cluster]["embedding"] = [
            float(x) for x in np.asarray(vector, dtype=np.float64)]
        sl.save_labels(self.workdir, labels)
        return webui.api_assign({"job": "j1", "work_name": WORK,
                                 "names": {cluster: page}})

    def test_switching_the_model_makes_the_old_vectors_unusable(self):
        self.make_job()
        cfg_a = self._pipeline("a", "wespeaker-voxceleb-resnet34-LM")
        cfg_b = self._pipeline("b", "other-embedder-256")
        tag_a = webui.embedding_model_tag(str(cfg_a))
        tag_b = webui.embedding_model_tag(str(cfg_b))
        self.assertNotEqual(tag_a, tag_b)

        self.write_state(diarize_model=str(cfg_a))
        obj, code = self._name_cluster("SPEAKER_00", mix(0.9), "艾丽丝")
        self.assertEqual(code, 200, obj)
        tag_a = webui.embedding_model_tag(str(cfg_a))   # 命名时真正用的那个
        reg = sr.load_registry(self.reg_root)
        self.assertEqual(reg["embedding_model"], tag_a,
                         "库级模型标识不是从实际分割模型推导的（还是写死的常量？）")
        self.assertTrue(sr._work_vectors_usable(reg["works"][WORK], tag_a))
        self.assertFalse(sr._work_vectors_usable(reg["works"][WORK], tag_b),
                         "换了模型，旧模型的向量还判成可用")

        # 用模型 B 再命一次名 → 这个作品的记录换成 B，A 的向量当场失效
        self.write_state(diarize_model=str(cfg_b))
        obj, code = self._name_cluster("SPEAKER_01", basis(9), "鲁迪乌斯")
        self.assertEqual(code, 200, obj)
        reg = sr.load_registry(self.reg_root)
        self.assertEqual(reg["embedding_model"], tag_b)
        self.assertFalse(sr._work_vectors_usable(reg["works"][WORK], tag_a),
                         "换模型后 A 的向量仍被判成可用（会静默混比）")

    def test_switching_the_model_within_one_work_keeps_the_spaces_apart(self):
        """Task 7 审查 P1（原 `test_known_gap_*` 的 expectedFailure，本轮修掉）。

        同一作品里先按模型 A 入库、再按模型 B 入库时，`speaker_registry.enroll`
        曾经把**作品级**的 `model` 直接改成 B，于是 A 空间的那批向量（包括同一
        角色旧的 `segNNN`）以 B 的名义复活：拿一个与「艾丽丝」的 A 空间向量完全
        同向的 B 空间质心去问，旧实现会把它算成 1.0 并预填给艾丽丝。

        现在库按**空间**分开存（`pages[P]["space"]` + `…@<空间>` 停放键），
        A 的行不参与 B 空间的匹配，也不参与 70/30 加权。
        """
        self.make_job()
        cfg_a = self._pipeline("a", "wespeaker-voxceleb-resnet34-LM")
        cfg_b = self._pipeline("b", "other-embedder-256")
        self.write_state(diarize_model=str(cfg_a))
        self._name_cluster("SPEAKER_00", mix(0.9), "艾丽丝")
        self.write_state(diarize_model=str(cfg_b))
        self._name_cluster("SPEAKER_01", basis(9), "鲁迪乌斯")

        # 现在库级是 B；拿一个**与艾丽丝的 A 空间向量完全同向**的质心去问：
        # 它不该匹配到艾丽丝（那个向量是 A 空间的，不该参与）。
        self._name_cluster("SPEAKER_00", mix(0.9), "")   # 先清掉名字，再看建议
        labels = sl.load_labels(self.workdir)
        labels["clusters"]["SPEAKER_00"]["name"] = ""
        labels["clusters"]["SPEAKER_00"]["embedding"] = [
            float(x) for x in mix(0.9)]
        sl.save_labels(self.workdir, labels)
        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertIsNone(matches["SPEAKER_00"]["auto"],
                          "A 空间的旧向量被当成了 B 空间，参与了一对一预填")
        self.assertNotEqual(matches["SPEAKER_00"]["page"], "艾丽丝",
                            "A 空间的旧向量参与了余弦：%r" % (matches["SPEAKER_00"],))

    def test_a_and_b_spaces_do_not_blend_into_one_centroid(self):
        """顺序敏感：先命名已有角色 vs 先命名新角色，两条路径都必须干净。

        旧实现的结果取决于用户**先点了谁**：先命名「艾丽丝」→ 质心是干净的
        B 向量（cos 1.0）；先命名「鲁迪乌斯」（`enroll` 把作品级 model 改成 B）
        → 再命名艾丽丝时 `centroid = normalize(0.7*A + 0.3*B)`，与新 B 向量的
        余弦只有 0.3939，本该 auto 的角色静默掉档。同一套用户操作不该因为顺序
        不同而结果不同。
        """
        cfg_a = self._pipeline("a", "wespeaker-voxceleb-resnet34-LM")
        cfg_b = self._pipeline("b", "other-embedder-256")
        v_a = mix(0.9)
        v_b = basis(7)
        for first in ("艾丽丝", "鲁迪乌斯"):
            with self.subTest(first_named=first):
                self._reset_library()
                self.make_job()
                self.write_state(diarize_model=str(cfg_a))
                self._name_cluster("SPEAKER_00", v_a, "艾丽丝")
                # 真流程里换 `diarize_model` ⇒ `_opts_key` 变 ⇒ 重新分割 ⇒
                # 标注状态与 sidecar 一起被清（`run_job` → `_wipe_annotation`）。
                # 不清的话「名字没变 = 不是新确认」那条规则会直接跳过这次入库。
                webui._wipe_annotation(self.workdir)
                self.write_state(diarize_model=str(cfg_b))
                # 换到 B 空间：按 `first` 的顺序命名这两个角色
                plan = {"艾丽丝": ("SPEAKER_00", v_b), "鲁迪乌斯": ("SPEAKER_01", basis(9))}
                for page in ([first] + [p for p in ("艾丽丝", "鲁迪乌斯") if p != first]):
                    cluster, vec = plan[page]
                    obj, code = self._name_cluster(cluster, vec, page)
                    self.assertEqual(code, 200, obj)

                vecs = sr.load_vectors(self.reg_root)
                centroid = vecs[sr.page_centroid_key(WORK, "艾丽丝")]
                self.assertAlmostEqual(float(sr.cosine(centroid, v_b)), 1.0,
                                       places=6,
                                       msg="质心被另一个空间的旧向量污染了：%r"
                                           % (list(np.round(centroid[:3], 4)),))
                # A 空间那两条没被删，只是停放着（换回 A 还能用）
                old_sid = sr.space_id(webui.embedding_model_tag(str(cfg_a)))
                self.assertIn(sr.page_parked_centroid_key(WORK, "艾丽丝", old_sid), vecs)

                # 同一个空间里完全同向的查询必须拿到 auto（不掉档）
                labels = sl.load_labels(self.workdir)
                labels["clusters"]["SPEAKER_00"]["name"] = ""
                labels["clusters"]["SPEAKER_00"]["embedding"] = [
                    float(x) for x in v_b]
                sl.save_labels(self.workdir, labels)
                obj, _ = webui.api_speakers("j1")
                matches = {c["id"]: c["match"] for c in obj["clusters"]}
                self.assertEqual(matches["SPEAKER_00"]["page"], "艾丽丝")
                self.assertEqual(matches["SPEAKER_00"]["band"], "auto",
                                 "本该 auto 的角色掉档了：%r" % (matches["SPEAKER_00"],))
                self.assertEqual(matches["SPEAKER_00"]["auto"], "艾丽丝")

    def test_switching_back_to_the_old_model_makes_its_vectors_usable_again(self):
        """不变量 2（端到端）：换回模型 A 后，面板重新给出 A 空间的建议。

        这条同时钉住 `_cluster_matches` 的闸门：旧实现只比「库级模型 vs 本集
        模型」两个标识，换回 A 时库级还是 B ⇒ 一条建议都不给，A 空间的向量
        等于被判了死刑（而它们明明还在盘上）。
        """
        self.make_job()
        cfg_a = self._pipeline("a", "wespeaker-voxceleb-resnet34-LM")
        cfg_b = self._pipeline("b", "other-embedder-256")
        tag_a = webui.embedding_model_tag(str(cfg_a))
        self.write_state(diarize_model=str(cfg_a))
        self._name_cluster("SPEAKER_00", basis(0), "艾丽丝")
        # B 空间里再入一个别的角色 → 库级模型变成 B，作品级也翻成 B
        webui._wipe_annotation(self.workdir)
        self.write_state(diarize_model=str(cfg_b))
        self._name_cluster("SPEAKER_01", basis(1), "鲁迪乌斯")
        reg = sr.load_registry(self.reg_root)
        self.assertNotEqual(reg["embedding_model"], tag_a)

        # 换回 A：没有任何新的入库，只是一个与艾丽丝 A 空间向量同向的查询
        webui._wipe_annotation(self.workdir)
        self.write_state(diarize_model=str(cfg_a))
        labels = sl.load_labels(self.workdir)
        labels["work_name"] = WORK
        labels["clusters"].setdefault("SPEAKER_00", {"name": ""})
        labels["clusters"]["SPEAKER_00"]["name"] = ""
        labels["clusters"]["SPEAKER_00"]["embedding"] = [
            float(x) for x in basis(0)]
        sl.save_labels(self.workdir, labels)
        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertIsNotNone(matches["SPEAKER_00"],
                             "换回旧模型后 A 空间的向量被闸门判死了")
        self.assertEqual(matches["SPEAKER_00"]["page"], "艾丽丝")
        self.assertEqual(matches["SPEAKER_00"]["score"], 1.0)
        self.assertEqual(matches["SPEAKER_00"]["auto"], "艾丽丝")

    def test_a_legacy_library_with_two_roles_is_fully_adopted_on_the_model_switch(self):
        """Task 7 复审 D1（端到端）：老库多角色 + 换模型，只命名其中一个角色时，
        同一作品里**其它角色**的旧行也必须归属到旧空间。

        改前只有「第一个被命名的角色」拿到归属：另一个角色的 A 空间旧行留在当前
        键位、以 B 的名义参与匹配并被 `/api/speakers` 预填，再命名它时
        `centroid = normalize(0.7*A + 0.3*B)`（`0.9191` 特征），而且没有任何停放行
        —— 换回 A 也拿不回来。
        """
        self.make_job()
        cfg_a = self._pipeline("a", "wespeaker-voxceleb-resnet34-LM")
        cfg_b = self._pipeline("b", "other-embedder-256")
        tag_a = webui.embedding_model_tag(str(cfg_a))
        tag_b = webui.embedding_model_tag(str(cfg_b))
        self.assertNotEqual(tag_a, tag_b)
        # 老库：本提交之前的形态 —— 两个角色都是 A 的向量，JSON/npz 里没有 space
        sr.save_vectors(self.reg_root, {
            sr.page_centroid_key(WORK, "艾丽丝"): basis(0),
            sr.page_segment_key(WORK, "艾丽丝", 0): basis(0),
            sr.page_centroid_key(WORK, "鲁迪乌斯"): basis(5),
            sr.page_segment_key(WORK, "鲁迪乌斯", 0): basis(5)})
        sr.save_registry(self.reg_root, {
            "embedding_model": tag_a, "dim": DIM,
            "works": {WORK: {"model": tag_a, "dim": DIM, "pages": {
                "艾丽丝": {"n_enroll": 1, "total_s": 10.0, "episodes": []},
                "鲁迪乌斯": {"n_enroll": 1, "total_s": 10.0, "episodes": []}}}}})

        # 换模型 B，第 1 集只命名「鲁迪乌斯」
        self.write_state(diarize_model=str(cfg_b))
        obj, code = self._name_cluster("SPEAKER_01", basis(9), "鲁迪乌斯")
        self.assertEqual(code, 200, obj)
        sid_a = sr.space_id(tag_a)
        vecs = sr.load_vectors(self.reg_root)
        self.assertIn(sr.page_parked_centroid_key(WORK, "艾丽丝", sid_a), vecs,
                      "没被命名的角色还留在当前键位（A 的行以 B 的名义复活）")

        # 消费方：拿一个与「艾丽丝」A 空间向量完全同向的质心问建议，不能预填她
        labels = sl.load_labels(self.workdir)
        labels["clusters"]["SPEAKER_00"] = {
            "name": "", "embedding": [float(x) for x in basis(0)]}
        sl.save_labels(self.workdir, labels)
        obj, _ = webui.api_speakers("j1")
        matches = {c["id"]: c["match"] for c in obj["clusters"]}
        self.assertNotEqual(matches["SPEAKER_00"]["page"], "艾丽丝",
                            "A 空间的旧行参与了 B 空间的建议：%r"
                            % (matches["SPEAKER_00"],))
        self.assertIsNone(matches["SPEAKER_00"]["auto"])

        # 第 2 集再命名「艾丽丝」（B）：质心是干净的 B 向量，没被 70/30 掺进 A
        obj, code = self._name_cluster("SPEAKER_00", basis(1), "艾丽丝")
        self.assertEqual(code, 200, obj)
        centroid = sr.load_vectors(self.reg_root)[sr.page_centroid_key(WORK, "艾丽丝")]
        self.assertAlmostEqual(float(sr.cosine(centroid, basis(1))), 1.0, places=6,
                               msg="质心被 A 空间的旧质心污染了：%r"
                                   % (list(np.round(centroid[:3], 4)),))

        # 不变量 2：换回 A（没有任何新的入库），那个「没被第一个命名」的角色照样能用
        back = sr.match_vectors(self.reg_root, work=WORK, clusters={"K": basis(0)},
                                model=tag_a)
        self.assertEqual(back["K"]["top"]["page"], "艾丽丝")
        self.assertEqual(back["K"]["top"]["score"], 1.0)

    def test_state_json_decides_the_model_not_a_global_default(self):
        """同一个 work 目录里的质心是用哪个模型算的，只有 state.json 知道。"""
        self.make_job()
        cfg = self._pipeline("c", "third-embedder")
        self.write_state(diarize_model=str(cfg))
        self.assertEqual(webui._work_embedding_tag(self.workdir),
                         webui.embedding_model_tag(str(cfg)))

    def test_work_without_state_falls_back_to_the_default_resolution(self):
        self.make_job()
        self.assertEqual(webui._work_embedding_tag(self.workdir),
                         webui.embedding_model_tag(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
