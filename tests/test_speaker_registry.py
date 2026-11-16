#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""全局角色库（speaker_registry）—— 纯逻辑 + 两个文件的读写。

规则来源：
- 计划 Task 4 / 设计 §6.2：`speakers_registry.json`（元数据）+ `embeddings.npz`（向量）
- Lead 的实测（推翻设计原锚点）：一对被 pyannote 判为**不同**说话人的组合余弦高达 0.5194，
  所以 auto 阈值必须放到保守侧（默认 0.65 / 0.45）且可配置。

本文件用 tempfile 造库根目录，不碰 output\ 、work\ 里任何既有文件。
"""

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_registry as sr  # noqa: E402


def unit(vec):
    """把任意向量 L2 归一化（测试自己实现，不复用被测代码）。"""
    a = np.asarray(vec, dtype=np.float64)
    return a / float(np.linalg.norm(a))


def vec_at_cos(c, dim=256):
    """返回 dim 维单位向量 e0，且 e0 与 e01 的余弦恰为 c。"""
    e0 = np.zeros(dim, dtype=np.float64)
    e0[0] = 1.0
    e01 = np.zeros(dim, dtype=np.float64)
    e01[0] = c
    e01[1] = float(np.sqrt(max(0.0, 1.0 - c * c)))
    return e0, e01


def cos_of(ca, cb):
    """两个 vec_at_cos 出来的向量之间的余弦 —— 期望值自己算，不写死数字。"""
    return ca * cb + math.sqrt(max(0.0, 1.0 - ca * ca)) * math.sqrt(max(0.0, 1.0 - cb * cb))


def unit_from(comps, dim=256):
    """按 {下标: 分量} 造单位向量（先算范数再归一化）。"""
    v = np.zeros(dim, dtype=np.float64)
    for i, c in comps.items():
        v[int(i)] = float(c)
    return v / float(np.linalg.norm(v))


def vec_from(angle_deg, tilt=0.0, dim=256):
    """e0 绕到 angle_deg 的单位向量；tilt 是往 e2 方向再偏一点（打破 2 维共面）。"""
    a = math.radians(angle_deg)
    return unit_from({0: math.cos(a), 1: math.sin(a), 2: float(tilt)}, dim=dim)


class RegistryCase(unittest.TestCase):
    """每个用例一个临时库根目录（系统临时目录）。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="t4_reg_"))
        self.addCleanup(shutil.rmtree, str(self.root), True)

    def reg_path(self):
        return self.root / sr.REGISTRY_FILE

    def npz_path(self):
        return self.root / sr.EMBEDDINGS_FILE

    def raw_npz_keys(self):
        with np.load(str(self.npz_path())) as z:
            return list(z.keys())

    def raw_npz(self, key):
        with np.load(str(self.npz_path())) as z:
            return z[key]

    def segs_of(self, page, work="W"):
        """该角色盘上的 seg 向量（按 segNNN 序号排序）。"""
        vecs = sr.load_vectors(str(self.root))
        prefix = "page__%s__%s__seg" % (work, page)
        keys = sorted((k for k in vecs if k.startswith(prefix)),
                      key=lambda k: int(k[-3:]))
        return [vecs[k] for k in keys]


class TestCosine(unittest.TestCase):
    def test_cosine_basic(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(sr.cosine(a, b), float(np.sqrt(0.5)), places=6)
        self.assertAlmostEqual(sr.cosine(a, a), 1.0, places=6)
        self.assertAlmostEqual(sr.cosine(a, -a), -1.0, places=6)

        # 未归一化的输入也要按余弦的定义算（不是内积）。两条同向但模长差 100 倍时
        # 仍是 1.0；换成互相成 60 度的两条则是 0.5 —— 内积会算成 2400 这种量级。
        self.assertAlmostEqual(sr.cosine([3.0, 0.0], [300.0, 0.0]), 1.0, places=9)
        self.assertAlmostEqual(sr.cosine([3.0, 0.0], [150.0, 259.8076211353316]),
                               0.5, places=9)

        dim = 256
        q = np.zeros(dim, dtype=np.float64)
        q[0] = 1.0
        stored = np.zeros(dim, dtype=np.float64)
        stored[0] = 250.0
        self.assertAlmostEqual(sr.top_mean(q, [stored], k=3), 1.0, places=9)
        small = np.zeros(dim, dtype=np.float64)
        small[0:2] = [0.6, 0.8]
        big = small * 400.0
        self.assertAlmostEqual(sr.cosine(small, big), 1.0, places=9)

    def test_zero_vector_returns_none(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        zero = np.zeros(2, dtype=np.float32)
        self.assertIsNone(sr.cosine(zero, a))
        self.assertIsNone(sr.cosine(a, zero))
        self.assertIsNone(sr.cosine(zero, zero))

    def test_nan_vector_returns_none(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        bad = np.array([np.nan, 1.0], dtype=np.float32)
        self.assertIsNone(sr.cosine(bad, a))
        self.assertIsNone(sr.cosine(np.array([np.inf, 1.0], dtype=np.float32), a))

    def test_shape_mismatch_returns_none(self):
        a = np.zeros(4, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(256, dtype=np.float32)
        b[0] = 1.0
        self.assertIsNone(sr.cosine(a, b))


class TestScoring(RegistryCase):
    def test_top3_average_beats_single_outlier(self):
        """三条好样本 + 一条坏样本：top-3 平均不被那一条拖下水。"""
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        sims = [0.9, 0.88, 0.86, 0.0]
        stored = []
        for i, s in enumerate(sims):
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1 + i] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            stored.append(v)

        got = sr.top_mean(q, stored, k=3)

        mean_all = float(np.mean(sims))
        expected = float(np.mean(sorted(sims, reverse=True)[:3]))
        self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(got, 0.88, places=6)   # (0.90+0.88+0.86)/3
        self.assertGreater(got, mean_all)             # 明显好于「全平均」（0.66）
        self.assertNotAlmostEqual(got, 0.9, places=3)  # 没退化成单条（否则就是 0.9）
        self.assertLess(got, 0.9)

    def test_top_mean_ignores_zero_and_nan_rows(self):
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        good = np.zeros(256, dtype=np.float32)
        good[0] = 0.5
        good[1] = float(np.sqrt(1.0 - 0.25))
        rows = [good, np.zeros(256, dtype=np.float32),
                np.array([np.nan] * 256, dtype=np.float32)]
        self.assertAlmostEqual(sr.top_mean(q, rows, k=3), 0.5, places=6)

    def test_top_mean_of_nothing_is_none(self):
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        self.assertIsNone(sr.top_mean(q, [], k=3))
        self.assertIsNone(sr.top_mean(q, None, k=3))


class TestStoredVectorsAreNormalised(RegistryCase):
    def test_stored_vectors_are_l2_normalised(self):
        """存进去的每一条向量范数都必须是 1（pyannote 原始输出范数 2.2~2.6）。"""
        raw = np.arange(256, dtype=np.float32) + 1.0     # 范数远大于 1
        self.assertGreater(float(np.linalg.norm(raw)), 2.0)

        sr.enroll(str(self.root), work="无职转生", page="艾丽丝", vector=raw,
                  job="ab12", cluster="SPEAKER_00")
        second = np.zeros(256, dtype=np.float32)
        second[0] = 3.0
        sr.enroll(str(self.root), work="无职转生", page="艾丽丝", vector=second,
                  job="ab12", cluster="SPEAKER_00")

        keys = self.raw_npz_keys()
        self.assertTrue(keys)
        for k in keys:
            v = self.raw_npz(k)
            self.assertEqual(v.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_normalise_vector_helper(self):
        out = sr.normalise_vector([3.0, 4.0])
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)
        self.assertAlmostEqual(float(out[0]), 0.6, places=6)
        self.assertIsNone(sr.normalise_vector([0.0, 0.0]))
        self.assertIsNone(sr.normalise_vector([np.nan, 1.0]))
        self.assertIsNone(sr.normalise_vector(np.array([])))
        self.assertIsNone(sr.normalise_vector(np.zeros((3, 256), dtype=np.float32)))


class TestThresholdBands(RegistryCase):
    def test_threshold_bands(self):
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        spec = {"AUTO": 0.70, "SUGGEST": 0.50, "FAR": 0.30}
        for page, s in spec.items():
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            sr.enroll(str(self.root), work="W", page=page, vector=v,
                      job="j1", cluster="C_" + page)

        out = sr.match_vectors(str(self.root), work="W", clusters={"K1": q})
        got = {c["page"]: (c["band"], c["auto"]) for c in out["K1"]["candidates"]}

        self.assertEqual(got["AUTO"][0], "auto")
        self.assertTrue(got["AUTO"][1])
        self.assertEqual(got["SUGGEST"][0], "suggest")
        self.assertFalse(got["SUGGEST"][1])
        self.assertEqual(got["FAR"][0], "none")
        self.assertFalse(got["FAR"][1])

        self.assertAlmostEqual(out["K1"]["top"]["score"], 0.70, places=4)
        self.assertEqual(out["K1"]["top"]["page"], "AUTO")
        self.assertEqual(out["K1"]["auto_page"], "AUTO")

    def test_match_returns_every_page_sorted_by_score(self):
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        for page, s in (("low", 0.20), ("high", 0.90), ("mid", 0.55)):
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            sr.enroll(str(self.root), work="W", page=page, vector=v)

        out = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        pages = [c["page"] for c in out["K"]["candidates"]]
        self.assertEqual(pages, ["high", "mid", "low"])
        self.assertAlmostEqual(out["K"]["candidates"][0]["score"], 0.90, places=4)

    def test_untouched_page_without_vectors_is_not_a_candidate(self):
        sr.register_page(str(self.root), work="W", page="空角色")
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        self.assertEqual(out["K"]["candidates"], [])
        self.assertIsNone(out["K"]["top"])


class TestOneToOnePairing(RegistryCase):
    def test_one_to_one_greedy_pairing(self):
        """两个聚类争一个角色：只有分数高的那个拿到，另一个拿不到。"""
        sr.enroll(str(self.root), work="W", page="甲", vector=vec_from(0.0))
        sr.enroll(str(self.root), work="W", page="乙", vector=vec_from(60.0))
        q1 = vec_from(5.0)      # 与 甲 cos(5)=0.9962（auto）、与 乙 cos(55)=0.5736（suggest）
        q2 = vec_from(10.0)     # 与 甲 cos(10)=0.9848（auto）、与 乙 cos(50)=0.6428（suggest）

        out = sr.match_vectors(str(self.root), work="W", clusters={"K1": q1, "K2": q2})

        # 两对都指向 甲 且都进 auto（0.9962 / 0.9848）；K1-甲 分更高 → K1 拿走甲，
        # K2 落到 suggest 档的 乙 上（0.6428）
        self.assertEqual(out["K1"]["auto_page"], "甲")
        self.assertIsNone(out["K2"]["auto_page"])      # 甲被占，K2 的 top（甲）拿不到
        self.assertTrue(out["K1"]["top"]["auto"])
        # 候选列表本身不受一对一影响，仍然列全（面板要显示「被 K1 占用」）
        self.assertIn("乙", [c["page"] for c in out["K2"]["candidates"]])
        self.assertEqual(out["K2"]["candidates"][0]["page"], "甲")
        self.assertGreater(out["K1"]["top"]["score"], out["K2"]["top"]["score"])
        # 一对一约束的核心：两个聚类不会同时拿到同一个角色
        self.assertNotEqual(out["K1"]["auto_page"], out["K2"]["auto_page"])

    def test_pairing_is_by_score_not_by_cluster_order(self):
        """分配顺序由分数决定，不由聚类在字典里的先后决定。"""
        sr.enroll(str(self.root), work="W", page="甲", vector=vec_from(0.0))
        q_weak = vec_from(70.0)     # cos(70)=0.342 → suggest
        q_strong = vec_from(5.0)    # cos(5)=0.9962 → auto
        out = sr.match_vectors(str(self.root), work="W",
                               clusters={"ZZ_weak_first": q_weak,
                                         "AA_strong_second": q_strong})
        self.assertEqual(out["AA_strong_second"]["auto_page"], "甲")
        self.assertIsNone(out["ZZ_weak_first"]["auto_page"])
        self.assertFalse(out["ZZ_weak_first"]["top"]["auto"])

    def test_higher_score_wins_even_when_listed_later(self):
        """贪心按分数从高到低配，不受聚类/角色字典顺序影响。"""
        for page, s in (("A", 0.80), ("B", 0.70)):
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            sr.enroll(str(self.root), work="W", page=page, vector=v)
        clusters = {}
        for cid, s in (("ZZ_first_in_dict", 0.99), ("AA_second_in_dict", 0.75)):
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            clusters[cid] = v
        out = sr.match_vectors(str(self.root), work="W", clusters=clusters)
        # 字典顺序里 ZZ 在前，但决定顺序的是分数：最优对 ZZ-A(0.99*) 先配
        self.assertEqual(out["ZZ_first_in_dict"]["auto_page"], "A")
        self.assertEqual(out["AA_second_in_dict"]["auto_page"], "B")
        self.assertEqual(out["ZZ_first_in_dict"]["candidates"][0]["page"], "A")
        # 把贪心次序钉死在纯函数上（不依赖 top-3 平均的具体数值）
        got = sr.greedy_pair({c: i["candidates"] for c, i in out.items()})
        self.assertEqual(got, {"ZZ_first_in_dict": "A", "AA_second_in_dict": "B"})
        # ZZ 的**第一个**候选必须是 A（它最高分那一对就是 A）
        self.assertEqual(out["ZZ_first_in_dict"]["candidates"][0]["page"], "A")

    def test_one_cluster_never_takes_two_pages(self):
        """一个聚类只能配一个角色（反向唯一）。"""
        v = np.zeros(256, dtype=np.float32)
        v[0] = 0.95
        v[1] = float(np.sqrt(1.0 - 0.95 ** 2))
        sr.enroll(str(self.root), work="W", page="A", vector=v)
        sr.enroll(str(self.root), work="W", page="B", vector=v)

        out = sr.match_vectors(str(self.root), work="W", clusters={"K": v})
        self.assertIn(out["K"]["auto_page"], ("A", "B"))
        self.assertEqual(len([c for c in out["K"]["candidates"] if c["auto"]]), 2)
        # top / auto_page 只有一个（score 相同时按角色名稳定取 A）
        self.assertEqual(out["K"]["top"]["page"], "A")
        self.assertEqual(out["K"]["auto_page"], "A")

    def test_greedy_prefers_global_best_not_first_cluster(self):
        """一个角色被两个聚类抢时，落在分数**更高**的那对，而不是先遍历到的聚类。"""
        for page, s in (("A", 0.95), ("B", 0.60)):
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            sr.enroll(str(self.root), work="W", page=page, vector=v)
        clusters = {}
        for cid, s in (("K1", 0.70), ("K2", 0.94)):
            v = np.zeros(256, dtype=np.float32)
            v[0] = s
            v[1] = float(np.sqrt(max(0.0, 1.0 - s * s)))
            clusters[cid] = v
        out = sr.match_vectors(str(self.root), work="W", clusters=clusters)
        self.assertEqual(out["K2"]["auto_page"], "A")     # 0.94 > 0.70，A 归 K2
        self.assertEqual(out["K1"]["auto_page"], "B")
        # 若实现「按聚类顺序配」：K1 先拿 A → K2 只能拿 B。断言把这个钉住。
        self.assertGreater(out["K2"]["top"]["score"], out["K1"]["top"]["score"])


class TestCentroidUpdate(RegistryCase):
    def test_centroid_update_is_70_30(self):
        """center = normalize(0.7*old + 0.3*new)，用可精确表示的 old/new 手算期望。"""
        old = np.zeros(256, dtype=np.float32)
        old[0] = 1.0
        new = np.zeros(256, dtype=np.float32)
        new[1] = 1.0
        expected = unit(0.7 * unit(old) + 0.3 * unit(new))

        sr.enroll(str(self.root), work="W", page="P", vector=old)
        sr.enroll(str(self.root), work="W", page="P", vector=new)

        got = self.raw_npz(sr.page_centroid_key("W", "P"))
        self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=5)
        for i in (0, 1):
            self.assertAlmostEqual(float(got[i]), float(expected[i]), places=5)

    def test_first_enroll_centroid_is_the_vector_itself(self):
        v = np.zeros(256, dtype=np.float32)
        v[5] = 2.0
        sr.enroll(str(self.root), work="W", page="P", vector=v)
        got = self.raw_npz(sr.page_centroid_key("W", "P"))
        self.assertAlmostEqual(float(got[5]), 1.0, places=6)

    def test_segments_are_appended_not_replaced(self):
        for i in range(3):
            v = np.zeros(256, dtype=np.float32)
            v[i] = 1.0
            sr.enroll(str(self.root), work="W", page="P", vector=v, job="j1",
                      cluster="SPEAKER_0%d" % i)
        keys = self.raw_npz_keys()
        segs = sorted(k for k in keys if k.endswith(("seg000", "seg001", "seg002")))
        self.assertEqual(len(segs), 3)
        self.assertTrue(sr.page_segment_key("W", "P", 0) in keys)
        self.assertTrue(sr.page_segment_key("W", "P", 1) in keys)
        self.assertTrue(sr.page_segment_key("W", "P", 2) in keys)
        jobs = [k for k in keys if k.startswith("job__")]
        self.assertEqual(len(jobs), 3)

    def test_thirty_enrolls_do_not_explode_the_npz(self):
        """seg 键循环 0..29，不无限增长。"""
        for i in range(35):
            v = np.zeros(256, dtype=np.float32)
            v[i % 256] = 1.0
            v[255] = 0.001
            sr.enroll(str(self.root), work="W", page="P", vector=v)
        keys = [k for k in self.raw_npz_keys() if k.startswith("page__W__P__seg")]
        self.assertEqual(len(keys), sr.MAX_SEGMENTS)
        self.assertEqual(sr.MAX_SEGMENTS, 30)


class TestDefence(unittest.TestCase):
    def test_nan_and_zero_rows_are_dropped(self):
        good = np.zeros(256, dtype=np.float32)
        good[0] = 1.0
        rows = np.stack([
            np.zeros(256, dtype=np.float32),                      # 全零（pyannote 补位）
            np.full(256, np.nan, dtype=np.float32),               # NaN
            np.full(256, np.inf, dtype=np.float32),               # Inf
            good,                                                 # 好行
        ])
        out = sr.extract_centroids(rows, labels=["A", "B", "C", "D"])
        self.assertEqual(list(out.keys()), ["D"])
        self.assertAlmostEqual(float(np.linalg.norm(out["D"])), 1.0, places=5)

    def test_short_centroid_array_is_zero_padded(self):
        """labels 比质心行数多时，pyannote 会补全零行 —— 补出来的必须丢掉。"""
        good = np.zeros(256, dtype=np.float32)
        good[0] = 1.0
        rows = np.stack([np.zeros(256, dtype=np.float32), good])
        out = sr.extract_centroids(rows, labels=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        self.assertNotIn("SPEAKER_00", out)
        self.assertIn("SPEAKER_01", out)
        self.assertNotIn("SPEAKER_02", out)
        self.assertEqual(len(out), 1)

    def test_none_centroids_returns_empty(self):
        self.assertEqual(sr.extract_centroids(None, labels=["A"]), {})

    def test_silent_input_zero_row_returns_empty(self):
        self.assertEqual(sr.extract_centroids(np.zeros((0, 256)), labels=[]), {})
        self.assertEqual(
            sr.extract_centroids(np.zeros((3, 256), dtype=np.float32),
                                 labels=["A", "B", "C"]), {})

    def test_no_labels_returns_empty(self):
        good = np.zeros(256, dtype=np.float32)
        good[0] = 1.0
        self.assertEqual(sr.extract_centroids(np.stack([good]), labels=None), {})
        self.assertEqual(sr.extract_centroids(np.stack([good]), labels=[]), {})

    def test_wrong_shape_returns_empty(self):
        self.assertEqual(
            sr.extract_centroids(np.zeros((3, 128), dtype=np.float32),
                                 labels=["A", "B", "C"]), {})
        self.assertEqual(sr.extract_centroids([], labels=["A"]), {})

    def test_extract_centroids_l2_normalises(self):
        """实测范数 2.2~2.6：抽出来必须已经归一化。"""
        row = np.arange(256, dtype=np.float32) + 1.0
        out = sr.extract_centroids(np.stack([row]), labels=["SPEAKER_00"])
        self.assertAlmostEqual(float(np.linalg.norm(out["SPEAKER_00"])), 1.0, places=5)

    def test_defence_helpers_never_raise(self):
        weird = [None, 5, "abc", {"a": 1}, np.array(3.0), [1, 2, 3],
                 np.zeros((2, 2, 2), dtype=np.float32),
                 np.array([np.nan, np.nan])]
        for w in weird:
            sr.normalise_vector(w)
            sr.cosine(w, w)
            sr.top_mean(w, [w], k=3)


class TestRegistryIO(RegistryCase):
    def _seed(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="无职转生", page="艾丽丝", vector=v,
                  job="ab12", cluster="SPEAKER_00", seconds=812.4)

    def test_corrupt_registry_returns_empty_and_does_not_raise(self):
        bad = [
            "{ not json",
            "[]",
            '"hello"',
            "123",
            "null",
            json.dumps({"works": "not-a-dict"}),
            json.dumps({"works": {"W": {"pages": ["艾丽丝"]}}}),
            json.dumps({"works": {"W": {"pages": {"P": {"n_enroll": "很多"}}}}, "thresholds": "x"}),
            "",
            "\ufeff",  # 只有 BOM
        ]
        for text in bad:
            self.root.mkdir(parents=True, exist_ok=True)
            self.reg_path().write_text(text, encoding="utf-8")
            reg = sr.load_registry(str(self.root))          # 绝不抛
            self.assertIsInstance(reg, dict)
            self.assertIsInstance(reg["works"], dict)
            self.assertIsInstance(reg["work_hints"], dict)
            self.assertEqual(reg["thresholds"]["auto"], sr.DEFAULT_AUTO)
            self.assertEqual(reg["thresholds"]["suggest"], sr.DEFAULT_SUGGEST)

        # 目录整个不存在也不抛
        missing = self.root / "nope" / "nope"
        self.assertIsInstance(sr.load_registry(str(missing)), dict)
        self.assertIsInstance(sr.load_vectors(str(missing)), dict)

    def test_wrong_typed_thresholds_fall_back_to_defaults(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for bad in ("x", None, [], {"auto": "high"}, {"auto": True},
                    {"suggest": None}, {"auto": 1.5}, {"suggest": -0.2},
                    {"auto": float("nan")}):
            self.reg_path().write_text(
                json.dumps({"version": 1, "thresholds": bad}), encoding="utf-8")
            reg = sr.load_registry(str(self.root))
            self.assertGreater(reg["thresholds"]["auto"], 0.0)
            self.assertGreater(reg["thresholds"]["suggest"], 0.0)
            if isinstance(bad, dict) and isinstance(bad.get("auto"), (int, float)) \
                    and not isinstance(bad.get("auto"), bool) and 0.0 < bad["auto"] < 1.0:
                self.assertEqual(reg["thresholds"]["auto"], bad["auto"])
            else:
                self.assertEqual(reg["thresholds"]["auto"], sr.DEFAULT_AUTO)

    def test_save_is_atomic_and_npz_keys_roundtrip(self):
        self._seed()
        self.assertTrue(self.reg_path().is_file())
        self.assertTrue(self.npz_path().is_file())

        # JSON 里绝不能出现向量（float32 过十进制会掉精度）
        text = self.reg_path().read_text(encoding="utf-8")
        self.assertNotIn("embedding", text.replace("embedding_model", ""))
        self.assertLess(len(text), 4000)

        keys = set(self.raw_npz_keys())
        self.assertIn("page__无职转生__艾丽丝__centroid", keys)
        self.assertIn("page__无职转生__艾丽丝__seg000", keys)
        self.assertIn("job__ab12__SPEAKER_00__centroid", keys)

        # 原子写：必须是 tempfile + os.replace，且不留 .tmp
        with mock.patch.object(sr.os, "replace", wraps=sr.os.replace) as rep:
            self._seed()
        self.assertGreaterEqual(rep.call_count, 1)
        leftovers = [p.name for p in self.root.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

        # 二次加载能读回同一条向量
        v1 = self.raw_npz("page__无职转生__艾丽丝__centroid")
        reg = sr.load_registry(str(self.root))
        vecs = sr.load_vectors(str(self.root))
        self.assertIn("page__无职转生__艾丽丝__centroid", vecs)
        np.testing.assert_allclose(vecs["page__无职转生__艾丽丝__centroid"], v1, rtol=0, atol=0)
        self.assertEqual(reg["works"]["无职转生"]["pages"]["艾丽丝"]["n_enroll"], 2)
        # 两次 seed 的 seconds 累加（每次 812.4）
        self.assertAlmostEqual(
            reg["works"]["无职转生"]["pages"]["艾丽丝"]["total_s"], 1624.8, places=3)
        eps = reg["works"]["无职转生"]["pages"]["艾丽丝"]["episodes"]
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["job"], "ab12")
        self.assertEqual(eps[0]["cluster"], "SPEAKER_00")

    def test_corrupt_npz_is_ignored_without_raising(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.npz_path().write_bytes(b"this is not an npz file at all")
        self.assertEqual(sr.load_vectors(str(self.root)), {})
        # 坏 npz 不该阻止新向量入库（仍是覆盖写）
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v)
        self.assertIn(sr.page_centroid_key("W", "P"), sr.load_vectors(str(self.root)))

    def test_save_vectors_does_not_trust_the_caller(self):
        """`save_vectors` 是公开接口：喂未归一化的向量进去，盘上也必须是单位向量。"""
        raw = np.arange(256, dtype=np.float32) + 1.0
        self.assertGreater(float(np.linalg.norm(raw)), 2.0)
        sr.save_vectors(str(self.root), {sr.page_centroid_key("W", "P"): raw,
                                         sr.page_segment_key("W", "P", 0): raw * 5.0})
        vecs = sr.load_vectors(str(self.root))
        self.assertEqual(sorted(vecs), ["page__W__P__centroid", "page__W__P__seg000"])
        for key, v in vecs.items():
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5,
                                   msg="%s 没被归一化" % key)
        # 全零/NaN 直接不写盘
        sr.save_vectors(str(self.root), {sr.page_centroid_key("W", "P"): np.zeros(256)})
        self.assertEqual(sr.load_vectors(str(self.root)), {})
        self.assertFalse(self.npz_path().is_file())

    def test_load_vectors_drops_corrupt_entries_in_npz(self):
        self.root.mkdir(parents=True, exist_ok=True)
        good = np.zeros(256, dtype=np.float32)
        good[0] = 1.0
        np.savez(str(self.npz_path()),
                 **{"page__W__P__centroid": good,
                    "page__W__P__seg000": np.zeros(256, dtype=np.float32),
                    "page__W__Q__centroid": np.zeros((1, 256), dtype=np.float32),
                    "junk": np.zeros((2, 2), dtype=np.float32)})
        vecs = sr.load_vectors(str(self.root))
        # 全零行（pyannote 的补位）被丢；二维数组与不在键格式里的 junk 也丢
        self.assertEqual(sorted(vecs.keys()), ["page__W__P__centroid"])
        self.assertNotIn("page__W__P__seg000", vecs)     # 零向量 = 坏数据
        self.assertNotIn("page__W__Q__centroid", vecs)   # 二维，不是向量
        self.assertNotIn("junk", vecs)

    def test_work_hint_roundtrip(self):
        media = r"D:\anime\无职转生\第01话.mkv"
        sr.save_registry(str(self.root), {"work_name": "无职转生",
                                          "work_hints": {}})
        sr.set_work_hint(str(self.root), media, "无职转生")
        reg = sr.load_registry(str(self.root))
        self.assertEqual(sr.work_hint(reg, media), "无职转生")
        # 同目录其他集自动带出
        self.assertEqual(sr.work_hint(reg, r"D:\anime\无职转生\第07话.mkv"), "无职转生")
        self.assertIsNone(sr.work_hint(reg, r"D:\anime\别的作品\第01话.mkv"))
        self.assertIsNone(sr.work_hint({}, media))
        self.assertIsNone(sr.work_hint(reg, None))

    def test_registry_and_vectors_paths(self):
        self.assertEqual(sr.registry_path(str(self.root)),
                         Path(self.root) / sr.REGISTRY_FILE)
        self.assertEqual(sr.vectors_path(str(self.root)),
                         Path(self.root) / sr.EMBEDDINGS_FILE)
        self.assertEqual(sr.REGISTRY_FILE, "speakers_registry.json")
        self.assertEqual(sr.EMBEDDINGS_FILE, "embeddings.npz")
        self.assertEqual(sr.vectors_path(self.root), Path(self.root) / "embeddings.npz")

    def test_default_root_is_inside_the_project_and_outside_the_export_root(self):
        self.addCleanup(sr.set_default_root, None)
        base = sr.default_root()
        self.assertTrue(str(base).startswith(str(HERE)), str(base))
        # 库是**状态**，不能放在 `output\说话人\`（导出根）里：
        # 导出会按作品名重建目录，作品名撞上库目录名就会把库删掉。
        self.assertEqual(base, Path(HERE) / sr._STORAGE_DIRNAME)
        self.assertNotIn(str(Path("output") / "说话人"), str(base))

        sr.set_default_root(str(self.root))
        self.assertEqual(sr.default_root(), Path(self.root))
        self.assertEqual(sr.registry_path(), Path(self.root) / sr.REGISTRY_FILE)
        # 默认根改掉后，不传 root 的调用落在新根上
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(work="W", page="P", vector=v)
        self.assertTrue((Path(self.root) / sr.REGISTRY_FILE).is_file())
        self.assertTrue((Path(self.root) / sr.EMBEDDINGS_FILE).is_file())

        sr.set_default_root(None)
        self.assertEqual(sr.default_root(), base)

    def test_save_registry_partial_update_keeps_other_fields(self):
        """只传 thresholds 的一次保存，不能把 works 或 work_hints 抹掉。"""
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v, seconds=3.0)
        sr.set_work_hint(str(self.root), r"D:\anime\W\01.mkv", "W")
        sr.save_registry(str(self.root), {"thresholds": {"auto": 0.7, "suggest": 0.5}})
        reg = sr.load_registry(str(self.root))
        self.assertEqual(reg["thresholds"], {"auto": 0.7, "suggest": 0.5})
        self.assertIn("W", reg["works"])
        self.assertIn("P", reg["works"]["W"]["pages"])
        self.assertAlmostEqual(reg["works"]["W"]["pages"]["P"]["total_s"], 3.0, places=3)
        self.assertEqual(sr.work_hint(reg, r"D:\anime\W\02.mkv"), "W")
        self.assertIn(sr.page_centroid_key("W", "P"), sr.load_vectors(str(self.root)))


class TestThresholdTuning(RegistryCase):
    def test_thresholds_are_configurable_and_default_conservative(self):
        self.assertEqual(sr.DEFAULT_AUTO, 0.65)
        self.assertEqual(sr.DEFAULT_SUGGEST, 0.45)
        self.assertEqual(sr.DEFAULT_THRESHOLDS,
                         {"auto": 0.65, "suggest": 0.45})

        fresh = sr.load_registry(str(self.root))
        self.assertEqual(fresh["thresholds"]["auto"], 0.65)
        self.assertEqual(fresh["thresholds"]["suggest"], 0.45)

        # 库里存的值优先（设计 §6.2 示例里是 0.55 / 0.40）
        sr.save_registry(str(self.root), {"thresholds": {"auto": 0.55, "suggest": 0.40}})
        reg = sr.load_registry(str(self.root))
        self.assertEqual(reg["thresholds"], {"auto": 0.55, "suggest": 0.40})

        # 存进去的 0.60：在默认阈值（0.65）下是 suggest，在库阈值（0.55）下必须是 auto
        v = np.zeros(256, dtype=np.float32)
        v[0] = 0.60
        v[1] = float(np.sqrt(1.0 - 0.36))
        sr.enroll(str(self.root), work="W", page="P", vector=v)
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        self.assertEqual(out["K"]["top"]["band"], "auto")
        self.assertTrue(out["K"]["top"]["auto"])
        self.assertAlmostEqual(out["K"]["top"]["score"], 0.60, places=4)
        self.assertEqual(sr._band_of(0.60, sr.DEFAULT_THRESHOLDS), "suggest")

    def test_threshold_band_boundary_is_inclusive(self):
        """正好等于阈值算进那一档（`>=`，不是 `>`）。边界是半自动预填的决策点。"""
        for auto, suggest in ((0.70, 0.50), (1.0, 0.5), (0.65, 0.45)):
            self.assertEqual(sr._band_of(auto, {"auto": auto, "suggest": suggest}), "auto")
            self.assertEqual(sr._band_of(suggest, {"auto": auto, "suggest": suggest}),
                             "suggest")
            self.assertEqual(sr._band_of(suggest - 0.01,
                                         {"auto": auto, "suggest": suggest}), "none")

        # 走完整链路：分数恰好 1.0（同一个向量、且分量在 float32 下精确可表示，
        # 归一化不会引入 1ulp 误差）而 auto 阈值也是 1.0 —— `>` 会在这里判错。
        sr.save_registry(str(self.root), {"thresholds": {"auto": 1.0, "suggest": 0.45}})
        v = np.zeros(256, dtype=np.float32)
        v[0] = 4.0                      # 4.0 精确可表示，归一化后仍精确是 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v)
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": v})
        self.assertEqual(out["K"]["top"]["score"], 1.0)      # 必须是精确 1.0
        self.assertEqual(out["K"]["top"]["band"], "auto")
        self.assertEqual(out["K"]["auto_page"], "P")

    def test_a_known_different_speaker_pair_is_not_auto_matched(self):
        """Lead 实测：pyannote 判为**不同**说话人的一对，余弦 0.5194。

        0.5194 离设计原锚点 0.55 只差 0.03 —— 这条把它钉成回归：
        在默认阈值（auto=0.65）下必须是 suggest，绝不能自动预填。
        """
        q, stored = vec_at_cos(0.5194)
        self.assertAlmostEqual(sr.cosine(q, stored), 0.5194, places=4)

        sr.enroll(str(self.root), work="W", page="别的角色", vector=stored)
        out = sr.match_vectors(str(self.root), work="W", clusters={"K1": q})
        cand = out["K1"]["candidates"][0]

        self.assertEqual(cand["band"], "suggest")
        self.assertFalse(cand["auto"])
        self.assertIsNone(out["K1"]["auto_page"])          # 关键：没有自动预填
        self.assertGreater(cand["score"], sr.DEFAULT_SUGGEST)
        self.assertLess(cand["score"], sr.DEFAULT_AUTO)

    def test_health_check_reports_same_min_and_diff_max(self):
        # 角色 A：三条向量（e0 / 18 度 / 30 度）
        sr.enroll(str(self.root), work="W", page="A", vector=vec_from(0.0))
        sr.enroll(str(self.root), work="W", page="A", vector=vec_from(18.0))
        sr.enroll(str(self.root), work="W", page="A", vector=vec_from(30.0))
        # 角色 B：两条（10 度 / 45 度 → 自己内部跨 35 度，比 A 的 30 度更松）
        sr.enroll(str(self.root), work="W", page="B", vector=vec_from(10.0))
        sr.enroll(str(self.root), work="W", page="B", vector=vec_from(45.0))
        # 角色 C：一条（跟谁都远，但 A 内部本身就散，所以这组数据仍然重叠）
        sr.enroll(str(self.root), work="W", page="C", vector=vec_from(80.0, tilt=0.2))

        h = sr.health_check(str(self.root))

        self.assertEqual(h["n_pages"], 3)
        self.assertEqual(h["n_segments"], 6)
        self.assertEqual(h["n_pairs_same"], 3 + 1)        # A 三条=3 对，B 两条=1 对
        self.assertEqual(h["n_pairs_diff"], 6 + 3 + 2)    # (A,B)=6 (A,C)=3 (B,C)=2

        # 两个数必须真的等于「同角色两两最低 / 不同角色两两最高」——
        # 不写死数字，直接从盘上的向量自己算一遍来对照
        a_segs = self.segs_of("A")
        b_segs = self.segs_of("B")
        c_segs = self.segs_of("C")
        same = [sr.cosine(a_segs[i], a_segs[j])
                for i in range(len(a_segs)) for j in range(i + 1, len(a_segs))]
        same.append(sr.cosine(b_segs[0], b_segs[1]))
        diff = [sr.cosine(x, y) for x in a_segs for y in b_segs]
        diff.append(sr.cosine(a_segs[0], c_segs[0]))
        self.assertAlmostEqual(h["same_min"], min(same), places=5)
        self.assertAlmostEqual(h["diff_max"], max(diff), places=5)
        # A 内部最松（跨 30 度 = 0.8660），B 内部跨 35 度（0.8192）→ 最低分出自 B
        self.assertLess(sr.cosine(b_segs[0], b_segs[1]), min(same) + 1e-12)
        self.assertEqual(h["same_min_page"], "B")
        self.assertAlmostEqual(h["same_min"], sr.cosine(b_segs[0], b_segs[1]), places=5)
        # 这组数据本来就是重叠的（B 自己内部比 A 更松）
        self.assertLess(h["same_min"], h["diff_max"])
        self.assertFalse(h["ok"])
        self.assertTrue(h["overlap"])
        self.assertIsNone(h["suggested"])                 # 没有间隔就不该给建议

    def test_health_check_numbers_are_correct(self):
        v0 = vec_at_cos(1.0)[0]
        sr.enroll(str(self.root), work="W", page="A", vector=v0)
        sr.enroll(str(self.root), work="W", page="A", vector=vec_at_cos(0.95)[1])
        sr.enroll(str(self.root), work="W", page="B", vector=vec_at_cos(0.60)[1])

        h = sr.health_check(str(self.root))

        # 同角色最低分 = A 的 seg000 与 seg001 的余弦 = 0.95
        # 不同角色最高分 = A 的 seg001 与 B 的 seg000 的余弦（A 的 seg000 是纯 e0，与 B 只有 0.60）
        self.assertAlmostEqual(h["same_min"], 0.95, places=4)
        self.assertAlmostEqual(h["diff_max"], cos_of(0.95, 0.60), places=4)
        self.assertEqual(h["same_min_page"], "A")
        self.assertEqual(h["n_pairs_same"], 1)
        self.assertEqual(h["n_pairs_diff"], 2)
        self.assertIsNotNone(h["gap"])
        self.assertTrue(h["ok"])

    def test_health_check_gap_and_verdict(self):
        """同角色最低分明显高于不同角色最高分 → ok；出现重叠 → 不 ok。"""
        # 情形 1：间隔明显（A 内两条 cos=0.99；B 与 A 都只有 cos(0.99,0.10)）
        v0 = vec_at_cos(1.0)[0]
        sr.enroll(str(self.root), work="W", page="A", vector=v0)
        sr.enroll(str(self.root), work="W", page="A", vector=vec_at_cos(0.99)[1])
        sr.enroll(str(self.root), work="W", page="B", vector=vec_at_cos(0.10)[1])
        h = sr.health_check(str(self.root))
        self.assertAlmostEqual(h["same_min"], 0.99, places=4)
        self.assertAlmostEqual(h["diff_max"], cos_of(0.99, 0.10), places=4)
        self.assertGreater(h["gap"], 0.5)
        self.assertTrue(h["overlap"] is False)
        self.assertTrue(h["ok"])
        self.assertIsNotNone(h["suggested"])

        # 情形 2：重叠。A 内两条 cos(1.0, 45 度)=0.7071，但 B 与 A 的第二条是
        # cos(45 度, 78.5 度)=0.4540、与 A 的第一条 cos(1.0, 78.5 度)=0.20
        # → 同角色最低 0.7071 < 不同角色最高 0.85（B 自己那条）
        root2 = Path(tempfile.mkdtemp(prefix="t4_reg2_"))
        self.addCleanup(shutil.rmtree, str(root2), True)
        sr.enroll(str(root2), work="W", page="A", vector=vec_from(0.0))
        sr.enroll(str(root2), work="W", page="A", vector=vec_from(45.0))
        sr.enroll(str(root2), work="W", page="B", vector=vec_from(78.5))
        h2 = sr.health_check(str(root2))
        a2 = [sr.load_vectors(str(root2))[k] for k in
              sorted(sr.load_vectors(str(root2)), key=lambda k: k[-3:])
              if k.startswith("page__W__A__seg")]
        b2 = [sr.load_vectors(str(root2))[k] for k in
              sorted(sr.load_vectors(str(root2)), key=lambda k: k[-3:])
              if k.startswith("page__W__B__seg")]
        self.assertAlmostEqual(h2["same_min"], sr.cosine(a2[0], a2[1]), places=5)
        self.assertAlmostEqual(h2["diff_max"], max(sr.cosine(x, y)
                                                   for x in a2 for y in b2), places=5)
        self.assertLess(h2["same_min"], h2["diff_max"])
        self.assertLess(h2["gap"], 0.0)
        self.assertTrue(h2["overlap"])
        self.assertFalse(h2["ok"])
        self.assertIsNone(h2["suggested"])

    def test_health_check_does_not_mix_stale_model_vectors(self):
        """换了模型的作品，它的旧 seg 不能进体检（跟新模型的向量不是一个空间）。"""
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W1", page="P", vector=v, model="modelA")
        sr.enroll(str(self.root), work="W2", page="Q", vector=vec_from(20.0),
                  model="modelB")
        h = sr.health_check(str(self.root))
        # W1 的模型已失效 → 只有 W2 的 seg 参与
        self.assertEqual(h["n_segments"], 1)
        self.assertEqual(h["n_pages"], 1)
        self.assertIsNone(h["same_min"])
        self.assertIsNone(h["diff_max"])
        self.assertFalse(h["ok"])

    def test_health_check_on_empty_registry_does_not_raise(self):
        h = sr.health_check(str(self.root))
        self.assertEqual(h["n_pages"], 0)
        self.assertEqual(h["n_segments"], 0)
        self.assertIsNone(h["same_min"])
        self.assertIsNone(h["diff_max"])
        self.assertIsNone(h["gap"])
        self.assertFalse(h["ok"])

    def test_health_check_ignores_centroids_and_job_keys(self):
        """体检只统计 page__*__segNNN（job__ 与 centroid 不参与，否则会自我污染）。"""
        v0 = vec_at_cos(1.0)[0]
        sr.enroll(str(self.root), work="W", page="A", vector=v0, job="j1",
                  cluster="SPEAKER_00")
        sr.enroll(str(self.root), work="W", page="A", vector=vec_at_cos(0.30)[1],
                  job="j1", cluster="SPEAKER_01")
        h = sr.health_check(str(self.root))
        self.assertEqual(h["n_segments"], 2)
        self.assertAlmostEqual(h["same_min"], 0.30, places=4)
        self.assertEqual(h["n_pairs_diff"], 0)

    def test_health_check_can_propose_thresholds(self):
        sr.enroll(str(self.root), work="W", page="A", vector=vec_from(0.0))
        sr.enroll(str(self.root), work="W", page="A", vector=vec_from(20.0))
        sr.enroll(str(self.root), work="W", page="B", vector=vec_from(70.0))
        h = sr.health_check(str(self.root))
        self.assertIsNotNone(h["suggested"])
        self.assertLess(h["suggested"]["auto"], h["same_min"])
        self.assertGreater(h["suggested"]["auto"], h["diff_max"])
        self.assertLessEqual(h["suggested"]["suggest"], h["suggested"]["auto"])
        self.assertTrue(h["ok"])


class TestModelChange(RegistryCase):
    def test_model_change_ignores_old_vectors(self):
        sr.save_registry(str(self.root), {"embedding_model": "modelA", "dim": 256})
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v)

        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        old = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        self.assertEqual(old["K"]["top"]["page"], "P")
        self.assertEqual(old["K"]["top"]["band"], "auto")

        # 换模型后：旧向量键不再参与匹配（角色还在库里，但不是候选）
        reg = sr.load_registry(str(self.root))
        reg["embedding_model"] = "modelB"
        sr.save_registry(str(self.root), reg)
        new = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        self.assertEqual(new["K"]["candidates"], [])
        self.assertIsNone(new["K"]["auto_page"])

        # 旧键仍在盘上（没有静默删除用户数据），只是被忽略
        self.assertIn(sr.page_centroid_key("W", "P"), sr.load_vectors(str(self.root)))
        self.assertFalse(sr.list_registry(str(self.root))[0]["vectors_usable"])

        # 用新模型重新入库后又能匹配。**旧质心不能参与 70/30 加权** ——
        # 用一个方向不同的新向量，这样若实现把旧模型的质心也算进去，
        # 新质心就会偏，这条断言立刻抓到。
        v2 = np.zeros(256, dtype=np.float32)
        v2[1] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v2, model="modelB")
        vecs = sr.load_vectors(str(self.root))
        centroid = vecs[sr.page_centroid_key("W", "P")]
        self.assertAlmostEqual(float(centroid[1]), 1.0, places=6)   # 就是 v2，没混旧的
        self.assertAlmostEqual(float(centroid[0]), 0.0, places=6)
        self.assertIn(sr.page_segment_key("W", "P", 0), vecs)       # 旧 seg000 没被覆盖
        # 旧空间那两条被**归属**到旧空间后停放在那里（Task 7 P1：换模型后旧行不得
        # 以新空间的名义复活），逐位没动 —— 换回 modelA 时它们照样能用
        old_sid = sr.space_id("modelA")
        parked_centroid = sr.page_parked_centroid_key("W", "P", old_sid)
        parked_seg = sr.page_parked_segment_key("W", "P", old_sid, 0)
        self.assertIn(parked_centroid, vecs)
        self.assertIn(parked_seg, vecs)
        np.testing.assert_array_equal(vecs[parked_seg], v)
        # 当前那批键（新空间）只有新的 centroid + 新 seg000；旧空间那条在停放区，
        # 所以「能参与匹配的行」是 2 条 —— 旧空间的不算新空间的证据
        live = [k for k in vecs if k.startswith("page__W__P__seg") and "@" not in k]
        self.assertEqual(len(live), 1)
        row = sr.list_registry(str(self.root))[0]
        self.assertEqual(row["n_vectors"], 2)
        self.assertEqual(row["space"], sr.space_id("modelB"))
        self.assertTrue(row["vectors_usable"])

        out2 = sr.match_vectors(str(self.root), work="W", clusters={"K": v2})
        self.assertEqual(out2["K"]["top"]["page"], "P")
        self.assertEqual(out2["K"]["top"]["band"], "auto")

    def test_same_model_re_enroll_keeps_matching(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v, model="modelA")
        sr.enroll(str(self.root), work="W", page="P", vector=v, model="modelA")
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        self.assertEqual(out["K"]["top"]["page"], "P")
        self.assertEqual(out["K"]["top"]["n_vectors"], 3)   # centroid + seg000 + seg001

    def test_model_change_invalidates_every_work(self):
        """换模型是全局事件：旧的向量跟新质心不是一个空间，所有作品一起失效。

        （库只记一个当前 embedding_model，所以「某个作品单独换模型」没有意义；
        真要换回去，重新入库即可 —— 旧向量一直在盘上，没有被删。）
        """
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W1", page="P", vector=v, model="modelA")
        sr.enroll(str(self.root), work="W2", page="P", vector=v, model="modelB")
        used = {r["work"]: (r["vectors_usable"], r["n_vectors"])
                for r in sr.list_registry(str(self.root))}
        self.assertEqual(used, {"W1": (False, 0), "W2": (True, 2)})

        # W1 用新模型重新入库后立刻可用（旧 seg000 仍在盘上，未被覆盖）
        sr.enroll(str(self.root), work="W1", page="P", vector=v, model="modelB")
        used2 = {r["work"]: (r["vectors_usable"], r["n_vectors"])
                 for r in sr.list_registry(str(self.root))}
        # W1 是 2 而不是 3：旧空间那条 seg000 已经**停放**到旧空间（仍在盘上），
        # 不再算作 B 空间的证据 —— 旧实现把跨空间的向量也计进 n_vectors，
        # 那正是 Task 7 审查 P1 里「质心被 70% 陈旧向量拖走」的同一类错误。
        self.assertEqual(used2, {"W1": (True, 2), "W2": (True, 2)})
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        self.assertIsNotNone(sr.match_vectors(str(self.root), work="W1",
                                              clusters={"K": q})["K"]["top"])

    def test_dim_is_inferred_from_the_vector_when_not_given(self):
        """没传 dim 就自己数 —— 记错维度会让这个作品**所有**向量都匹配不上。"""
        v = np.zeros(64, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v)
        reg = sr.load_registry(str(self.root))
        self.assertEqual(reg["dim"], 64)
        self.assertEqual(reg["works"]["W"]["dim"], 64)
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": v})
        self.assertEqual(out["K"]["top"]["page"], "P")
        self.assertEqual(out["K"]["top"]["band"], "auto")

    def test_enroll_records_model_and_dim(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v,
                  model="some/model", dim=256)
        reg = sr.load_registry(str(self.root))
        self.assertEqual(reg["embedding_model"], "some/model")
        self.assertEqual(reg["dim"], 256)
        self.assertTrue(reg["normalized"])
        self.assertEqual(reg["version"], 1)
        self.assertEqual(reg["works"]["W"]["model"], "some/model")
        self.assertEqual(reg["works"]["W"]["dim"], 256)

    def test_dim_change_ignores_old_vectors(self):
        sr.save_registry(str(self.root), {"embedding_model": "m", "dim": 256})
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v)
        sr.save_registry(str(self.root), {"dim": 512})
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": q})
        self.assertEqual(out["K"]["candidates"], [])


class TestSpaceIsolation(RegistryCase):
    """Task 7 审查 P1：**同一个作品内**换嵌入模型也必须分空间。

    四个不变量（每条一个用例，或一条用例里的独立断言）：
    1. 模型 A 空间入的向量绝不参与「当前模型是 B」那次匹配（余弦与质心加权都不）；
    2. 不丢数据：换回 A 时 A 的行照样能用（不靠「重新入库」才回来）；
    3. 质心只在同一空间内 70/30 混合，`n_enroll`/`n_vectors` 不算跨空间次数；
    4. 库级 `dim` 被污染后仍能自愈（这条在 webui 侧，见 test_webui_registry_api）。
    """

    A = "pyannote/speaker-diarization-3.1/wespeaker-voxceleb-resnet34-LM"
    B = "pyannote/speaker-diarization-3.1/other-embedder-256"

    def _vec(self, index):
        v = np.zeros(256, dtype=np.float32)
        v[index] = 1.0
        return v

    def test_old_space_vectors_never_enter_the_new_space(self):
        """不变量 1：A 空间配对的行不参与 B 空间的匹配。"""
        a = self._vec(0)
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=a, model=self.A)
        sr.enroll(str(self.root), work="W", page="鲁迪乌斯", vector=self._vec(1),
                  model=self.B)

        reg = sr.load_registry(str(self.root))
        # 作品级身份已经翻成 B，但它管不到记过空间的行
        self.assertEqual(reg["works"]["W"]["model"], self.B)
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": a})
        pages = [c["page"] for c in out["K"]["candidates"]]
        self.assertNotIn("艾丽丝", pages,
                         "A 空间的向量参与了 B 空间的匹配：%r" % (out["K"],))
        self.assertIsNone(out["K"]["auto_page"])

    def test_centroid_never_blends_across_spaces(self):
        """不变量 3（前半）：换空间时质心退化成新向量本身，不与旧空间加权。"""
        a = self._vec(0)
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=a, model=self.A)
        sr.enroll(str(self.root), work="W", page="鲁迪乌斯", vector=self._vec(1),
                  model=self.B)
        # 再命名**库里已有**的「艾丽丝」，这次是 B 空间的向量
        b = np.zeros(256, dtype=np.float32)
        b[1] = 1.0
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=b, model=self.B)

        centroid = sr.load_vectors(str(self.root))[sr.page_centroid_key("W", "艾丽丝")]
        np.testing.assert_allclose(centroid, b, rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(sr.cosine(centroid, b)), 1.0, places=6)
        # 与本空间的一个完全同向的查询必须拿到 1.0 的 auto —— 掉档就是被旧空间污染了
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": b})
        self.assertEqual(out["K"]["top"]["page"], "艾丽丝")
        self.assertEqual(out["K"]["top"]["score"], 1.0)
        self.assertEqual(out["K"]["top"]["band"], "auto")

    def test_counts_do_not_mix_spaces(self):
        """不变量 3（后半）：`n_enroll`/`n_vectors` 只算当前空间。"""
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=self._vec(0),
                  model=self.A, seconds=10.0)
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=self._vec(1),
                  model=self.B, seconds=3.0)
        row = [r for r in sr.list_registry(str(self.root)) if r["page"] == "艾丽丝"][0]
        self.assertEqual(row["n_enroll"], 1, "换空间前的入库次数算进了新空间")
        self.assertAlmostEqual(row["total_s"], 3.0, places=3)
        self.assertEqual(row["n_vectors"], 2, "centroid + seg000，旧空间那两条不算")
        self.assertEqual(row["space"], sr.space_id(self.B))
        self.assertEqual(row["space_tag"], self.B)
        # 旧空间那份统计没丢，只是停在停放区
        rec = sr.load_registry(str(self.root))["works"]["W"]["pages"]["艾丽丝"]
        self.assertEqual(rec["parked"][sr.space_id(self.A)]["n_enroll"], 1)
        self.assertAlmostEqual(rec["parked"][sr.space_id(self.A)]["total_s"], 10.0,
                               places=3)

    def test_switching_back_makes_the_old_space_usable_again(self):
        """不变量 2：换回 A 时 A 的行**照样**能用，一条都没被删。"""
        a = self._vec(0)
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=a, model=self.A)
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=self._vec(1),
                  model=self.B)
        vecs_before = dict(sr.load_vectors(str(self.root)))

        # 换回 A（库级模型变了，但没有任何一次新的入库）
        reg = sr.load_registry(str(self.root))
        reg["embedding_model"] = self.A
        sr.save_registry(str(self.root), reg)

        # 用**显式的 model**问：A 的行必须重新参与，且分数是 1.0
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": a},
                               model=self.A)
        self.assertEqual(out["K"]["top"]["page"], "艾丽丝")
        self.assertEqual(out["K"]["top"]["score"], 1.0)
        self.assertEqual(out["K"]["top"]["band"], "auto")
        # B 空间的行反过来不参与（对称）
        out_b = sr.match_vectors(str(self.root), work="W", clusters={"K": self._vec(1)},
                                 model=self.B)
        self.assertEqual(out_b["K"]["top"]["page"], "艾丽丝")
        self.assertEqual(out_b["K"]["top"]["score"], 1.0)
        # 一行都没丢：键集合只多不少
        vecs_after = sr.load_vectors(str(self.root))
        self.assertFalse(set(vecs_before) - set(vecs_after),
                         "换回旧模型时丢了键：%r" % (sorted(set(vecs_before) - set(vecs_after)),))
        # 换回来再入库：A 的旧质心回到当前键位，70/30 与新向量混合（同一个空间）
        a2 = np.zeros(256, dtype=np.float32)
        a2[0] = 0.6
        a2[2] = 0.8
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=a2, model=self.A)
        centroid = sr.load_vectors(str(self.root))[sr.page_centroid_key("W", "艾丽丝")]
        expect = sr.normalise_vector(0.7 * a + 0.3 * a2)
        np.testing.assert_allclose(centroid, expect, rtol=0, atol=1e-6)
        row = [r for r in sr.list_registry(str(self.root)) if r["page"] == "艾丽丝"][0]
        self.assertEqual(row["n_enroll"], 2, "换回 A 后 A 空间那次的入库次数没取回来")

    def test_repeat_switching_never_loses_or_duplicates_rows(self):
        """来回换模型：一行都不丢、一行都不重，seg 序号在各自空间里继续。"""
        sr.enroll(str(self.root), work="W", page="P", vector=self._vec(0), model=self.A)
        for i, tag in enumerate([self.B, self.A, self.B, self.A], start=2):
            sr.enroll(str(self.root), work="W", page="P", vector=self._vec(i), model=tag)
        vecs = sr.load_vectors(str(self.root))
        keys = sorted(k for k in vecs if k.startswith("page__W__P__"))
        self.assertEqual(len(keys), len(set(keys)), "出现重复键：%r" % (keys,))
        # 当前空间（A，最后一次入库是它）= 3 次入库 + 1 条质心；另一个空间（B）
        # 停放着它的 2 次入库 + 1 条质心。一共 5 次入库 → 7 行，一行不多一行不少。
        live = [k for k in keys if "@" not in k]
        parked = [k for k in keys if "@" in k]
        sid_a, sid_b = sr.space_id(self.A), sr.space_id(self.B)
        self.assertNotEqual(sid_a, sid_b)
        self.assertEqual(sorted(k.split("__")[-1] for k in live),
                         ["centroid", "seg000", "seg001", "seg002"])
        self.assertEqual(sorted(k.split("__")[-1] for k in parked),
                         ["centroid@" + sid_b, "seg000@" + sid_b, "seg001@" + sid_b])
        # 两个空间的质心都在，且能被各自的查询命中（没有互相污染）
        out_a = sr.match_vectors(str(self.root), work="W", clusters={"K": self._vec(5)},
                                 model=self.A)
        self.assertEqual(out_a["K"]["top"]["page"], "P")
        self.assertEqual(out_a["K"]["top"]["n_vectors"], 4)
        out_b = sr.match_vectors(str(self.root), work="W", clusters={"K": self._vec(3)},
                                 model=self.B)
        self.assertEqual(out_b["K"]["top"]["page"], "P")
        self.assertEqual(out_b["K"]["top"]["n_vectors"], 3)

    def test_legacy_rows_without_a_space_still_follow_the_work_identity(self):
        """老库（没有 space 字段）：作品身份换了之后旧行不参与，且被归属到旧模型。"""
        a = self._vec(0)
        sr.save_registry(str(self.root), {"embedding_model": self.A, "dim": 256})
        sr.enroll(str(self.root), work="W", page="P", vector=a)   # 没传 model：跟随库
        reg = sr.load_registry(str(self.root))
        self.assertEqual(reg["works"]["W"]["pages"]["P"]["space"], sr.space_id(self.A))
        # 手改库级模型（模拟「老库 + 换了模型」）：旧行当场失效
        reg["embedding_model"] = self.B
        sr.save_registry(str(self.root), reg)
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": a})
        self.assertEqual(out["K"]["candidates"], [])

    def test_every_legacy_page_is_adopted_when_the_work_identity_flips(self):
        """Task 7 复审 D1：老库多角色时，换模型要**整作品一次性**归属到旧空间。

        只归属「换模型后第一个被命名的角色」（改前行为）时，同一作品里其它角色的
        旧行会以新空间的名义复活：参与新空间的匹配、被 `/api/speakers` 预填、
        再命名时被 70/30 掺进新质心 —— 而且**没有任何停放行**，换回旧模型也拿不回
        来（不变量 2 破）。
        """
        a, b_old = self._vec(0), self._vec(5)
        # 老库：两个角色都是 A 空间的向量，JSON/npz 里没有 space 字段
        sr.save_vectors(str(self.root), {
            sr.page_centroid_key("W", "艾丽丝"): a,
            sr.page_segment_key("W", "艾丽丝", 0): a,
            sr.page_centroid_key("W", "鲁迪乌斯"): b_old,
            sr.page_segment_key("W", "鲁迪乌斯", 0): b_old,
        })
        sr.save_registry(str(self.root), {
            "embedding_model": self.A, "dim": 256,
            "works": {"W": {"model": self.A, "dim": 256, "pages": {
                "艾丽丝": {"n_enroll": 1, "total_s": 5.0, "episodes": []},
                "鲁迪乌斯": {"n_enroll": 2, "total_s": 7.0, "episodes": []}}}}})
        # 换到 B，只命名「鲁迪乌斯」—— 作品级身份在这一刻翻新
        sr.enroll(str(self.root), work="W", page="鲁迪乌斯", vector=self._vec(9),
                  model=self.B, seconds=3.0)

        sid_a, sid_b = sr.space_id(self.A), sr.space_id(self.B)
        vecs = sr.load_vectors(str(self.root))
        # 两个角色的老行**都**停放在 A 空间，逐位没动（一个都没被覆盖）
        np.testing.assert_array_equal(
            vecs[sr.page_parked_centroid_key("W", "艾丽丝", sid_a)], a)
        np.testing.assert_array_equal(
            vecs[sr.page_parked_segment_key("W", "艾丽丝", sid_a, 0)], a)
        np.testing.assert_array_equal(
            vecs[sr.page_parked_centroid_key("W", "鲁迪乌斯", sid_a)], b_old)
        pages = sr.load_registry(str(self.root))["works"]["W"]["pages"]
        self.assertEqual(pages["艾丽丝"]["space"], sid_b,
                         "没被命名的角色没归属：旧行还留在当前键位（以新空间的名义复活）")
        self.assertEqual(pages["鲁迪乌斯"]["space"], sid_b)
        # 统计跟着向量走：老库那份停在 A 空间，当前空间从 0 起算
        self.assertEqual(pages["艾丽丝"]["parked"][sid_a]["n_enroll"], 1)
        self.assertAlmostEqual(pages["艾丽丝"]["parked"][sid_a]["total_s"], 5.0, places=3)
        self.assertEqual(pages["艾丽丝"]["n_enroll"], 0)
        self.assertEqual(pages["鲁迪乌斯"]["parked"][sid_a]["n_enroll"], 2)
        self.assertEqual(pages["鲁迪乌斯"]["n_enroll"], 1)
        self.assertAlmostEqual(pages["鲁迪乌斯"]["total_s"], 3.0, places=3)

        # 消费方：还没重新命名的「艾丽丝」不是 B 空间候选，一行都不能算
        out = sr.match_vectors(str(self.root), work="W", clusters={"K": a},
                               model=self.B)
        self.assertNotIn("艾丽丝", [c["page"] for c in out["K"]["candidates"]])
        self.assertNotEqual(out["K"]["auto_page"], "艾丽丝")
        row = [r for r in sr.list_registry(str(self.root)) if r["page"] == "艾丽丝"][0]
        # `n_vectors` 必须是 0：旧行一行都不能算成新空间的证据（改前是 2 且
        # `vectors_usable`，面板据此预填）。归属完之后这个角色的「当前空间」已经是
        # B、只是还没有 B 空间的行 —— 与「登记了还没入库」同一条口径，所以
        # `vectors_usable` 本身仍是 True，这里钉的是那个危险组合。
        self.assertEqual(row["n_vectors"], 0)
        self.assertEqual(row["space"], sid_b)
        self.assertFalse(row["vectors_usable"] and row["n_vectors"] > 0)

        # 再命名「艾丽丝」（B 空间）：质心就是新向量本身，没被 70/30 掺进 A
        sr.enroll(str(self.root), work="W", page="艾丽丝", vector=self._vec(1),
                  model=self.B)
        centroid = sr.load_vectors(str(self.root))[sr.page_centroid_key("W", "艾丽丝")]
        np.testing.assert_allclose(centroid, self._vec(1), rtol=0, atol=1e-6)

        # 不变量 2：换回 A，两个角色的老行**都**还能用，且分数是 1.0
        back_a = sr.match_vectors(str(self.root), work="W", clusters={"K": a},
                                  model=self.A)
        self.assertEqual(back_a["K"]["top"]["page"], "艾丽丝")
        self.assertEqual(back_a["K"]["top"]["score"], 1.0)
        back_b = sr.match_vectors(str(self.root), work="W", clusters={"K": b_old},
                                  model=self.A)
        self.assertEqual(back_b["K"]["top"]["page"], "鲁迪乌斯")
        self.assertEqual(back_b["K"]["top"]["score"], 1.0)

    def test_a_register_only_enroll_still_adopts_the_legacy_rows(self):
        """只登记（没有向量）也会翻新作品身份 —— 归属线索同样会丢，也要归属。

        `enroll(vector=None)` 不动**被登记角色自己**的空间（没有新行就没有可换的
        空间），但作品级身份照旧翻新。这条路径上不归属的话，那批无空间的行就永远
        没有归属线索了：下一次入库时 `old_work_model` 已经等于新模型（D1 的根因）。
        """
        a = self._vec(0)
        sr.save_vectors(str(self.root), {
            sr.page_centroid_key("W", "记过A"): a,
            sr.page_segment_key("W", "记过A", 0): a,
        })
        sr.save_registry(str(self.root), {
            "embedding_model": self.A, "dim": 256,
            "works": {"W": {"model": self.A, "dim": 256, "pages": {
                "记过A": {"n_enroll": 1, "total_s": 4.0, "episodes": []}}}}})

        out = sr.enroll(str(self.root), work="W", page="只登记",
                        vector=None, model=self.B)
        self.assertFalse(out["vector_stored"])
        sid_a, sid_b = sr.space_id(self.A), sr.space_id(self.B)
        vecs = sr.load_vectors(str(self.root))
        np.testing.assert_array_equal(
            vecs[sr.page_parked_centroid_key("W", "记过A", sid_a)], a)
        pages = sr.load_registry(str(self.root))["works"]["W"]["pages"]
        self.assertEqual(pages["记过A"]["space"], sid_b)
        self.assertEqual(pages["记过A"]["parked"][sid_a]["n_enroll"], 1)
        self.assertAlmostEqual(pages["记过A"]["parked"][sid_a]["total_s"], 4.0, places=3)
        # 被登记的那个角色自己：没有新行，就不动它的空间
        self.assertEqual(pages["只登记"]["space"], "")
        self.assertEqual([k for k in vecs if "只登记" in k], [])

    def test_forget_page_deletes_every_space_and_leaves_other_keys_alone(self):
        """Task 7 复审 M7：忘掉角色是不可逆删除 —— 当前键**与所有停放键**一起删。

        行为本身是对的（复审者实测过），但这条分支零覆盖：漏删停放键就是
        「删了还在」（换回旧模型它又回来了）。
        """
        a = self._vec(0)
        sr.enroll(str(self.root), work="W", page="甲", vector=a, model=self.A,
                  seconds=5.0, job="j1", cluster="C1")
        sr.enroll(str(self.root), work="W", page="甲", vector=self._vec(1),
                  model=self.B, seconds=2.0)      # A 那批停放到 @A
        sr.enroll(str(self.root), work="W", page="甲", vector=a, model=self.A,
                  seconds=1.0)                    # B 那批停放到 @B
        sr.enroll(str(self.root), work="W", page="乙", vector=self._vec(3),
                  model=self.A, seconds=4.0)
        sr.enroll(str(self.root), work="别的作品", page="甲", vector=self._vec(4),
                  model=self.A, seconds=6.0)
        before = dict(sr.load_vectors(str(self.root)))

        def belongs_to_target(key):
            ident = sr._key_ident(key)
            return ident is not None and (ident[0], ident[1]) == ("W", "甲")

        target = [k for k in before if belongs_to_target(k)]
        self.assertEqual(len(target), 5, "前提：当前键 3 条 + 停放键 2 条：%r" % (target,))
        self.assertEqual(len([k for k in target if "@" in k]), 2, "前提：有停放键")
        others = {k: v for k, v in before.items() if not belongs_to_target(k)}

        out = sr.forget_page(str(self.root), work="W", page="甲")
        self.assertEqual(out["removed"], 5)
        after = sr.load_vectors(str(self.root))
        self.assertEqual([k for k in after if belongs_to_target(k)], [],
                         "忘掉角色后还有它的键（停放键没删？）：%r" % (sorted(after),))
        self.assertEqual(sorted(after), sorted(others), "删多了或删少了")
        for key, value in others.items():
            np.testing.assert_array_equal(after[key], value, err_msg=key)

    def test_rename_moves_the_parked_keys_of_every_space(self):
        """Task 7 复审 M8：改名要把**停放键**一起搬，`job__*` 审计键不动。

        不搬停放键 = 改个名就把它在旧模型下的历史丢在旧名字下面，换回旧模型
        再也回不来（零覆盖分支）。
        """
        sr.enroll(str(self.root), work="W", page="旧名", vector=self._vec(0),
                  model=self.A)
        sr.enroll(str(self.root), work="W", page="旧名", vector=self._vec(1),
                  model=self.B)                    # A 那批停放到 @A
        sr.enroll(str(self.root), work="W", page="旧名", vector=self._vec(2),
                  model=self.A)                    # B 那批停放到 @B
        sr.enroll(str(self.root), work="W", page="旧名", vector=self._vec(3),
                  model=self.A, job="j9", cluster="SPEAKER_00")
        before = dict(sr.load_vectors(str(self.root)))
        sid_b = sr.space_id(self.B)
        before_rows = {k: v for k, v in before.items() if "旧名" in k}
        job_key = sr.job_centroid_key("j9", "SPEAKER_00")
        self.assertEqual(len(before_rows), 6, "前提：当前键 4 条 + 停放键 2 条")
        self.assertIn(job_key, before)

        out = sr.rename_page(str(self.root), work="W", old="旧名", new="新名")
        self.assertEqual(out["moved"], 6, "停放键没跟着改名")

        after = dict(sr.load_vectors(str(self.root)))
        self.assertEqual([k for k in after if "旧名" in k], [], "旧名字下还留着键")
        self.assertIn(sr.page_centroid_key("W", "新名"), after)
        self.assertIn(sr.page_segment_key("W", "新名", 2), after)
        self.assertIn(sr.page_parked_centroid_key("W", "新名", sid_b), after)
        self.assertIn(sr.page_parked_segment_key("W", "新名", sid_b, 0), after)
        after_rows = {k: v for k, v in after.items() if "新名" in k}
        self.assertEqual(len(after_rows), len(before_rows), "改名丢了行")
        for key, value in list(before_rows.items()):
            self.assertTrue(any(np.array_equal(value, v) for v in after_rows.values()),
                            "改名后这条向量不见了：%r" % (key,))
        # `job__*` 是审计键：改名不该碰它（逐位不变）
        self.assertIn(job_key, after)
        np.testing.assert_array_equal(after[job_key], before[job_key])

    def test_rename_into_another_space_parks_the_stats_instead_of_adding_them(self):
        """Task 7 复审 M12：跨空间并入时统计不能相加（那是面板上的虚高数字）。"""
        sr.enroll(str(self.root), work="W", page="甲", vector=self._vec(0),
                  model=self.A, seconds=6.0)
        sr.enroll(str(self.root), work="W", page="甲", vector=self._vec(1),
                  model=self.A, seconds=4.0)                       # 甲：A 空间 2 次
        sr.enroll(str(self.root), work="W", page="乙", vector=self._vec(2),
                  model=self.B, seconds=2.0)
        sr.enroll(str(self.root), work="W", page="乙", vector=self._vec(3),
                  model=self.B, seconds=1.0)                       # 乙：B 空间 2 次
        sid_a, sid_b = sr.space_id(self.A), sr.space_id(self.B)
        pages_before = sr.load_registry(str(self.root))["works"]["W"]["pages"]
        self.assertEqual(pages_before["甲"]["space"], sid_a)
        self.assertEqual(pages_before["乙"]["space"], sid_b)

        sr.rename_page(str(self.root), work="W", old="甲", new="乙")

        dst = sr.load_registry(str(self.root))["works"]["W"]["pages"]["乙"]
        self.assertEqual(dst["n_enroll"], 2, "A 空间的入库次数算到了 B 空间头上")
        self.assertAlmostEqual(dst["total_s"], 3.0, places=3)
        self.assertEqual(dst["parked"][sid_a]["n_enroll"], 2)
        self.assertAlmostEqual(dst["parked"][sid_a]["total_s"], 10.0, places=3)
        self.assertEqual(len(dst["episodes"]), 4, "episodes 是审计流水：两边都要留")
        # 面板看到的数字也跟着对
        row = [r for r in sr.list_registry(str(self.root)) if r["page"] == "乙"][0]
        self.assertEqual(row["n_enroll"], 2)
        self.assertAlmostEqual(row["total_s"], 3.0, places=3)
        # 甲在 A 空间的行进了停放区（一条都没丢）
        vecs = sr.load_vectors(str(self.root))
        self.assertEqual([k for k in vecs if "__W__甲__" in k], [])
        self.assertEqual(len([k for k in vecs if "__W__乙__" in k and "@" in k]), 3)

    def test_list_registry_marks_a_page_without_rows_in_the_current_space_unusable(self):
        """Task 7 复审 M16：`vectors_usable` 必须按空间算（面板据此标「旧模型」）。

        「记过 A、当前库是 B、这个角色一条 B 空间的行都没有」时正确值是 False；
        去掉空间判据（只留 `work_ok`）会让 ui.html 显示一个假的「可用」。
        """
        sr.enroll(str(self.root), work="W", page="记过A", vector=self._vec(0),
                  model=self.A, seconds=5.0)
        sr.enroll(str(self.root), work="W", page="换到B", vector=self._vec(1),
                  model=self.B, seconds=2.0)            # 作品级身份翻成 B
        sr.register_page(str(self.root), work="W", page="还没入库")
        rows = {r["page"]: r for r in sr.list_registry(str(self.root))}
        sid_a = sr.space_id(self.A)

        stale = rows["记过A"]
        self.assertEqual(stale["n_vectors"], 0)
        self.assertEqual(stale["space"], sid_a)
        self.assertEqual(stale["space_tag"], self.A)
        self.assertFalse(stale["vectors_usable"], "没有当前空间的行却报成可用")
        fresh = rows["换到B"]
        self.assertEqual(fresh["n_vectors"], 2)
        self.assertTrue(fresh["vectors_usable"])
        # 登记了还没入库的角色不该被标成「旧模型」（那同样是假话）
        self.assertEqual(rows["还没入库"]["n_vectors"], 0)
        self.assertEqual(rows["还没入库"]["space"], "")
        self.assertTrue(rows["还没入库"]["vectors_usable"])

    def test_rename_never_silently_drops_a_space_less_source_batch(self):
        """无空间的源行并入记过空间的角色：可以归属不明，但**绝不能静默删行**。

        判据是 `save_vectors` 会丢弃非法键：`src_space` 为空时若按「有目标空间就
        停放」去停，目标键就是 `page__W__P__centroid@`（空空间 id）—— 那两个文件
        里一行都不会剩下（复审 P9-3 实测）。这条只钉「不许丢」，归属口径见 D1/D2。
        """
        v = self._vec(0)
        sr.save_vectors(str(self.root), {
            sr.page_centroid_key("W", "X"): v,
            sr.page_segment_key("W", "X", 0): v,
        })
        sr.save_registry(str(self.root), {
            "embedding_model": self.B, "dim": 256,
            "works": {"W": {"model": self.B, "dim": 256, "pages": {
                "X": {"n_enroll": 1, "total_s": 5.0, "episodes": []},
                "Y": {"n_enroll": 0, "total_s": 0.0, "episodes": [],
                      "space": sr.space_id(self.B)}}}}})
        vecs = sr.load_vectors(str(self.root))
        self.assertEqual(len(vecs), 2, "前提：源行在盘上")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")
        self.assertEqual(out["moved"], 2)
        after = sr.load_vectors(str(self.root))
        self.assertEqual(len(after), 2, "改名把这些行静默删掉了：%r" % (sorted(after),))
        self.assertEqual([k for k in after if "@" in k], [],
                         "空空间 id 的停放键（会被 save_vectors 丢掉）")
        for value in after.values():
            np.testing.assert_array_equal(value, v)


class TestRenameMergeKeepsEverySegRow(RegistryCase):
    r"""Task 7 复审 D3：并入**已存在**的角色时，源的 seg 行按序号**续排**，不按键覆盖。

    「角色名填一个已存在的名字 = 并入该角色」是面板文档化的正常路径（跨集并把本集
    聚类并进历史角色）。而 `rename_page` 原先把源的 `segNNN` 直接映射到目标的**同一
    个序号**，目标已有该键就被覆盖 —— 复审者实测 6 行 → 4 行、5 行 → 3 行。向量是
    用户的声纹数据，覆盖了就再也拿不回来。

    本类钉住三条不变量：
    1. **seg 行数守恒**：合并前后「源的 seg 行 + 目标原有的 seg 行」一条都不许少；
    2. **目标原有的行逐位不变**（`np.testing.assert_array_equal`）；
    3. 源的 seg 行排在**目标该命名空间的最大序号之后**（当前那批键算一个命名空间，
     `@<空间>` 停放键各算一个），到顶就绕回低位找**空位**，绝不覆盖已有行。

    质心行**不在**这三条里：一个角色只有一条质心，合并时按既有语义**覆盖**（不续排）。
    所以下面断「总行数」时写的是 `合并后 = 合并前 - 被并掉的质心条数`，而不是无条件
    相等 —— 无条件相等与「一个角色只有一条质心」互斥。
    """

    A = "pyannote/speaker-diarization-3.1/wespeaker-voxceleb-resnet34-LM"
    B = "pyannote/speaker-diarization-3.1/other-embedder-256"
    C = "pyannote/speaker-diarization-3.1/third-embedder-256"

    def _vec(self, index):
        v = np.zeros(256, dtype=np.float32)
        v[index] = 1.0
        return v

    def rows_of(self, page, work="W"):
        """该角色盘上的全部行（当前那批键 + 所有停放键）→ {键: 向量}。"""
        needle = "__%s__%s__" % (work, page)
        return {k: v for k, v in sr.load_vectors(str(self.root)).items() if needle in k}

    def segs_of_page(self, page, work="W"):
        """该角色的 seg 行（当前 + 停放）→ {键: 向量}。质心不在这里。"""
        return {k: v for k, v in self.rows_of(page, work).items() if "__seg" in k}

    def assert_kept(self, before_rows, after_rows, keys):
        """这些目标原有的行必须还在，且**逐位**与合并前相同。"""
        for key in keys:
            self.assertIn(key, after_rows, "合并把这行弄丢了：%r" % (key,))
            np.testing.assert_array_equal(after_rows[key], before_rows[key], err_msg=key)

    def assert_values_all_present(self, before_segs, after_segs):
        """合并前每一条 seg 向量的**值**都还能在合并后找到（键可以换名，值不能少）。"""
        for key, value in before_segs.items():
            hit = any(np.array_equal(value, v) for v in after_segs.values())
            self.assertTrue(hit, "这条 seg 向量的值不见了：%r" % (key,))

    def test_merge_into_a_target_with_one_seg_row_keeps_every_seg_row(self):
        """目标已有 seg000：源的 3 条 seg 排到 1/2/3，目标的 seg000 一位都不许动。"""
        sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(10), model=self.A)
        for i in (20, 21, 22):
            sr.enroll(str(self.root), work="W", page="X", vector=self._vec(i), model=self.A)
        y_before, x_before = self.rows_of("Y"), self.rows_of("X")
        before_segs = {**self.segs_of_page("Y"), **self.segs_of_page("X")}
        self.assertEqual(len(y_before), 2, "前提：Y = centroid + seg000")
        self.assertEqual(len(x_before), 4, "前提：X = centroid + seg000..002")
        self.assertEqual(len(before_segs), 4, "前提：seg 行共 4 条")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")

        self.assertEqual(out["moved"], 4)
        self.assertEqual(self.rows_of("X"), {}, "源的键没搬干净")
        y_after = self.rows_of("Y")
        self.assert_kept(y_before, y_after, [sr.page_segment_key("W", "Y", 0)])
        self.assertEqual(sorted(self.segs_of_page("Y")),
                         [sr.page_segment_key("W", "Y", i) for i in range(4)],
                         "源的 seg 行没排在目标最大序号之后")
        self.assertEqual(len(self.segs_of_page("Y")), len(before_segs), "seg 行丢了")
        self.assert_values_all_present(before_segs, self.segs_of_page("Y"))
        # 总行数：只剩质心被并掉的那 1 条差额（既有语义）
        self.assertEqual(len(y_after) + 1, len(x_before) + len(y_before), "行数不守恒")

    def test_merge_into_a_target_with_three_seg_rows_keeps_every_seg_row(self):
        """目标已有 seg000..002：源的 2 条排到 3/4（按源序号先后），值一条不少。"""
        for i in (10, 11, 12):
            sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(i), model=self.A)
        for i in (20, 21):
            sr.enroll(str(self.root), work="W", page="X", vector=self._vec(i), model=self.A)
        y_before, x_before = self.rows_of("Y"), self.rows_of("X")
        y_before_segs = self.segs_of_page("Y")
        before_segs = {**y_before_segs, **self.segs_of_page("X")}
        self.assertEqual(len(y_before_segs), 3, "前提：Y 已有 3 条 seg")
        self.assertEqual(len(self.segs_of_page("X")), 2, "前提：X 有 2 条 seg")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")

        self.assertEqual(out["moved"], 3)
        y_after_segs = self.segs_of_page("Y")
        self.assert_kept(y_before_segs, y_after_segs,
                         [sr.page_segment_key("W", "Y", i) for i in range(3)])
        self.assertEqual(len(y_after_segs), 5, "seg 行数不守恒（3 + 2）")
        self.assert_values_all_present(before_segs, y_after_segs)
        # 续排按**源序号**的先后：X 的 seg000(v20) → 3，seg001(v21) → 4
        np.testing.assert_array_equal(y_after_segs[sr.page_segment_key("W", "Y", 3)],
                                      self._vec(20))
        np.testing.assert_array_equal(y_after_segs[sr.page_segment_key("W", "Y", 4)],
                                      self._vec(21))
        self.assertEqual(len(self.rows_of("Y")) + 1, len(x_before) + len(y_before))

    def test_merge_into_a_parked_namespace_keeps_the_targets_parked_rows(self):
        """复审 D3 的最小复现（`p3_rename.py` P3-2）：源的当前批并进目标的 `@A` 停放区。

        Y：A 入一次 → B 入一次（当前批 B 2 行；`@A` 停放 2 行）；X：A 入一次（当前批 2 行）。
        原名同序号覆盖：6 行 → 4 行（X 的 `centroid`/`seg000` 盖掉 Y 的 `@A` 那两条）。
        """
        sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(10), model=self.A)
        sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(11), model=self.B)
        sr.enroll(str(self.root), work="W", page="X", vector=self._vec(20), model=self.A)
        sid_a = sr.space_id(self.A)
        y_before, x_before = self.rows_of("Y"), self.rows_of("X")
        y_before_segs, x_before_segs = self.segs_of_page("Y"), self.segs_of_page("X")
        self.assertEqual(sorted(y_before_segs),
                         [sr.page_segment_key("W", "Y", 0),
                          sr.page_parked_segment_key("W", "Y", sid_a, 0)],
                         "前提：Y 的当前批与 @A 停放区各有一条 seg")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")

        self.assertEqual(out["moved"], 2)
        self.assertEqual(self.rows_of("X"), {}, "源的键没搬干净")
        y_after_segs = self.segs_of_page("Y")
        self.assert_kept(y_before_segs, y_after_segs, sorted(y_before_segs))
        self.assertIn(sr.page_parked_segment_key("W", "Y", sid_a, 1), y_after_segs,
                      "源的 seg 没排在 @A 停放区的最大序号之后")
        self.assertEqual(len(y_after_segs), 3, "seg 行数不守恒（目标 2 + 源 1）")
        self.assert_values_all_present({**y_before_segs, **x_before_segs}, y_after_segs)
        self.assertEqual(len(self.rows_of("Y")) + 1, len(x_before) + len(y_before))

    def test_merge_with_parked_rows_on_both_sides_keeps_the_namespaces_apart(self):
        """双方都有停放行（且源当前批也要进停放区）：两个命名空间各自续排，互不覆盖。

        Y：A → C → B 换两次空间 ⇒ 当前批 B 2 行、`@A` 停放 2 行、`@C` 停放 2 行；
        X：A → C → C → A ⇒ 当前批 A 3 行、`@C` 停放 3 行。
        并入后源当前批进 `@A`（源空间 A）、源的 `@C` 停放行留在 `@C` —— 两处都要
        避开目标已有的序号。
        """
        sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(10), model=self.A)
        sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(11), model=self.C)
        sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(12), model=self.B)
        sr.enroll(str(self.root), work="W", page="X", vector=self._vec(20), model=self.A)
        sr.enroll(str(self.root), work="W", page="X", vector=self._vec(21), model=self.C)
        sr.enroll(str(self.root), work="W", page="X", vector=self._vec(22), model=self.C)
        sr.enroll(str(self.root), work="W", page="X", vector=self._vec(23), model=self.A)
        sid_a, sid_c = sr.space_id(self.A), sr.space_id(self.C)
        y_before, x_before = self.rows_of("Y"), self.rows_of("X")
        y_before_segs = self.segs_of_page("Y")
        before_segs = {**y_before_segs, **self.segs_of_page("X")}
        self.assertEqual(len(y_before_segs), 3, "前提：Y 有 3 条 seg（当前/@A/@C）")
        self.assertEqual(len(self.segs_of_page("X")), 4, "前提：X 有 4 条 seg（当前 2 + @C 2）")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")

        self.assertEqual(out["moved"], 6)
        self.assertEqual(self.rows_of("X"), {}, "源的键没搬干净")
        y_after_segs = self.segs_of_page("Y")
        # 目标原有的 seg 行（当前批 + @A + @C）逐位不变
        self.assert_kept(y_before_segs, y_after_segs, sorted(y_before_segs))
        # 源的当前批 2 条 → @A 的 1/2；源的 @C 停放 2 条 → @C 的 1/2
        for key in (sr.page_parked_segment_key("W", "Y", sid_a, 1),
                    sr.page_parked_segment_key("W", "Y", sid_a, 2),
                    sr.page_parked_segment_key("W", "Y", sid_c, 1),
                    sr.page_parked_segment_key("W", "Y", sid_c, 2)):
            self.assertIn(key, y_after_segs, "源的行没排在目标最大序号之后：%r" % (key,))
        self.assertEqual(len(y_after_segs), 7, "seg 行数不守恒（目标 3 + 源 4）")
        self.assert_values_all_present(before_segs, y_after_segs)
        # Y 有 3 条质心（当前/`@A`/`@C`），源的 2 条质心各自并在 `@A`/`@C` 上 ⇒ 总行数只少 2
        self.assertEqual(len(self.rows_of("Y")) + 2, len(x_before) + len(y_before))

    def test_two_source_rows_in_one_namespace_are_ranked_without_colliding(self):
        """源自己「当前批 + 同空间停放行」共存：两条同序号的行也要各占一个位。

        这种形态是**库文件层面的现实**：旧版本代码的 rename 会把它造出来（复审 P3-3
        的「共存态」），手改过或外部工具重写过的库也可能有。两条源行（`segNNN` 与
        `segNNN@<它自己的空间>`）都会落进同一个命名空间 —— 「边遍历边定序号」在这里
        会让源行互相覆盖，所以序号必须**一起排**。
        """
        sid_a, sid_b = sr.space_id(self.A), sr.space_id(self.B)
        sr.save_vectors(str(self.root), {
            # X：当前批记 A（seg000），且 @A 停放区里还留着一条 seg000（共存态）
            sr.page_centroid_key("W", "X"): self._vec(1),
            sr.page_segment_key("W", "X", 0): self._vec(1),
            sr.page_parked_centroid_key("W", "X", sid_a): self._vec(2),
            sr.page_parked_segment_key("W", "X", sid_a, 0): self._vec(2),
            # Y：当前批记 B（seg000）；@A 停放区已有一条 seg000（目标原有的行）
            sr.page_centroid_key("W", "Y"): self._vec(10),
            sr.page_segment_key("W", "Y", 0): self._vec(10),
            sr.page_parked_segment_key("W", "Y", sid_a, 0): self._vec(11),
        })
        sr.save_registry(str(self.root), {
            "embedding_model": self.B, "dim": 256,
            "works": {"W": {"model": self.B, "dim": 256, "pages": {
                "X": {"n_enroll": 2, "total_s": 2.0, "episodes": [], "space": sid_a},
                "Y": {"n_enroll": 1, "total_s": 1.0, "episodes": [], "space": sid_b}}}}})
        y_before_segs = self.segs_of_page("Y")
        before_segs = {**y_before_segs, **self.segs_of_page("X")}
        before_total = len(self.rows_of("Y")) + len(self.rows_of("X"))
        self.assertEqual(len(before_segs), 4, "前提：X 两条 + Y 两条 seg")
        self.assertEqual(before_total, 7, "前提：X 4 行 + Y 3 行")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")

        self.assertEqual(out["moved"], 4)
        self.assertEqual(self.rows_of("X"), {}, "源的键没搬干净")
        y_after_segs = self.segs_of_page("Y")
        self.assert_kept(y_before_segs, y_after_segs, sorted(y_before_segs))
        # 两条源行都进 @A：排在目标已有的 0 号之后（当前批那条按源键名排在前面）
        np.testing.assert_array_equal(
            y_after_segs[sr.page_parked_segment_key("W", "Y", sid_a, 1)], self._vec(1))
        np.testing.assert_array_equal(
            y_after_segs[sr.page_parked_segment_key("W", "Y", sid_a, 2)], self._vec(2))
        self.assertEqual(len(y_after_segs), 4, "seg 行数不守恒（目标 2 + 源 2）")
        self.assert_values_all_present(before_segs, y_after_segs)
        # 源的两条质心落进同一个停放命名空间（一个角色只有一条质心）⇒ 总行数少 1
        self.assertEqual(len(self.rows_of("Y")) + 1, before_total)

    def test_a_full_segment_namespace_reports_the_reuse_instead_of_lying(self):
        """一个命名空间塞满 `MAX_SEGMENTS` 条之后无位可续：只能环回最老的，但要**如实回报**。

        同时钉住「续排不越过 `MAX_SEGMENTS`」：序号一旦 ≥ 30，下一次 `enroll` 的
        `(existing[-1][0] + 1) % MAX_SEGMENTS` 会算到 1 并把刚并进来的行删掉
        （`seg_idx >= MAX_SEGMENTS` 那步），等于把丢行换个地方发生。
        """
        for i in range(sr.MAX_SEGMENTS):
            sr.enroll(str(self.root), work="W", page="Y", vector=self._vec(i), model=self.A)
        sr.enroll(str(self.root), work="W", page="X", vector=self._vec(200), model=self.A)
        y_before = self.rows_of("Y")
        y_before_segs = self.segs_of_page("Y")
        self.assertEqual(len(y_before_segs), sr.MAX_SEGMENTS, "前提：Y 的 seg 区已满")

        out = sr.rename_page(str(self.root), work="W", old="X", new="Y")

        self.assertEqual(out["overwritten"], 1, "覆盖了却不说")
        self.assertEqual(len(self.rows_of("Y")), len(y_before), "满了也不该改变总行数")
        keys = sorted(self.segs_of_page("Y"))
        self.assertEqual(keys, [sr.page_segment_key("W", "Y", i)
                                for i in range(sr.MAX_SEGMENTS)], "序号越过了 MAX_SEGMENTS")
        # 兜底只该动最老的那条（0 号），其余 29 条逐位不变
        self.assert_kept(y_before_segs, self.segs_of_page("Y"),
                         [sr.page_segment_key("W", "Y", i)
                          for i in range(1, sr.MAX_SEGMENTS)])
        # 源的那条确实落盘了（换掉的是最老的 seg000）
        np.testing.assert_array_equal(self.segs_of_page("Y")[sr.page_segment_key("W", "Y", 0)],
                                      self._vec(200))


class TestLibraryDimSelfHeals(RegistryCase):
    """Task 7 审查 §4：库级 dim 被污染成 3 之后，下一次 256 维入库必须自愈。

    这条路径**不能焊死**：`webui._expected_embedding_dim()` 恒返回 DEFAULT_DIM
    就是靠「照收 256 维 → `enroll` 按向量重量维度」把库救回来。真去读库里的
    `dim`（3）会拒掉 256 维向量 ⇒ dim 永远是 3 ⇒ 库彻底卡死。
    """

    def test_a_256_dim_enroll_repairs_a_polluted_library_dim(self):
        sr.save_registry(str(self.root), {"embedding_model": "m", "dim": 3})
        self.assertEqual(sr.load_registry(str(self.root))["dim"], 3)
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        out = sr.enroll(str(self.root), work="W", page="P", vector=v, model="m")
        self.assertTrue(out["vector_stored"])
        self.assertEqual(sr.load_registry(str(self.root))["dim"], 256)
        got = sr.match_vectors(str(self.root), work="W", clusters={"K": v})
        self.assertEqual(got["K"]["top"]["page"], "P")
        self.assertEqual(got["K"]["top"]["score"], 1.0)


class TestGreedyPairingFunction(unittest.TestCase):
    def test_greedy_pairing_spec(self):
        by_cluster = {
            "K1": [{"page": "A", "score": 0.9, "band": "auto", "auto": True},
                   {"page": "B", "score": 0.7, "band": "auto", "auto": True}],
            "K2": [{"page": "A", "score": 0.8, "band": "auto", "auto": True}],
        }
        got = sr.greedy_pair(by_cluster)
        self.assertEqual(got["K1"], "A")
        self.assertIsNone(got["K2"])       # A 已被 K1 占走

    def test_greedy_pairing_skips_non_auto(self):
        by_cluster = {
            "K1": [{"page": "A", "score": 0.5, "band": "suggest", "auto": False}],
        }
        self.assertIsNone(sr.greedy_pair(by_cluster)["K1"])

    def test_greedy_pairing_is_deterministic_on_ties(self):
        a = [{"page": "A", "score": 0.9, "band": "auto", "auto": True}]
        b = [{"page": "B", "score": 0.9, "band": "auto", "auto": True}]
        got = sr.greedy_pair({"K2": list(b), "K1": list(a)})
        self.assertEqual(got["K1"], "A")
        self.assertEqual(got["K2"], "B")

    def test_greedy_pairing_tolerates_empty(self):
        self.assertEqual(sr.greedy_pair({}), {})
        self.assertEqual(sr.greedy_pair(None), {})
        self.assertEqual(sr.greedy_pair({"K": []}), {"K": None})


class TestWorkScoping(RegistryCase):
    def test_pages_of_other_works_do_not_match(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="作品一", page="同名角色", vector=v)
        q = np.zeros(256, dtype=np.float32)
        q[0] = 1.0
        out = sr.match_vectors(str(self.root), work="作品二", clusters={"K": q})
        self.assertEqual(out["K"]["candidates"], [])

    def test_two_works_can_hold_the_same_page_name(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="作品一", page="同名角色", vector=v)
        sr.enroll(str(self.root), work="作品二", page="同名角色", vector=v)
        reg = sr.load_registry(str(self.root))
        self.assertIn("作品一", reg["works"])
        self.assertIn("作品二", reg["works"])
        self.assertIn(sr.page_centroid_key("作品一", "同名角色"), sr.load_vectors(str(self.root)))
        self.assertIn(sr.page_centroid_key("作品二", "同名角色"), sr.load_vectors(str(self.root)))


class TestRegistryOps(RegistryCase):
    def test_rename_page_moves_all_vectors(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="旧名", vector=v, job="j1",
                  cluster="C1", seconds=10.0)
        sr.enroll(str(self.root), work="W", page="旧名", vector=v, job="j2",
                  cluster="C2", seconds=20.0)

        sr.rename_page(str(self.root), work="W", old="旧名", new="新名")

        vecs = sr.load_vectors(str(self.root))
        self.assertEqual([k for k in vecs if "旧名" in k], [])
        self.assertIn(sr.page_centroid_key("W", "新名"), vecs)
        self.assertIn(sr.page_segment_key("W", "新名", 0), vecs)
        self.assertIn(sr.page_segment_key("W", "新名", 1), vecs)
        reg = sr.load_registry(str(self.root))
        self.assertNotIn("旧名", reg["works"]["W"]["pages"])
        self.assertEqual(reg["works"]["W"]["pages"]["新名"]["n_enroll"], 2)

    def test_forget_page_removes_page_and_vectors(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="A", vector=v)
        sr.enroll(str(self.root), work="W", page="B", vector=v)
        sr.forget_page(str(self.root), work="W", page="A")
        reg = sr.load_registry(str(self.root))
        self.assertNotIn("A", reg["works"]["W"]["pages"])
        self.assertIn("B", reg["works"]["W"]["pages"])
        vecs = sr.load_vectors(str(self.root))
        self.assertEqual([k for k in vecs if "__A__" in k], [])
        self.assertIn(sr.page_centroid_key("W", "B"), vecs)

    def test_registry_ops_on_missing_work_do_not_raise(self):
        sr.rename_page(str(self.root), work="无", old="x", new="y")
        sr.forget_page(str(self.root), work="无", page="x")
        sr.set_work_hint(str(self.root), r"D:\none\x.mkv", "作品")
        self.assertIn("作品", sr.load_registry(str(self.root))["works"])

    def test_list_registry_is_json_friendly(self):
        v = np.zeros(256, dtype=np.float32)
        v[0] = 1.0
        sr.enroll(str(self.root), work="W", page="P", vector=v, job="j1",
                  cluster="C1", seconds=12.5)
        rows = sr.list_registry(str(self.root))
        json.dumps(rows)                      # 必须能直接 json.dumps
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["work"], "W")
        self.assertEqual(rows[0]["page"], "P")
        self.assertEqual(rows[0]["n_enroll"], 1)
        self.assertAlmostEqual(rows[0]["total_s"], 12.5, places=3)
        self.assertEqual(rows[0]["n_vectors"], 2)     # centroid + seg000


class TestPublicApiNames(unittest.TestCase):
    def test_module_exports(self):
        for name in ("DEFAULT_AUTO", "DEFAULT_SUGGEST", "DEFAULT_THRESHOLDS",
                     "REGISTRY_FILE", "EMBEDDINGS_FILE", "MAX_SEGMENTS",
                     "load_registry", "save_registry", "load_vectors", "save_vectors",
                     "registry_path", "vectors_path", "normalise_vector", "cosine",
                     "top_mean", "extract_centroids", "match_vectors", "enroll",
                     "health_check", "greedy_pair", "work_hint", "set_work_hint",
                     "page_centroid_key", "page_segment_key", "job_centroid_key",
                     "rename_page", "forget_page", "list_registry", "register_page",
                     # Task 7 修复：空间（一条向量只属于一个模型空间）
                     "space_id", "page_parked_centroid_key", "page_parked_segment_key",
                     "work_rows_in_space"):
            self.assertTrue(hasattr(sr, name), "缺少接口: " + name)


if __name__ == "__main__":
    unittest.main()
