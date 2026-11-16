#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Task 10：声纹体检接口 `GET /api/registry/health`。

`speaker_registry.health_check` 早就写好了（契约见它的 docstring），但一直没人
消费：面板上没有入口，用户看不到「现在这套阈值到底有没有依据」。这里钉的是**接线**
那一段，不是体检算法本身（后者由 tests/test_speaker_registry.py 覆盖）。

三条不变量：
1. **只读**：字节 + `mtime_ns` + 写入口 Mock 零调用，三层判据缺一不可
   （照 `test_webui_registry_api.test_get_registry_never_writes` 的口径。
   Task 7 审查实测过：只比字节会被「同一秒写回同样内容」漏掉）。
2. **空库不撒谎**：`same_min` / `diff_max` / `gap` / `suggested` 是 **None 而不是
   0**，`ok=False`，HTTP 仍然是 200（「阈值没依据」不是请求失败）。
3. **路由真的接上了**：模块级函数全绿也可能压根没挂到 `do_GET` 上，所以有一组
   走真 HTTP 的用例。

库根隔离同 `test_webui_speaker_api.py`：`setUpModule` 指到临时目录，
`tearDownModule` 恢复**进入本模块时**的根 —— 绝不 `set_default_root(None)`
（那是把根指回真实库，Task 5 的 D5 就是踩了这个）。
"""

import json
import math
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_registry as sr  # noqa: E402
import webui                   # noqa: E402

_GUARD_ROOT = None
_GUARD_PREV = None


def setUpModule():
    global _GUARD_ROOT, _GUARD_PREV
    _GUARD_ROOT = tempfile.TemporaryDirectory(prefix="jisheng_health_guard_")
    _GUARD_PREV = sr.default_root()
    sr.set_default_root(_GUARD_ROOT.name)


def tearDownModule():
    global _GUARD_ROOT
    sr.set_default_root(_GUARD_PREV)
    if _GUARD_ROOT is not None:
        _GUARD_ROOT.cleanup()
        _GUARD_ROOT = None


DIM = sr.DEFAULT_DIM
WORK = "无职转生"
ALICE = "艾丽丝"
RUDEUS = "鲁迪乌斯"


def basis(index, dim=DIM):
    """第 index 维是 1 的单位向量：两个不同 index 的余弦**恰好** 0。"""
    v = np.zeros(dim, dtype=np.float64)
    v[index] = 1.0
    return v


def near_basis(index, cos, other=0, dim=DIM):
    """与 `basis(index)` 余弦**恰好**为 cos 的单位向量。"""
    v = np.zeros(dim, dtype=np.float64)
    v[index] = cos
    v[other] = math.sqrt(max(0.0, 1.0 - cos * cos))
    return v


class HealthCase(unittest.TestCase):
    """公共夹具：临时库根（**不预建目录**，空库那条用例要靠它）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="jisheng_health_")
        self.root = Path(self._tmp.name)
        self._reg_prev = sr.default_root()
        self.reg_root = str(self.root / "registry")
        sr.set_default_root(self.reg_root)

    def tearDown(self):
        # 恢复进入本用例前的根，**绝不**写 None（那是指回真实库）
        sr.set_default_root(getattr(self, "_reg_prev", None))
        self._tmp.cleanup()

    def seed(self, page, vector, work=WORK):
        """往库里塞一个角色（走真实 `enroll`，不手写文件）。"""
        return sr.enroll(self.reg_root, work=work, page=page,
                         vector=np.asarray(vector, dtype=np.float64),
                         job="j_" + page, cluster="SPEAKER_00", seconds=10.0)

    def health(self):
        obj, code = webui.api_registry_health()
        self.assertEqual(code, 200, "体检不该返回错误码：%r" % (obj,))
        return obj


# --------------------------------------------------------------------------
# 只读：三层判据
# --------------------------------------------------------------------------

class TestHealthNeverWrites(HealthCase):
    def test_health_check_never_writes(self):
        """三层判据缺一不可（与 `GET /api/registry` 那条同口径）：

        * 字节比对抓**内容**变化；
        * `mtime_ns` 抓「内容一样也重写了一遍」—— 同一秒内用同样的内容
          `write_bytes` 产出的字节完全相同，只比字节抓不到；
        * 写入口 Mock 抓「文件一个字节没动，但确实调了保存函数」。

        体检尤其要钉这一条：`suggested` 就在返回里，顺手把它 `save_registry`
        进库是最自然的「顺手优化」，而那等于用户在没同意的情况下被改掉了匹配
        规则。
        """
        self.seed(ALICE, basis(0))
        self.seed(ALICE, basis(0))
        self.seed(RUDEUS, basis(2))
        reg_path = sr.registry_path(self.reg_root)
        vec_path = sr.vectors_path(self.reg_root)
        before = (reg_path.read_bytes(), vec_path.read_bytes())
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
                obj, code = webui.api_registry_health()
                self.assertEqual(code, 200)
                self.assertIn("suggested", obj)
        finally:
            for m in writers:
                m.stop()

        called = [m for m in mocks if m.called]
        self.assertEqual(called, [], "只读接口调用了写入口：%r"
                         % ([str(m) for m in called],))
        self.assertEqual(reg_path.read_bytes(), before[0], "体检一次就改了元数据")
        self.assertEqual(vec_path.read_bytes(), before[1], "体检一次就改了向量")
        self.assertEqual((reg_path.stat().st_mtime_ns, vec_path.stat().st_mtime_ns),
                         times, "只读接口重写了文件（内容一样也算写了）")

    def test_missing_registry_is_not_created_by_the_health_check(self):
        self.assertFalse(Path(self.reg_root).exists())
        self.health()
        self.assertFalse(Path(self.reg_root).exists(),
                         "体检一次就把空库目录建出来了")


# --------------------------------------------------------------------------
# 空库 / 样本不足：不许把「没算」渲染成 0
# --------------------------------------------------------------------------

class TestHealthOnAnEmptyLibrary(HealthCase):
    def test_empty_library_says_no_data_instead_of_zero(self):
        """空库时四个字段必须是 **None**，不是 0。

        这是接口契约的一部分：面板要照它区分「没算出来」和「算出来是 0」。
        中间层补一个 `or 0.0` 就会让面板显示「同角色最低分 0.000」，看着像
        「算过了，很糟」——彻底的假信息。
        """
        obj = self.health()
        self.assertFalse(obj["ok"], "空库不可能有可分性结论")
        for key in ("same_min", "diff_max", "gap", "suggested"):
            self.assertIsNone(obj[key], "空库时 %s 必须是 None（不是 0）" % key)
        self.assertEqual(obj["n_pages"], 0)
        self.assertEqual(obj["n_segments"], 0)
        self.assertEqual(obj["n_pairs_same"], 0)
        self.assertEqual(obj["n_pairs_diff"], 0)

    def test_a_single_role_has_nothing_to_compare_against(self):
        """一个角色的库：同角色配对算得出来，跨角色一个都没有 → 依然没有结论。"""
        for vec in (basis(0), basis(1), basis(2)):
            self.seed(ALICE, vec)
        obj = self.health()
        self.assertEqual(obj["n_pages"], 1)
        self.assertGreater(obj["n_pairs_same"], 0, "同角色配对应当算出来了")
        self.assertEqual(obj["n_pairs_diff"], 0)
        self.assertIsNone(obj["diff_max"])
        self.assertIsNone(obj["gap"], "只有一半数据时 gap 必须是 None，不能当成 0")
        self.assertIsNone(obj["suggested"])
        self.assertFalse(obj["ok"])


# --------------------------------------------------------------------------
# 数值：可分 / 不可分
# --------------------------------------------------------------------------

class TestHealthNumbers(HealthCase):
    def test_separable_library_reports_a_positive_gap_and_a_suggestion(self):
        # 每个角色两条一模一样的向量 → 同角色最低分 1.0；两个角色正交 → 跨角色 0.0
        self.seed(ALICE, basis(0))
        self.seed(ALICE, basis(0))
        self.seed(RUDEUS, basis(2))
        self.seed(RUDEUS, basis(2))

        obj = self.health()
        self.assertTrue(obj["ok"], "完全可分却说没有依据：%r" % (obj,))
        self.assertAlmostEqual(obj["same_min"], 1.0, places=6)
        self.assertAlmostEqual(obj["diff_max"], 0.0, places=6)
        self.assertAlmostEqual(obj["gap"], 1.0, places=6)
        self.assertFalse(obj["overlap"])
        self.assertEqual(obj["same_min_page"], ALICE, "没报出最低分是哪个角色")
        self.assertEqual(obj["diff_pair"], [ALICE, RUDEUS], "没报出最像的那一对")

        sg = obj["suggested"]
        self.assertIsNotNone(sg, "有正间隔却没给建议阈值")
        # 建议值必须落在 (diff_max, same_min) 这个间隔**里面**：落到外面就是
        # 「把不同的人认成同一个」或「把同一个人拆开」的阈值。
        self.assertGreater(sg["auto"], obj["diff_max"])
        self.assertLess(sg["auto"], obj["same_min"])
        self.assertLess(sg["suggest"], sg["auto"], "保守档必须比 auto 更低")

    def test_overlapping_library_is_not_ok_and_gives_no_suggestion(self):
        # 艾丽丝自身两条只有 0.8；而 basis(0) 同时也是鲁迪乌斯的一条 → 跨角色 1.0
        self.seed(ALICE, basis(0))
        self.seed(ALICE, near_basis(0, 0.8, other=1))
        self.seed(RUDEUS, basis(0))
        self.seed(RUDEUS, basis(0))

        obj = self.health()
        self.assertFalse(obj["ok"], "跨角色最高分反超同角色最低分，居然还有依据")
        self.assertAlmostEqual(obj["same_min"], 0.8, places=6)
        self.assertAlmostEqual(obj["diff_max"], 1.0, places=6)
        self.assertLess(obj["gap"], 0.0)
        self.assertTrue(obj["overlap"], "重叠没有被标出来")
        self.assertIsNone(obj["suggested"], "没有间隔时不该给建议阈值")


# --------------------------------------------------------------------------
# HTTP 路由（模块级函数全绿也可能根本没接上线）
# --------------------------------------------------------------------------

class TestHealthHttpRoute(HealthCase):
    def setUp(self):
        super().setUp()
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        daemon=True)
        self._thread.start()
        self.base = "http://127.0.0.1:%d" % self._srv.server_address[1]

    def tearDown(self):
        try:
            self._srv.shutdown()
            self._srv.server_close()
        finally:
            super().tearDown()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_route_is_wired_and_the_empty_library_does_not_crash(self):
        status, obj = self._get("/api/registry/health")
        self.assertEqual(status, 200)
        self.assertIn("ok", obj)
        self.assertFalse(obj["ok"])
        self.assertIsNone(obj["same_min"])
        self.assertIsNone(obj["diff_max"])
        self.assertIsNone(obj["gap"])
        self.assertIsNone(obj["suggested"])
        # 面板要靠它说人话，所以这两样必须在
        self.assertIn("thresholds", obj)
        self.assertIn("n_segments", obj)

    def test_route_reports_the_gap_once_there_is_data(self):
        self.seed(ALICE, basis(0))
        self.seed(ALICE, basis(0))
        self.seed(RUDEUS, basis(2))
        self.seed(RUDEUS, basis(2))
        status, obj = self._get("/api/registry/health")
        self.assertEqual(status, 200)
        self.assertTrue(obj["ok"])
        self.assertAlmostEqual(obj["gap"], 1.0, places=6)

    def test_health_route_does_not_shadow_the_registry_listing(self):
        """两个路径只差一段后缀：接错线时 `/api/registry/health` 会返回库列表
        （或者反过来，库列表被体检抢走）。这里把两边的形状都钉住。"""
        self.seed(ALICE, basis(0))
        status, listing = self._get("/api/registry")
        self.assertEqual(status, 200)
        self.assertIn("works", listing)
        self.assertNotIn("same_min", listing)
        status, health = self._get("/api/registry/health")
        self.assertEqual(status, 200)
        self.assertIn("same_min", health)
        self.assertNotIn("works", health)


if __name__ == "__main__":
    unittest.main(verbosity=2)
