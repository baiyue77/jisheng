#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""声纹体检小节的**可执行**前端检查（真 ui.html 脚本 + 假 DOM，node 跑）。

为什么不能只写静态子串断言（沿用 `test_ui_registry_behavior.py` 的理由）：
`test_ui_speaker_panel.py` 那类断言看得见「有没有写 `fmtHealthScore(...)`」，
看不见「渲染出来的到底是 0.812 还是 0.000、样本不够时说的是不是人话、
接口 500 时这一节是不是整块消失」。Task 6 已经证明过一次静态断言拦不住 bug
（`if(j.kind==='export')` 改成 `if(false)` 后全量仍绿）。

这里跑的是真实路径：`openSpeakersByWork` → `loadRegistry` → `loadHealth`
→ `renderHealth`，然后断言假 DOM 上观察到的事实。

node 不存在时**只跳过这个文件**（理由写在 skip 里）。
"""

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

HTML = HERE / "ui.html"
PROBE = Path(__file__).resolve().parent / "frontend" / "registry_probe.mjs"
NODE = shutil.which("node")

MARK = "__PROBE_JSON__ "
SKIP_REASON = "缺 node，前端行为未验证"


def run_probe(scenario):
    proc = subprocess.run([NODE, str(PROBE), str(HTML), scenario],
                          capture_output=True, cwd=str(HERE))
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    line = next((ln for ln in out.splitlines() if ln.startswith(MARK)), None)
    if line is None:
        raise AssertionError(
            "探针没有输出结果（node 退出码 %s）\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, out[-2000:], err[-2000:]))
    facts = json.loads(line[len(MARK):])
    if not facts.get("ok"):
        raise AssertionError("探针自己报错：%s" % (facts.get("error")
                                                  or facts.get("fatal"),))
    return facts


def texts(dump):
    """把假 DOM 子树里所有 textContent 收集起来（含嵌套）。"""
    out = [dump.get("text") or ""]
    for child in dump.get("children") or ():
        out.extend(texts(child))
    return out


def find_all(dump, cls):
    out = []
    if dump.get("cls") == cls:
        out.append(dump)
    for child in dump.get("children") or ():
        out.extend(find_all(child, cls))
    return out


def health_text(facts):
    return " ".join(t for t in texts(facts["spkHealthDump"]) if t)


def next_text(facts):
    """「下一步」那一行的文字（假 DOM 的 textContent 不聚合，得递归取）。"""
    return " ".join(t for t in texts(facts.get("nextDump") or {}) if t)


def health_rows(facts):
    """[(行标签, 值文本)] —— 体检区每一行都是 `.regr`（`nm` + `meta`）。"""
    out = []
    for row in find_all(facts["spkHealthDump"], "regr"):
        kids = row.get("children") or []
        if len(kids) >= 2:
            out.append((kids[0].get("text") or "", kids[1].get("text") or ""))
    return out


def row_value(facts, label):
    for name, value in health_rows(facts):
        if label in name:
            return value
    return None


@unittest.skipUnless(NODE, SKIP_REASON)
class TestHealthSectionWithData(unittest.TestCase):
    """有数据：四个数各自成行，外加一句人话结论。"""

    def test_shows_all_four_numbers(self):
        f = run_probe("prefill_auto")
        t = health_text(f)
        self.assertIn("声纹体检", t, "角色库区没有体检小节：%r" % (t,))
        self.assertIn("0.812", row_value(f, "同角色最低分") or "",
                      "同角色最低分没显示")
        self.assertIn("0.501", row_value(f, "不同角色最高分") or "",
                      "不同角色最高分没显示")
        self.assertIn("0.311", row_value(f, "间隔") or "", "间隔没显示")
        self.assertIn("0.701", row_value(f, "建议阈值") or "",
                      "建议阈值没显示（接口给了 suggested 却没用）")

    def test_the_panel_actually_requests_the_health_endpoint(self):
        """恰好请求一次：一次都没请求 = 那一节是摆样子；重复请求 = 白算一遍
        （`health_check` 是 O(角色对 × 向量对) 的纯计算）。"""
        f = run_probe("prefill_auto")
        self.assertEqual([c["url"] for c in f["healthCalls"]],
                         ["/api/registry/health"],
                         "体检接口的请求次数不对：%r" % (f["healthCalls"],))
        self.assertEqual(f["healthCalls"][0]["method"], "GET")

    def test_scores_are_formatted_not_raw_floats(self):
        """0.8123 要按 0.812 显示：把原始浮点直接贴上来，一屏都是无意义的精度。"""
        f = run_probe("prefill_auto")
        t = health_text(f)
        self.assertIn("0.812", t)
        self.assertNotIn("0.8123", t, "分数没格式化，原始浮点直接贴上来了：%r" % (t,))

    def test_verdict_is_a_sentence_not_just_numbers(self):
        f = run_probe("prefill_auto")
        t = health_text(f)
        self.assertIn("可分性良好", t, "没有给人话结论：%r" % (t,))
        # 结论句要带上那个数，否则「良好」是没有依据的形容词
        self.assertRegex(t, r"可分性良好[^。]*0\.311", "结论句里没带间隔：%r" % (t,))
        # 并说清这个结论是怎么来的（哪些角色拉低了 / 哪一对最像）
        self.assertIn(f["rudeus"], t, "没说是哪个角色把同角色最低分拉下来的")
        self.assertIn(f["alice"], t, "没说是哪一对最像")


@unittest.skipUnless(NODE, SKIP_REASON)
class TestHealthSectionWithThinData(unittest.TestCase):
    """样本不够：必须明说，**不许**把 None 渲染成看起来像分数的东西。"""

    def test_thin_data_says_so_instead_of_showing_scores(self):
        f = run_probe("health_thin")
        t = health_text(f)
        self.assertIn("样本还不够", t, "数据不够时没有明说：%r" % (t,))
        self.assertNotIn("可分性良好", t, "样本不够却报「可分性良好」：%r" % (t,))
        self.assertNotIn("阈值可能偏松", t, "样本不够却报「阈值偏松」：%r" % (t,))
        # `(v||0).toFixed(3)` 会让「没有配对」显示成 0.000 —— 那看着像
        # 「算过了，相似度 0」，用户会据此去调阈值。
        self.assertNotIn("0.000", t, "把 null 渲染成了 0.000：%r" % (t,))

    def test_no_score_rows_are_rendered_when_there_is_nothing_to_show(self):
        """样本不够时**不铺「无数据」行**，收成一行说清「不是 0 分」。

        这条原先是「三个分数行都渲染成『无数据』」。行为改成折叠之后，它的**意图**
        （不许把 null 编成数字、不许显得像「算过了」）由更强的形式保证：
        行**根本不存在**，自然无从编。所以改的是断言，不是放宽要求。
        """
        f = run_probe("health_thin")
        for label in ("同角色最低分", "不同角色最高分", "间隔"):
            self.assertIsNone(row_value(f, label),
                              "样本不够时仍然渲染了「%s」行" % (label,))
        t = health_text(f)
        self.assertIn("样本还不够", t, "折叠后没说清为什么没有数：%r" % (t,))
        self.assertIn("不是 0 分", t, "没点明「没算」与「0 分」的区别：%r" % (t,))
        self.assertNotIn("0.000", t, "把 null 渲染成了 0.000：%r" % (t,))
        # 当前阈值任何时候都可用，不该被一起折叠掉
        self.assertIn("0.650", t, "折叠把「当前阈值」也带走了：%r" % (t,))

    def test_the_health_block_stays_short_when_there_is_no_data(self):
        """折叠的实际收益：别把导出区挤出屏幕。行数要明显少于有数据时。"""
        thin = run_probe("health_thin")
        full = run_probe("prefill_auto")
        self.assertLess(len(health_rows(thin)), len(health_rows(full)),
                        "样本不够时行数没减少（折叠没生效）")
        self.assertLessEqual(len(health_rows(thin)), 2,
                             "样本不够时还是铺了太多行：%r" % (health_rows(thin),))

    def test_it_names_what_is_missing(self):
        """说清缺哪一半：只缺跨角色配对时该多标几个角色，而不是多录几集同一个。"""
        f = run_probe("health_thin")
        t = health_text(f)
        self.assertIn("角色", t)
        self.assertRegex(t, r"至少要有?\s*2", "没说要几个角色/几条向量：%r" % (t,))


@unittest.skipUnless(NODE, SKIP_REASON)
class TestHealthSectionWhenTheApiFails(unittest.TestCase):
    """体检接口自己坏了：这一节显示错误，页面别处照常（不是一片空白）。"""

    def test_broken_health_api_shows_the_error(self):
        f = run_probe("health_down")
        t = health_text(f)
        self.assertIn("读不到", t, "接口 500 时体检区没有显示错误：%r" % (t,))
        self.assertIn("体检接口炸了", t, "没把后端给的原因说出来：%r" % (t,))

    def test_a_broken_health_api_is_not_reported_as_thin_data(self):
        """**接口坏了 ≠ 样本不够**：混成一句，用户会跑去多录几集，白费功夫。"""
        f = run_probe("health_down")
        t = health_text(f)
        self.assertNotIn("样本还不够", t, "接口坏了却说「样本不够」：%r" % (t,))
        self.assertNotIn("可分性良好", t)

    def test_the_rest_of_the_panel_still_renders(self):
        f = run_probe("health_down")
        self.assertEqual(len(f["regRows"]), 3,
                         "体检接口坏了把角色库也弄没了：%r" % (f["regRows"],))
        self.assertTrue(any(r["text"][0] == f["alice"] for r in f["regRows"]))
        self.assertIn(f["alice"], f["datalist"], "补全也一起没了")


@unittest.skipUnless(NODE, SKIP_REASON)
class TestHealthSectionBuildsNoMarkup(unittest.TestCase):
    """`same_min_page` / `diff_pair` 都是**角色名** = 用户数据。"""

    def test_hostile_role_names_in_the_health_block_stay_text(self):
        f = run_probe("hostile_names")
        self.assertEqual(f["alertCalls"], [], "面板执行了 alert")
        # 1) 任何一次 innerHTML 赋值里都不许出现这两个名字
        for w in f["innerHtmlWrites"]:
            self.assertNotIn(f["hostile"], w["value"],
                             "innerHTML（%s%s）里出现了角色名：%r"
                             % (w["tag"], ("#" + w["id"]) if w["id"] else "",
                                w["value"][:120]))
            self.assertNotIn(f["hostileWork"], w["value"],
                             "innerHTML（%s%s）里出现了作品名：%r"
                             % (w["tag"], ("#" + w["id"]) if w["id"] else "",
                                w["value"][:120]))
        # 2) 阳性对照：这两个名字**确实**渲染进了体检区，否则上面是空断言
        t = health_text(f)
        self.assertIn(f["hostile"], t,
                      "敌意角色名没渲染到体检区（上面那条断言就是空的）：%r" % (t,))
        self.assertIn(f["hostileWork"], t,
                      "敌意作品名没渲染到体检区：%r" % (t,))


@unittest.skipUnless(NODE, SKIP_REASON)
class TestNextStepHint(unittest.TestCase):
    """标注完「然后呢」：面板顶部要直接说下一步，并给一个跳到导出区的按钮。

    来由：用户标注完角色后不知道要做什么 —— 导出区在屏幕外，往下翻才看得到，
    于是他以为界面坏了。所以既要有字，也要真的能跳过去。
    """

    def test_unnamed_clusters_are_called_out_and_no_jump_button_yet(self):
        """一个都没命名时：说清要去命名，且**不给**跳转按钮
        （那时导出只会被拒绝，给了按钮就是把人送到错误提示前）。"""
        f = run_probe("prefill_auto")
        t = next_text(f)
        self.assertTrue(t.strip(), "面板顶部没有「下一步」提示")
        self.assertIn("命名", t, "提示里没说命名这件事：%r" % (t,))
        self.assertIsNone(f.get("nextButton"),
                          "一个都没命名却给了跳去导出的按钮：%r" % (f.get("nextButton"),))

    def test_all_named_says_to_export_and_the_button_scrolls_there(self):
        f = run_probe("named_kept")          # 该场景里 SPEAKER_00 已命名为 ALICE
        t = next_text(f)
        self.assertIn("都命名好了", t, "全命名完没说下一步：%r" % (t,))
        self.assertIn("导出", t, "全命名完没说去导出：%r" % (t,))
        btn = f.get("nextButton")
        self.assertIsNotNone(btn, "全命名完了却没有跳转按钮")
        self.assertIn("导出", btn.get("label") or "", btn)
        self.assertGreaterEqual(f.get("nextScrolledToExport") or 0, 1,
                                "点了按钮但没滚到导出区（按钮是摆设）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
