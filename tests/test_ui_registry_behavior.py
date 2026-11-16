#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""角色库面板的**可执行**前端检查（真 ui.html 脚本 + 假 DOM，node 跑）。

为什么不能只写静态子串断言：`test_ui_speaker_panel.py` 那类断言看得见
「有没有写 `m.band === 'auto'`」，看不见「它到底有没有执行、输入框里到底是
什么、下拉里有没有那个候选」。Task 6 已经证明过静态断言拦不住 bug
（`if(j.kind==='export')` 改成 `if(false)` 后 617 条全绿）。

这里跑的是真实路径：`openSpeakersByWork` → `loadRegistry` → `renderSpeakers`
→ 用户输入 / 点下拉 / 点改名 / 点删除，然后断言假 DOM 上观察到的事实。
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


def option_values(row):
    return [o["value"] for o in row["selectOptions"]]


@unittest.skipUnless(NODE, SKIP_REASON)
class TestSpeakerSuggestionBehavior(unittest.TestCase):
    """建议式预填：auto 档填进输入框、suggest 档只进候选，**绝不静默替换**。"""

    def test_auto_band_prefills_and_says_it_is_a_suggestion(self):
        f = run_probe("prefill_auto")
        row = f["rows"][0]
        self.assertEqual(row["inputValue"], f["alice"],
                         "≥ auto 阈值（默认 0.65）的候选没有预填进输入框")
        note = row["suggestText"]
        self.assertIn("库建议", note)
        self.assertIn(f["alice"], note)
        self.assertIn("0.912", note, "预填没标相似度：%r" % (note,))
        self.assertIn("确认", note, "看不出「还没写入、要确认」：%r" % (note,))
        # 输入框挂着库中角色名的补全
        self.assertEqual(row["inputListAttr"], "spkNames",
                         "角色名输入框没有接上 <datalist> 补全")

    def test_editing_the_name_drops_the_stale_suggestion_note(self):
        """提示只描述输入框此刻的值：改成别的名字后还说「库建议：X」就是假话。"""
        f = run_probe("prefill_auto")
        self.assertEqual(f["suggestAfterEdit"], "", "改了名字仍显示旧建议")
        self.assertIn("将新建角色", f["mergeAfterEdit"])
        self.assertIn("库建议", f["suggestAfterRestore"],
                      "改回建议的名字后提示没回来")

    def test_suggest_band_is_not_prefilled_but_is_selectable(self):
        """0.512 落在 suggest 档：**不能**出现在输入框里，只能进候选下拉。"""
        f = run_probe("suggest_only")
        row = f["rows"][0]
        self.assertEqual(row["inputValue"], "",
                         "suggest 档被预填进输入框了：%r" % (row["inputValue"],))
        self.assertNotIn(f["rudeus"], row["inputValue"])
        self.assertIn(f["rudeus"], option_values(row),
                      "suggest 档的候选没有出现在下拉里：%r" % (row["selectOptions"],))
        cand = [o for o in row["selectOptions"] if o["value"] == f["rudeus"]][0]
        self.assertIn("0.512", cand["text"], "候选没标相似度：%r" % (cand,))
        self.assertIn("候选", row["suggestText"])
        self.assertNotIn("库建议", row["suggestText"])

    def test_the_candidate_option_is_actually_usable(self):
        """候选下拉不是摆设：选中它要真的把那句话并入该角色。"""
        f = run_probe("suggest_only")
        self.assertTrue(f["candidateOptionFound"], "候选选项没找到")
        self.assertIsNotNone(f["assignCall"], "选中候选后没有提交命名")
        body = json.loads(f["assignCall"]["body"])
        self.assertEqual(body["names"], {"SPEAKER_00": f["rudeus"]})

    def test_the_backend_band_decides_not_a_hardcoded_threshold(self):
        """0.58：计划书里过时的 0.55 会把它判成「可预填」，库里的 band 说它是候选。

        Task 4 实测里有一对被 pyannote 判为**不同**说话人的组合余弦高达 0.5194，
        所以现行默认是 auto=0.65 / suggest=0.45（`speaker_registry.DEFAULT_AUTO`
        / `DEFAULT_SUGGEST`，还写进库可配置）。面板另写一套阈值 = 把那个余量
        又丢掉一次。
        """
        f = run_probe("stale_threshold")
        row = f["rows"][0]
        self.assertEqual(row["inputValue"], "",
                         "0.58 被判成 auto 预填了（前端自己写了 0.55？）")
        self.assertIn(f["rudeus"], option_values(row),
                      "0.58 的候选没进下拉")
        self.assertIn("候选", row["suggestText"])
        self.assertNotIn("库建议", row["suggestText"])

    def test_named_cluster_is_never_replaced(self):
        """已经有名字的聚类：输入框保持用户的名字，不许被建议覆盖。"""
        f = run_probe("named_kept")
        row = f["rows"][0]
        self.assertEqual(row["inputValue"], f["alice"])
        self.assertEqual(row["suggestText"], "", "没有建议却显示了建议行")

    def test_one_to_one_prefills_only_the_first_cluster(self):
        """同一个角色不能同时预填给两个聚类（一对一）。"""
        f = run_probe("one_to_one")
        self.assertEqual(f["rows"][0]["inputValue"], f["alice"],
                         "分高的那个聚类没有预填")
        self.assertEqual(f["rows"][1]["inputValue"], "",
                         "同一个角色被预填给了第二个聚类")
        self.assertIn(f["alice"], option_values(f["rows"][1]),
                      "没抢到的聚类连手动候选都没有")


@unittest.skipUnless(NODE, SKIP_REASON)
class TestNameCompletionAndMergeHint(unittest.TestCase):
    """补全 + 「将并入已有角色」提示：并入是正常路径，但要看得出来。"""

    def test_datalist_holds_the_library_role_names(self):
        f = run_probe("prefill_auto")
        self.assertIn(f["alice"], f["datalist"])
        self.assertIn(f["rudeus"], f["datalist"])

    def test_hint_tells_merge_from_new(self):
        f = run_probe("merge_hint")
        hints = {h["value"]: h["mg"] for h in f["hints"]}
        self.assertIn("将并入已有角色", hints[f["alice"]])
        self.assertIn("2 集", hints[f["alice"]], "并入时没说这个角色有几集历史")
        self.assertIn("将新建角色", hints["新人"])
        self.assertNotIn("并入", hints["新人"])
        # 别的作品的同名角色不是并入 —— 库按 (作品, 角色) 索引
        self.assertIn("将新建角色", hints[f["stranger"]])
        self.assertIn("别的作品", hints[f["stranger"]])

    def test_hint_explains_a_missing_work_name(self):
        f = run_probe("merge_hint")
        self.assertIn("作品名", f["hintNoWork"])
        self.assertIn("不会入库", f["hintNoWork"])
        self.assertIn(f["stranger"], f["datalistNoWork"],
                      "作品名为空时不给全库名字补全")

    def test_datalist_follows_the_work_name(self):
        f = run_probe("merge_hint")
        self.assertIn(f["alice"], f["datalistAfterWorkChange"])
        self.assertNotIn(f["stranger"], f["datalistAfterWorkChange"],
                         "写回作品名后补全里还留着别的作品的角色")


@unittest.skipUnless(NODE, SKIP_REASON)
class TestRegistryPanelActions(unittest.TestCase):
    """角色库区：列出现有角色、改名、删除（删除必须确认）。"""

    def test_registry_section_lists_works_pages_and_counts(self):
        f = run_probe("prefill_auto")
        rows = f["regRows"]
        self.assertEqual(len(rows), 3, "角色库没有按 (作品, 角色) 列出全部角色")
        alice = [r for r in rows if r["text"][0] == f["alice"]][0]
        self.assertIn("2 次入库", alice["text"][1])
        self.assertEqual(alice["buttons"], ["改名", "删除"])
        header = " ".join(t for t in texts(f["spkRegDump"]) if t)
        self.assertIn("角色库", header)
        self.assertIn("3 个角色", header, "库头没给出角色/作品/集数：%r" % (header,))

    def test_stale_model_roles_are_marked(self):
        f = run_probe("prefill_auto")
        rudeus = [r for r in f["regRows"] if r["text"][0] == f["rudeus"]][0]
        self.assertIn("不参与匹配", rudeus["text"][1],
                      "换过模型的角色没标出来：%r" % (rudeus,))

    def test_delete_asks_for_confirmation_and_only_then_posts(self):
        no = run_probe("forget_cancelled")
        self.assertTrue(no["clicked"], "删除按钮没找到")
        self.assertTrue(no["confirmCalls"], "删除没有弹确认框")
        self.assertIn("不可撤销", no["confirmCalls"][0][0])
        self.assertEqual([c for c in no["registryCalls"]
                          if c["method"] == "POST"], [],
                         "用户取消了，仍然发了删除请求")

        yes = run_probe("forget_confirmed")
        posts = [c for c in yes["registryCalls"] if c["method"] == "POST"]
        self.assertEqual(len(posts), 1, "确认后没有（或重复）发删除请求")
        self.assertEqual(json.loads(posts[0]["body"]),
                         {"work": yes["work"], "page": yes["alice"]},
                         "删除提交的 (作品, 角色) 不对")

    def test_rename_posts_the_prompted_name_and_skips_on_cancel(self):
        f = run_probe("rename_prompt")
        self.assertTrue(f["promptCalls"], "改名没有弹输入框")
        posts = [c for c in f["registryCalls"] if c["method"] == "POST"]
        self.assertEqual(len(posts), 1, "改名没有（或重复）提交")
        self.assertEqual(json.loads(posts[0]["body"]),
                         {"work": f["work"], "from": f["alice"], "to": "新名字"})

        # 三种「什么都没输入」都要拦在发请求之前：取消（prompt 返回 null）、
        # 只有空白、以及输入的名字与原名相同。取消走的就是空名那条判据
        # （`(null||'')` 是空串），所以三条一起钉 —— 哪条判据被删掉都会红。
        for scenario in ("rename_cancel", "rename_blank", "rename_unchanged"):
            with self.subTest(scenario=scenario):
                c = run_probe(scenario)
                self.assertTrue(c["promptCalls"], "%s：没弹输入框" % scenario)
                self.assertEqual(
                    [x for x in c["registryCalls"] if x["method"] == "POST"], [],
                    "%s：不该发改名请求" % scenario)


@unittest.skipUnless(NODE, SKIP_REASON)
class TestNoMarkupFromUserData(unittest.TestCase):
    """作品名/角色名都是用户输入：**只能** textContent，拼 innerHTML 就是自 XSS。"""

    def test_hostile_role_names_stay_text(self):
        f = run_probe("hostile_names")
        self.assertEqual(f["alertCalls"], [], "面板执行了 alert")
        # 1) 有没有把用户数据拼进 innerHTML。判据不是「哪个容器被写了」——
        #    真出事的写法往往写在一个没有 id 的 <span> 上 —— 而是
        #    **任何一次 innerHTML 赋值里都不许出现用户数据**。
        for w in f["innerHtmlWrites"]:
            self.assertNotIn(f["hostile"], w["value"],
                             "innerHTML（%s%s）里出现了角色名：%r"
                             % (w["tag"], ("#" + w["id"]) if w["id"] else "",
                                w["value"][:120]))
        for wid in [w for w in f["innerHtmlWrites"]
                    if w["id"] in ("spkList", "spkReg", "spkNames")]:
            self.fail("面板用 innerHTML 写了 %r：%r" % (wid["id"], wid["value"]))
        # 阳性对照：这个负载**确实**带着可执行片段，所以上面那条不是空断言
        self.assertRegex(f["hostile"], r"\son\w+\s*=")
        self.assertIn("<img", f["hostile"])
        # 2) 恶意名字按原文进 textContent（没有被当成标签吃掉）
        self.assertEqual(f["inputValue"], f["hostile"])
        self.assertIn(f["hostile"], f["inputIsSuggestion"])
        self.assertIn(f["hostile"], f["mergeHint"])
        listed = [r["text"][0] for r in f["regRows"]]
        self.assertIn(f["hostile"], listed, "角色库没把角色名按原文渲染")
        self.assertIn(f["hostile"], f["datalist"])
        # 3) 删除提交的是原样的名字，不是被 HTML 解析过的东西
        posts = [c for c in f["registryCalls"] if c["method"] == "POST"]
        self.assertEqual(len(posts), 1, "恶意名字下删除请求异常：%r"
                         % (f["registryCalls"],))
        self.assertEqual(json.loads(posts[0]["body"]),
                         {"work": f["hostileWork"], "page": f["hostile"]})

    def test_hostile_work_names_stay_text(self):
        """**作品名**也是用户数据（Task 7 审查 T2：变异 F10 当时零红）。

        只把角色名做成 payload 时，`renderRegistry` 里 `wh.textContent = w.work …`
        改成 `innerHTML` 一个用例都不会红。作品名会写进库、回显在角色库区，
        所以它和角色名是同一类数据、要同一套判据。
        """
        f = run_probe("hostile_names")
        self.assertEqual(f["alertCalls"], [], "面板执行了 alert")
        hostile_works = (f["hostileWork"], f["hostileWork2"])
        for work in hostile_works:
            self.assertRegex(work, r"\son\w+\s*=|<script", work)
        # 阳性对照：作品名**确实**渲染出来了，否则下面的循环是空断言
        # （本集的作品名进了输入框，两行作品进了角色库区的作品头）
        self.assertEqual(f["spkWorkValue"], f["hostileWork"])
        self.assertTrue(any(f["hostileWork"] in t for t in f["workHeaders"]),
                        "敌意作品名没渲染到角色库区：%r" % (f["workHeaders"],))
        self.assertTrue(any(f["hostileWork2"] in t for t in f["workHeaders"]),
                        "第二个敌意作品名没渲染到角色库区：%r" % (f["workHeaders"],))
        # 任何一次 innerHTML 赋值里都不许出现作品名
        for w in f["innerHtmlWrites"]:
            for work in hostile_works:
                self.assertNotIn(work, w["value"],
                                 "innerHTML（%s%s）里出现了作品名：%r"
                                 % (w["tag"], ("#" + w["id"]) if w["id"] else "",
                                    w["value"][:120]))
        # 作品名也按原文进了角色库区的 textContent
        header_text = " ".join(t for t in texts(f["spkRegDump"]) if t)
        self.assertIn(f["hostileWork"], header_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
