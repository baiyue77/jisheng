#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""导出面板的**可执行**前端检查（真 ui.html 脚本 + 假 DOM，node 跑）。

为什么单独有这个文件：`test_ui_speaker_panel.py` 里的断言全是**静态子串**检查。
审查者做过变异性实验：把 `updateJob` 里的
`if(j.kind==='export') renderExport(j);` 改成 `if(false) renderExport(j);`，
617 条测试**全绿** —— 于是「导出完成后面板什么都不显示」这种事故一条都拦不住。
静态断言看得见「有没有写 renderExport(j)」，看不见「它到底有没有被执行、
els[id] 有没有建出来、结果区有没有亮」。

这里改成真跑一遍：`tests/frontend/export_probe.mjs` 把 ui.html 里那段真脚本
抽出来，在极简假 DOM 上按真实路径走
`startExport()` -> `kickPoll()` 轮询到 done -> `updateJob()` -> `renderExport()`，
把观察到的事实以 JSON 吐回来，Python 侧断言。所以它拦得住
「addJob 早退导致卡片不建 / 结果区不亮 / warning 无条件显示 / 面板提交字段被改名」
这一类只有跑起来才暴露的 bug。

node 不存在时**只跳过这个文件**（理由写在 skip 里），既有静态断言不受影响。
"""

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

HTML = HERE / "ui.html"
PROBE = Path(__file__).resolve().parent / "frontend" / "export_probe.mjs"
NODE = shutil.which("node")

#: 探针把事实打在这样一行里（ui.html 的脚本自己不打印，混不进别的东西）
MARK = "__PROBE_JSON__ "

SKIP_REASON = "缺 node，前端行为未验证"


def run_probe(scenario):
    """跑一个场景，返回它观察到的事实 dict。"""
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


@unittest.skipUnless(NODE, SKIP_REASON)
class TestExportPanelBehavior(unittest.TestCase):
    """走真脚本的行为断言。每条都对应一个「改坏了静态断言也全绿」的地方。"""

    def test_export_card_is_created_and_result_area_renders(self):
        """点导出 -> 轮询到 done：卡片要建出来，结果区要亮且内容正确。

        这一条同时钉住 Task 6 审查的 P1：`startExport` 以前是
        `jobs.push(...)` 之后又 `addJob(jobs[jobs.length-1])`，而 addJob 开头的
        「已经在 jobs 里就 return」让它当场早退 —— 卡片不建、`els[id]` 不存在、
        `updateJob` 早退、`renderExport` 永不执行，面板上永远看不到 `.list` 路径。
        """
        f = run_probe("enqueue_render")
        self.assertEqual(f["exportCalls"], ["/api/export"], "面板没有恰好提交一次导出")
        self.assertTrue(f["jobInJobs"], "导出任务没有登记进 jobs")
        self.assertTrue(f["cardCreated"],
                        "导出任务的卡片没建出来（els 里只有 %r）" % (f["cardIds"],))
        self.assertEqual(f["expResDisplay"], "",
                         "导出完成后结果区还是隐藏的（display=%r）" % (f["expResDisplay"],))
        self.assertEqual(f["expListText"], f["listPath"], "结果区的 .list 路径不对")
        counts = f["expCountsText"]
        for piece in ("片段 12", "进 .list 10", "短句 1", "未命名 1",
                      "45.6", "艾丽丝 7", "鲁迪乌斯 3"):
            self.assertIn(piece, counts, "计数行缺 %r：%r" % (piece, counts))
        self.assertEqual(f["expTip"], "导出完成。")

    def test_warning_area_is_shown_only_when_the_backend_sends_one(self):
        """warning 为空时那块必须是隐藏的（U8：`if(true)` 会让它永远显示）。"""
        f = run_probe("enqueue_render")
        self.assertEqual(f["expWarnDisplay"], "none",
                         "后端没给 warning，警告区却显示了：%r"
                         % (f.get("expWarnText"),))
        self.assertEqual(f["expWarnText"], "")

        g = run_probe("warning_shown")
        self.assertEqual(g["expWarnDisplay"], "", "后端给了 warning，警告区没显示")
        self.assertEqual(g["expWarnText"], g["warningSentinel"],
                         "警告区显示的不是后端给的那句话")

    def test_non_list_export_displays_its_manifest(self):
        from urllib.parse import parse_qs, urlsplit

        f = run_probe("indextts_render")
        self.assertEqual(f["exportFormat"], "indextts")
        self.assertEqual(f["expListText"], f["manifestPath"])
        self.assertIn("标注 10", f["expCountsText"])
        self.assertEqual(f["workName"], "无职转生")
        query = parse_qs(urlsplit(f["openedFolder"]).query)
        self.assertEqual(query["path"], [f["manifestPath"].rsplit("\\", 1)[0]])

    def test_progress_and_stage_are_visible_while_running(self):
        """分钟级的导出不能「零进度」：面板要显示阶段与百分比。"""
        f = run_probe("progress_tip")
        self.assertTrue(f["cardCreated"], "导出卡片没建出来")
        self.assertEqual(f["jobStatus"], "run")
        self.assertIn("42%", f["expTip"], "运行中面板没有显示百分比：%r" % (f["expTip"],))
        self.assertIn("导出中", f["expTip"])

    def test_error_status_is_shown_with_the_reason(self):
        f = run_probe("error_tip")
        self.assertTrue(f["expTip"].startswith("导出失败"), f["expTip"])
        self.assertIn("磁盘空间不足", f["expTip"])


@unittest.skipUnless(NODE, SKIP_REASON)
class TestExportButtonLock(unittest.TestCase):
    """导出按钮的锁定：连点只入队一次，而**任何**结局都要把按钮放回来。

    原来 `#expGo` 在 `finally` 里无条件解锁：`POST /api/export` 只负责入队、
    几十毫秒就返回，于是「导出中」那几分钟按钮一直是亮的 —— 用户看没反应就
    再点一次，第二次被后端 `_export_lock` 串行化，同一批素材导出两遍、白等。

    反过来「只在 renderExport 里解锁」也不行：`dropJob` / `abandonJobs` 会把
    任务从 `jobs` 里整个拿掉，那之后 `updateJob` 再也不会被调用，按钮就**永久
    卡死**（上一轮就是因为这个才没动它）。所以这里两组都钉：锁得住 + 放得开。
    """

    def test_double_click_enqueues_only_once(self):
        f = run_probe("export_lock")
        self.assertEqual(f["postsAfterFirstClick"], 1, "第一次点击没入队")
        self.assertTrue(f["disabledAfterEnqueue"],
                        "入队返回后按钮就解锁了 —— 这正是连点会二次入队的原因")
        self.assertEqual(f["postsAfterSecondClick"], 1,
                         "连点入队了两次：第二次会被后端 _export_lock 串行化，白跑几分钟")
        self.assertTrue(f["disabledAfterSecondClick"], "第二次点击之后按钮亮了")

    def test_button_comes_back_after_the_job_finishes(self):
        f = run_probe("enqueue_render")
        self.assertFalse(f["expGoDisabled"],
                         "导出完成后按钮没解锁，用户没法再导出")

    def test_button_comes_back_when_the_job_disappears(self):
        """任务在轮询里报 404（后端重启 / 任务没了）→ dropJob。"""
        f = run_probe("export_job_gone")
        self.assertTrue(f["disabledAfterEnqueue"], "入队后按钮没锁住")
        self.assertTrue(f["dropped"], "探针没能把任务拿掉（这条用例本身失效了）")
        self.assertFalse(f["disabledAfterJobGone"],
                         "任务被 dropJob 拿掉后按钮永久卡死")
        self.assertEqual(f["postsAfterReclick"], 2,
                         "按钮看着可用，点了却没入队（只是变了个颜色）")

    def test_button_comes_back_after_abandon(self):
        """用户点卡片上的「放弃」→ `abandonJobs` → `dropJob`（同一条解锁路径）。"""
        f = run_probe("export_abandon")
        self.assertTrue(f["confirmCalls"], "放弃没有弹确认框")
        self.assertEqual(f["jobsAfterAbandon"], 0, "放弃后任务还留在列表里")
        self.assertFalse(f["disabledAfterAbandon"],
                         "任务被放弃后按钮永久卡死")
        self.assertEqual(f["postsAfterReclick"], 2,
                         "放弃后按钮看着可用，点了却没入队")


@unittest.skipUnless(NODE, SKIP_REASON)
class TestPreviewBuildsNoMarkup(unittest.TestCase):
    """P7：导出任务的 `files[].name` 就是**用户填的作品名**，它不能变成标签。

    老写法 `$('acts').innerHTML = ...` 把文件名直接拼进 HTML：
    作品名 `x');alert(1);('` 会生成
    `onclick="preview('ID','x');alert(1);('.list')"` —— 自 XSS。
    这里喂一个既带引号又带尖括号的名字，断言面板压根没有拼过 HTML。
    """

    def test_hostile_file_name_stays_text(self):
        f = run_probe("preview_xss")
        self.assertEqual(f["alertCalls"], [], "预览时执行了 alert")
        # 1) 面板没有用 innerHTML 构造这段列表
        acts_writes = [w for w in f["innerHtmlWrites"] if w["id"] == "acts"]
        self.assertEqual(acts_writes, [],
                         "预览列表是用 innerHTML 拼的：%r" % (acts_writes,))
        self.assertEqual(f["actsInnerHtml"], "")
        # 2) 文件名按原文进 textContent，没有被当成标签吃掉
        buttons = [c for c in f["actsChildren"] if c["tag"] == "button"]
        self.assertTrue(buttons, "预览列表里一个按钮都没有")
        self.assertEqual(buttons[0]["text"], f["hostileName"],
                         "文件名没有按原文渲染")
        self.assertTrue(buttons[0]["hasOnclick"],
                        "文件名按钮没有绑定点击处理（点了没反应）")
        self.assertEqual(buttons[-1]["text"], "下载")
        self.assertEqual(f["srtText"], "探针字幕文本")

    def test_the_probe_payload_would_really_have_injected_markup(self):
        """阳性对照：这个负载拼进 HTML 时**确实**带可执行片段。

        没有这条，上面那条「没有拼串」可能只是因为负载根本无害
        （比如引号被转义了），那它就成了一条永远不会红的空断言。
        """
        legacy = run_probe("preview_xss")["legacyMarkup"]
        self.assertRegex(legacy, r"\son\w+\s*=", "阳性对照失效：负载里没有事件属性")
        self.assertIn("<img", legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
