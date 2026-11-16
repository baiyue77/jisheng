#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""ui.html 说话人面板的静态断言（前端没有 JS 测试框架，只能在 Python 侧查源码）。

这里只做**确定性**断言：从 ui.html 里切出函数体，检查特定绑定用的是哪个字段。
不做整体语法检查（那是 node --check 的活）。
"""

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import speaker_labels as sl  # noqa: E402
import webui  # noqa: E402

HTML = (HERE / "ui.html").read_text(encoding="utf-8")

# 刻意不用「一行匹配」：先定位函数头，再从第一个 { 开始做括号配对切出整个函数体。
def func_body(src, name):
    """按大括号配对从源码里切出函数体。找不到就 fail 得很难看（而不是静默通过）。"""
    head = (r"(?:^|\n)\s*(?:async\s+)?function\s+"
            + re.escape(name) + r"\s*\(.*?\)\s*\{")
    m = re.search(head, src)
    if m is None:
        raise AssertionError(f"ui.html 里找不到函数 {name}")
    start = src.index("{", m.end() - 1)
    depth = 0
    for i in range(start, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1:i]
    raise AssertionError(f"ui.html 里 {name} 的大括号不配对")


def _decode_js_string(literal):
    """把 JS 字符串字面量解码成 Python str（\\uXXXX / \\n 都要认）。

    用 json.loads 而不是 unicode_escape：后者会把中文字符按 latin-1 拆坏。
    """
    body = literal[1:-1]
    if literal[0] == "`":
        return body
    try:
        return json.loads('"' + body.replace('"', '\\"') + '"')
    except Exception:                                   # noqa: BLE001
        return body


def _block_from(src, start):
    """从 start（必须是 '{'）起按大括号配对切出块内部。"""
    assert src[start] == "{", repr(src[start:start + 20])
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1:i]
    raise AssertionError("大括号不配对")


def _options_named(body, var):
    """取 `const <var> = document.createElement('option'); ...; sel.appendChild(<var>);` 片段。"""
    m = re.search(
        r"const\s+%s\s*=\s*document\.createElement\('option'\);(.*?)appendChild\(%s\)"
        % (re.escape(var), re.escape(var)),
        body, re.S)
    if m is None:
        raise AssertionError(f"找不到 option 变量 {var}")
    return m.group(1)


def _assigned_value(frag):
    m = re.search(r"\.value\s*=\s*([^;]+);", frag)
    if m is None:
        raise AssertionError("这段 option 片段里没有 .value 赋值")
    return m.group(1).strip()


def _assigned_text(frag):
    m = re.search(r"\.textContent\s*=\s*(.+?);\s*\n", frag, re.S)
    if m is None:
        raise AssertionError("这段 option 片段里没有 .textContent 赋值")
    return m.group(1).strip()


def _string_literals(expr):
    out = []
    for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"", expr):
        lit = m.group(0)
        if len(lit) >= 2:
            out.append(_decode_js_string(lit))
    return out


class TestBuildTargetSelect(unittest.TestCase):
    """逐句改判的「改判到…」下拉必须提交**聚类键**，不能提交显示名。

    提交显示名会让 api_reassign 的 ensure_clusters 造出一个幽灵聚类键：
    那句话从所有列表里消失，而且聚类改名后字幕里留的还是旧名字。
    """

    def setUp(self):
        self.body = func_body(HTML, "buildTargetSelect")
        # 遍历 spk.clusters 的那段（第一个 for 循环体）
        m = re.search(r"for\s*\(\s*const\s+\w+\s+of\s+\(?\s*spk\.clusters\s*\|\|\s*\[\]\s*\)?\s*\)\s*\{",
                      self.body)
        self.assertIsNotNone(m, "buildTargetSelect 里没有遍历 spk.clusters 的循环")
        self.loop = _block_from(self.body, m.end() - 1)

    def test_option_value_is_the_cluster_key(self):
        opt = _options_named(self.loop, "opt")
        self.assertEqual(_assigned_value(opt), "o.id",
                         "逐句改判下拉的 value 必须是聚类键 o.id")

    def test_option_value_is_not_the_display_name(self):
        """这条单独列出来，是为了让「改回 o.name」这种回退当场红。"""
        opt = _options_named(self.loop, "opt")
        self.assertNotEqual(_assigned_value(opt), "o.name")
        self.assertNotIn("o.name", _assigned_value(opt))

    def test_option_text_stays_the_visible_name(self):
        """下拉里给用户看的仍应是角色名 —— 文案函数由循环调用，到它里面去看。"""
        opt = _options_named(self.loop, "opt")
        text = _assigned_text(opt)
        helper = re.match(r"([A-Za-z_$][\w$]*)\s*\(\s*o\s*\)", text)
        if helper:
            text = func_body(HTML, helper.group(1))
        self.assertIn("o.name", text)

    def test_merged_clusters_are_told_apart_by_their_label(self):
        """两个聚类同名（= 已合并）时，光看名字分不清提交的是哪一个，必须带上 label。"""
        opt = _options_named(self.loop, "opt")
        text = _assigned_text(opt)
        helper = re.match(r"([A-Za-z_$][\w$]*)\s*\(\s*o\s*\)", text)
        if helper:
            text = func_body(HTML, helper.group(1))
        self.assertIn("o.label", text,
                      "同名聚类必须靠 o.label 区分，否则用户无法知道选中的是哪个聚类")

    def test_question_mark_and_new_role_options_are_untouched(self):
        self.assertEqual(_assigned_value(_options_named(self.body, "unknown")), "'?'")
        self.assertEqual(_assigned_value(_options_named(self.body, "mk")), "'__new__'")

    def test_only_cluster_keys_and_two_literals_can_be_submitted(self):
        """下拉能提交的值只可能是：聚类键（动态取自 o.id）、`?`、`new:<名字>`、空串。

        这条把「提交的值一定存在于 labels['clusters'] 里」翻译成本文件可验证的形式：
        接口返回的聚类键就是 o.id，所以 value 取 o.id 时提交值必然是聚类键；
        取 o.name 则可能是一个 `clusters` 里根本没有的显示名（幽灵聚类）。
        """
        offered = set()
        for var in ("opt", "blank", "unknown", "mk"):
            offered |= set(_string_literals(_assigned_value(_options_named(self.body, var))))
        self.assertTrue(offered <= {"", "?", "__new__"},
                        f"出现了聚类键以外的字面量目标：{offered}")
        self.assertEqual(_assigned_value(_options_named(self.loop, "opt")), "o.id")


class TestFrontendToBackendIdentity(unittest.TestCase):
    """前端提交的聚类目标 与 后端认的聚类键 必须是同一个东西。

    这是 M1 那条「提交的值一定存在于 labels['clusters'] 里」的端到端版本：
    前端 value 取 o.id，后端 api_speakers 的 id 就是 diarization 的聚类键，
    api_reassign 又按聚类键写 override，所以整条链上不会冒出幽灵聚类。
    """

    def setUp(self):
        import json
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._work_dir = webui.WORK_DIR
        self._out_root = webui.OUTPUT_ROOT
        self._jobs = webui.JOBS
        webui.WORK_DIR = root / "work"
        webui.OUTPUT_ROOT = root / "output"
        webui.JOBS = {}
        self.workdir = webui.WORK_DIR / "EP01"
        self.workdir.mkdir(parents=True)
        dia = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
               {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"}]
        (self.workdir / "diarization.json").write_text(
            json.dumps(dia, ensure_ascii=False), encoding="utf-8")
        segs = [{"start": 0.0, "end": 4.0, "text": "第一句", "speaker": "SPEAKER_00"},
                {"start": 10.0, "end": 15.0, "text": "第二句", "speaker": "SPEAKER_01"}]
        (self.workdir / "segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in segs) + "\n",
            encoding="utf-8")

    def tearDown(self):
        webui.WORK_DIR = self._work_dir
        webui.OUTPUT_ROOT = self._out_root
        webui.JOBS = self._jobs
        self._tmp.cleanup()

    def test_api_cluster_ids_are_the_diarization_keys(self):
        obj, code = webui.api_speakers(None, "EP01")
        self.assertEqual(code, 200)
        self.assertEqual([c["id"] for c in obj["clusters"]],
                         ["SPEAKER_00", "SPEAKER_01"])

    def test_submitting_an_id_round_trips_without_creating_a_new_cluster(self):
        body = func_body(HTML, "buildTargetSelect")
        m = re.search(r"for\s*\(\s*const\s+\w+\s+of\s+\(?\s*spk\.clusters\s*\|\|\s*\[\]\s*\)?\s*\)\s*\{",
                      body)
        value = _assigned_value(_options_named(_block_from(body, m.end() - 1), "opt"))
        # 前端算出来的就是「接口返回的 id」——把它当成要提交的值
        obj, _ = webui.api_speakers(None, "EP01")
        submit = {"o.id": obj["clusters"][1]["id"], "o.name": obj["clusters"][1]["name"]}[value]

        webui.api_assign({"work": "EP01", "names": {"SPEAKER_00": "艾丽丝",
                                                    "SPEAKER_01": "鲁迪乌斯"}})
        res, code = webui.api_reassign({"work": "EP01", "index": 0, "to": submit})
        self.assertEqual(code, 200)
        labels = sl.load_labels(self.workdir)
        self.assertEqual(set(labels["clusters"]), {"SPEAKER_00", "SPEAKER_01"},
                         "提交的不是聚类键：多出了一个幽灵聚类")
        after, _ = webui.api_speakers(None, "EP01")
        by_id = {c["id"]: c for c in after["clusters"]}
        self.assertEqual([s["i"] for s in by_id["SPEAKER_01"]["segments"]], [0, 1])


class TestMergeSelectKeepsSubmittingNames(unittest.TestCase):
    """「合并到…」的语义就是共用名字，提交值必须是名字 —— 别顺手改成 id。

    与逐句改判相反：那里提交的是聚类键（同名聚类必须能区分），
    这里提交的是名字（合并的全部意义就是让两个聚类共用同一个名字）。
    """

    def _merge_loop(self):
        body = func_body(HTML, "renderSpeakers")
        m = re.search(r"sel\.className\s*=\s*'merge';", body)
        self.assertIsNotNone(m, "renderSpeakers 里找不到合并下拉")
        tail = body[m.end():]
        m2 = re.search(r"for\s*\(.*?of\s+spk\.clusters\s*\)\s*\{|"
                       r"for\s*\(let\s+\w+\s*=\s*0;\s*\w+\s*<\s*spk\.clusters\.length;"
                       r"[^)]*\)\s*\{", tail)
        self.assertIsNotNone(m2, "合并下拉里找不到遍历 spk.clusters 的循环")
        return _block_from(tail, m2.end() - 1)

    def test_merge_option_value_is_the_name(self):
        loop = self._merge_loop()
        self.assertEqual(_assigned_value(_options_named(loop, "opt")), "o.name")

    def test_merge_option_value_is_not_the_cluster_key(self):
        loop = self._merge_loop()
        self.assertNotIn("o.id", _assigned_value(_options_named(loop, "opt")))


class TestNoStringInterpolationIntoInnerHtml(unittest.TestCase):
    """前端一律 textContent：说话人面板的函数里不能有 innerHTML 插值。

    ui.html 里另有两处早就存在的 innerHTML 模板（模型下拉与音频标签），
    不在这次改动的范围里，所以这里只扫说话人面板自身的函数体。
    """

    PANEL_FUNCS = ("buildTargetSelect", "segmentRow", "segmentsDetails",
                   "renderSpeakers", "openSpeakers", "openSpeakersByWork",
                   "loadWorkdirs", "assignNames", "reloadSpeakers",
                   "reassignSegment")

    def test_panel_functions_do_not_assign_inner_html(self):
        for fn in self.PANEL_FUNCS:
            body = func_body(HTML, fn)
            self.assertNotIn("innerHTML", body, f"{fn} 里出现了 innerHTML")


def call_args(src, name):
    """切出一个函数所有调用点的实参文本（按括号配对）。"""
    out = []
    for m in re.finditer(re.escape(name) + r"\s*\(", src):
        start = m.end() - 1
        depth = 0
        for j in range(start, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    out.append(src[start + 1:j])
                    break
    return out


class TestOthersIsActuallyRendered(unittest.TestCase):
    """后端把「改判到新建角色」的句子放进 `others`，前端**必须渲染成可操作的行**。

    只显示一个数字是不够的：那些句子不在任何卡片里（它们的聚类键不在分割结果里，
    所以没有卡片），用户就没有任何路径把这句改回去 —— 那正是 M2 要消灭的单向门。
    这条测试的区分力在于它要求 `others` **作为渲染调用的实参**出现，
    而不是只出现在计数文案里。
    """

    def test_render_speakers_passes_others_segments_to_a_render_call(self):
        body = func_body(HTML, "renderSpeakers")
        self.assertIn("spk.others", body, "renderSpeakers 没读 spk.others")
        rendered = [a for a in call_args(body, "segmentsDetails")
                    if "oth.segments" in a]
        self.assertTrue(
            rendered,
            "others 的句子没有被交给 segmentsDetails 渲染（只出了个计数 = 单向门）")

    def test_others_block_offers_a_way_back(self):
        """渲染出来的行必须能改判回真实角色：复用 segmentRow 即可。

        segmentRow 里挂的是 buildTargetSelect，它的选项含所有已命名聚类的
        **聚类键**，所以「改回去」这条路是通的。
        """
        body = func_body(HTML, "renderSpeakers")
        rendered = [a for a in call_args(body, "segmentsDetails")
                    if "oth.segments" in a]
        self.assertTrue(rendered)
        row_body = func_body(HTML, "segmentRow")
        self.assertIn("buildTargetSelect", row_body)
        self.assertIn("reassignSegment", row_body)


class TestExportPanelControls(unittest.TestCase):
    """「导出训练集」面板：控件齐不齐、提交给谁、结果怎么显示。

    这些是纯静态断言（前端没有 JS 测试框架）。重点抓三类事故：
    1. 少了某个必须的控件（用户没法选采样率/音源/丢短句）；
    2. 提交到别的接口、或漏了某个字段（后端收到默认值，用户的选择被吞掉）；
    3. 用 innerHTML 渲染结果（`.list` 路径/文件名要按文本渲染）。
    """

    PANEL_IDS = ("expFormat", "expSr", "expSource", "expDropShort", "expGo",
                 "expList", "expCounts", "expWarn", "expOpen", "expRes")

    def test_every_control_exists_in_the_card(self):
        card = HTML[HTML.index('id="spkCard"'):HTML.index('id="previewCard"')]
        for cid in self.PANEL_IDS:
            self.assertIn('id="%s"' % cid, card, "导出区缺少 #%s" % cid)

    def test_sample_rate_options_are_the_two_allowed(self):
        card = HTML[HTML.index('id="expSr"'):HTML.index('id="expSource"')]
        self.assertIn('value="32000"', card)
        self.assertIn('value="44100"', card)
        # 只允许这两档：多一个 48000 就会出现「面板能选、后端 400」
        self.assertEqual(len(re.findall(r'<option value="(\d+)"', card)), 2)

    def test_source_options_are_the_two_modes(self):
        card = HTML[HTML.index('id="expSource"'):HTML.index('id="expDropShort"')]
        self.assertIn('value="hq"', card)
        self.assertIn('value="fast"', card)

    def test_export_posts_to_the_export_endpoint_with_all_fields(self):
        body = func_body(HTML, "startExport")
        self.assertIn("'/api/export'", body)
        # 五个字段一个都不能少：少 work_name 就用不了自定义作品名，
        # 少 drop_short 则是「勾了没用」。
        for field in ("job", "work", "work_name", "sr", "source", "drop_short", "format"):
            self.assertIn(field, body, "startExport 没提交 %s" % field)

    def test_export_button_is_bound(self):
        self.assertRegex(HTML, r"\$\('expGo'\)\.onclick\s*=")

    def test_open_folder_button_uses_the_existing_endpoint(self):
        m = re.search(r"\$\('expOpen'\)\.onclick[\s\S]{0,400}?\};", HTML)
        self.assertIsNotNone(m, "找不到 #expOpen 的点击处理")
        self.assertIn("/api/openfolder", m.group(0))

    def test_result_is_rendered_with_text_content(self):
        body = func_body(HTML, "renderExport")
        self.assertNotIn("innerHTML", body, "renderExport 里出现了 innerHTML")
        for cid in ("expList", "expCounts", "expWarn"):
            self.assertIn("'%s'" % cid, body, "renderExport 没写 #%s" % cid)
        # warning 只在后端给了的时候才显示（高质量源不该弹警告）
        self.assertIn("r.warning", body)
        self.assertIn("expWarn').style.display", body)

    def test_result_renderer_is_called_from_the_shared_update_loop(self):
        """导出结果必须由既有轮询更新路径推送，不能另起一个 setInterval。"""
        self.assertIn("renderExport(j)", func_body(HTML, "updateJob"))

    def test_start_export_does_not_push_then_add(self):
        """登记任务只能经 `addJob` 一次（不能先 `jobs.push` 再 `addJob`）。

        P1 的原始 bug：`jobs.push({...}); addJob(jobs[jobs.length-1]);` ——
        `addJob` 开头的「已经在 jobs 里就 return」让它当场早退，卡片不建、
        `els[id]` 不存在、`updateJob` 早退、`renderExport` 永不执行。

        这里只是**结构**上的快速护栏（真行为由
        `test_ui_export_behavior.py` 用真脚本 + 假 DOM 跑出来）。
        """
        body = func_body(HTML, "startExport")
        self.assertIn("addJob(", body, "startExport 没有经 addJob 登记任务")
        self.assertNotIn("jobs.push(", body,
                         "startExport 又自己 push 了一次：addJob 会早退，卡片建不出来")

    def test_help_text_names_the_backend_export_root(self):
        """面板帮助文案里的导出根必须与后端常量一致（T5）。

        导出子目录跟随任务的输出位置。具体文件名在结果区展示。
        """
        card = HTML[HTML.index('id="spkExport"'):HTML.index('id="expRes"')]
        name = webui.TRAIN_EXPORT_DIRNAME
        self.assertIn("<code>%s\\</code>" % name, card)


class TestPreviewBuildsNoMarkup(unittest.TestCase):
    """`preview()` 不能把文件名拼进 innerHTML（P7）。

    导出任务的 `files[].name` 就是**用户填的作品名**（`<作品名>.list`）。
    老写法 `$('acts').innerHTML = ...` 把名字原样拼进 HTML：作品名
    `x');alert(1);('` 会生成
    `onclick="preview('EXPID1','x');alert(1);('.list')"` —— 自 XSS。
    真行为断言在 `test_ui_export_behavior.py`（喂带引号/尖括号的名字跑一遍）。
    """

    def test_preview_does_not_assign_inner_html(self):
        body = func_body(HTML, "preview")
        # 只找**赋值**，不找那个词：函数里的注释正好在解释「绝不拼 innerHTML」
        self.assertIsNone(re.search(r"\.innerHTML\s*(\+)?=", body),
                          "preview 里给 innerHTML 赋值了")
        self.assertIn("textContent", body, "文件名没有按文本渲染")
        self.assertIn("createElement", body, "预览列表没有用 DOM 构造")


if __name__ == "__main__":
    unittest.main(verbosity=2)
