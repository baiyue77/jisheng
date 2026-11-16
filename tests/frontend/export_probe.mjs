/*
 * 可执行的**前端行为**检查：把 ui.html 里那段真脚本抽出来，在极简假 DOM 上跑
 * 「面板点导出」的真实路径，再把观察到的事实以 JSON 吐给 Python 侧的 unittest。
 *
 * 为什么需要它：`tests/test_ui_speaker_panel.py` 里那些断言全是**静态子串**检查，
 * 审查者把 `if(j.kind==='export') renderExport(j);` 改成 `if(false) ...` 之后
 * 617 条测试全绿 —— 静态断言拦不住「卡片没建 / 结果区不亮 / warning 永远显示」
 * 这一类只会跑起来才暴露的 bug（Task 6 审查 U4/U8）。
 *
 * 用法：node export_probe.mjs <ui.html 路径> <场景>
 * 输出：一行 `__PROBE_JSON__ {...}`（Python 侧按前缀取这行反序列化）。
 *
 * 这里刻意**不引第三方依赖**（没有 jsdom）：只实现脚本真正用到的那几个 DOM API。
 * 假 DOM 不实现 HTML 解析，所以 innerHTML 只被**记录**下来供 Python 断言
 * 「有没有拿用户数据去拼标签」。
 */
import fs from 'node:fs';
import vm from 'node:vm';

const [, , htmlPath, scenario = 'enqueue_render'] = process.argv;

function emit(facts) {
  process.stdout.write('__PROBE_JSON__ ' + JSON.stringify(facts) + '\n');
  process.exit(0);
}

const html = fs.readFileSync(htmlPath, 'utf8');
const matched = html.match(/<script>([\s\S]*)<\/script>/);
if (!matched) emit({ ok: false, fatal: 'ui.html 里找不到 <script> 块' });
let code = matched[1];

// --------------------------------------------------------------------------
// 极简假 DOM
// --------------------------------------------------------------------------
const innerHtmlWrites = [];   // {id, value}：任何一次 innerHTML 赋值都记下来

function mkEl(tag = 'div', id = '') {
  const el = {
    tagName: tag, __id: id, __children: [], style: {}, dataset: {}, files: [],
    value: '', checked: false, disabled: false, title: '', src: '', href: '',
    scrollTop: 0, scrollHeight: 0, clientHeight: 0, returnValue: '',
    textContent: '', _innerHTML: '', onclick: null,
    className: '',
    classList: { add() {}, remove() {}, contains() { return false; } },
    appendChild(c) { this.__children.push(c); return c; },
    removeChild(c) {
      const i = this.__children.indexOf(c);
      if (i >= 0) this.__children.splice(i, 1);
    },
    remove() {}, replaceChildren(...c) { this.__children = c; },
    addEventListener() {}, removeEventListener() {},
    scrollIntoView() {}, focus() {},
    /* 真浏览器里 `disabled` 的按钮**不派发** click。假 DOM 必须照这个走 ——
       否则「连点」用例测的是一个不存在宽松环境：按钮锁没锁住永远测不出来
       （`click()` 写成无条件调用 onclick 时，第二次点击照样会入队）。 */
    click() {
      if (this.disabled) return;
      if (typeof this.onclick === 'function') this.onclick();
    },
    setAttribute() {},
    getAttribute() { return null; },
    querySelector() { return mkEl('div'); },
    querySelectorAll() { return []; },
    insertAdjacentHTML() {},
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._innerHTML; },
    set(v) {
      this._innerHTML = String(v);
      innerHtmlWrites.push({ id: this.__id || this.tagName, value: String(v) });
    },
  });
  return el;
}

const byId = new Map();
const document = {
  getElementById(id) {
    if (!byId.has(id)) byId.set(id, mkEl('div', id));
    return byId.get(id);
  },
  createElement: (t) => mkEl(t),
  createDocumentFragment: () => mkEl('fragment'),
  addEventListener() {},
  querySelector() { return mkEl('div'); },
  querySelectorAll() { return []; },
  body: mkEl('body', 'body'),
};

// addJob 会对卡片根做 `root.querySelectorAll('.step')`（展开 + forEach），
// createElement('div') 必须能应答这个选择器，否则建卡片当场抛。
const stepEls = [0, 1, 2, 3, 4].map(() => mkEl('div'));
const origCreate = document.createElement;
document.createElement = (t) => {
  const el = origCreate(t);
  if (t === 'div') {
    el.querySelector = () => mkEl('div');
    el.querySelectorAll = (sel) => (sel === '.step' ? stepEls : []);
  }
  return el;
};

globalThis.document = document;
globalThis.window = { open() {}, prompt: () => '' };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
const alertCalls = [];
globalThis.alert = (s) => alertCalls.push(String(s));
/* `abandonJobs` 会先弹确认框。只有「放弃」那个场景点「确定」—— 别的场景里
   意外走到放弃路径时应当停下来（没被放弃，用例自然红），而不是静默成功。 */
const confirmCalls = [];
globalThis.confirm = (msg) => {
  confirmCalls.push(String(msg));
  return scenario === 'export_abandon';
};
globalThis.URL = URL;
globalThis.Blob = class {};
globalThis.FormData = class { append() {} };

// 轮询里 `await new Promise(r=>setTimeout(r,800))`：把长延时压成 0，
// 用例不必等 800 ms（语义不变，仍是异步等待）。
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms) => realSetTimeout(fn, ms > 100 ? 0 : ms);

// --------------------------------------------------------------------------
// fetch 桩：按场景决定 /api/status 的返回
// --------------------------------------------------------------------------
const EXPORT_ID = 'EXPID1';
const WORK_NAME = '无职转生';
const LIST_PATH = 'D:\\Jisheng\\output\\说话人\\无职转生.list';
const INDEX_PATH = 'D:\\Jisheng\\output\\说话人\\无职转生\\transcripts.jsonl';
const REPORT_PATH = 'D:\\Jisheng\\output\\说话人\\无职转生.导出报告.json';
const WARNING_SENTINEL = '【探针】warning 文案占位：不要钉在这里';

const DONE_RESULT = {
  list: LIST_PATH, report_path: REPORT_PATH, n_clips: 12, n_list: 10,
  n_short: 1, n_unknown: 1, total_mb: 45.6,
  folders: { '艾丽丝': 7, '鲁迪乌斯': 3 },
};

const SCENARIOS = {
  // 后端在轮询里报 done（无 warning）：结果区必须亮，且内容正确
  enqueue_render: { status: 'done', result: DONE_RESULT },
  indextts_render: { status: 'done', result: {
    ...DONE_RESULT, list: undefined, format: 'indextts', manifest_path: INDEX_PATH,
  } },
  // 后端报 done 且带 warning：warning 区必须显示
  warning_shown: { status: 'done', result: Object.assign({}, DONE_RESULT,
                                                         { warning: WARNING_SENTINEL }) },
  // 后端还在跑：面板必须显示阶段与百分比（不能等到 800 ms 轮询才有反馈）
  progress_tip: { status: 'run', stage: '导出中', progress: 0.42 },
  // 后端报错：面板必须显示失败原因
  error_tip: { status: 'error', error: '【探针】磁盘空间不足' },
  // P7：文件名里带引号/尖括号也不能拼出可执行片段
  preview_xss: { status: 'done', result: DONE_RESULT },
  // 还在跑：入队返回后按钮**必须**保持禁用，否则连点会二次入队、白跑几分钟
  export_lock: { status: 'run', stage: '导出中', progress: 0.10 },
  // 任务在轮询里报 404（后端重启 / 任务没了）→ 被 dropJob 拿掉
  export_job_gone: { status: 'run', stage: '导出中', progress: 0.10 },
  // 用户点卡片上的「放弃」→ abandonJobs → dropJob（同一条解锁路径）
  export_abandon: { status: 'run', stage: '导出中', progress: 0.10 },
};

if (!SCENARIOS[scenario]) emit({ ok: false, fatal: '未知场景：' + scenario });

/* 后端说这条任务不存在（重启 / 已被放弃 / 被别人删了）。kickPoll 拿到 404 的
   处理是 `gone.push(id)` → `dropJob`，正是「任务被拿掉」那条路径。
   这个开关**由用例自己打开**：一上来就 404 的话，任务在第一次 settle 里就被拿掉，
   「入队后按钮有没有锁住」那一瞬间根本没被观察到（那条断言就成了空的）。 */
const STATUS_404 = (scenario === 'export_job_gone');
let jobGone = false;

const fetchCalls = [];
let exportPayload;
globalThis.fetch = async (url, options) => {
  const u = String(url);
  fetchCalls.push(u);
  let body = {};
  if (u.startsWith('/api/models')) {
    body = { asr: [], separation: [], diarize: [], default_output: '',
             sep_output: '', hint: '' };
  } else if (u.startsWith('/api/jobs')) {
    body = { jobs: [] };
  } else if (u.startsWith('/api/resumable')) {
    body = { items: [] };
  } else if (u.startsWith('/api/workdirs')) {
    body = { items: [] };
  } else if (u.startsWith('/api/export')) {
    exportPayload = JSON.parse(options.body);
    body = { id: EXPORT_ID, status: 'wait', kind: 'export' };
  } else if (u.startsWith('/api/abandon')) {
    body = { ok: true };
  } else if (u.startsWith('/api/status')) {
    if (STATUS_404 && jobGone) {
      return { ok: false, status: 404,
               json: async () => ({ error: 'no such job' }),
               text: async () => '' };
    }
    const s = SCENARIOS[scenario];
    body = Object.assign({ id: EXPORT_ID, kind: 'export', stage: '', progress: 0,
                           logs: [], files: [], error: '', opts: {} }, s);
  } else if (u.startsWith('/api/result')) {
    body = {};
  }
  return {
    ok: true, status: 200,
    json: async () => body,
    text: async () => (u.startsWith('/api/result') ? '探针字幕文本' : ''),
  };
};

// 把内部函数暴露给探针（只读观察，不改行为）
code += `
;globalThis.__probe = {
  startExport, updateJob, renderExport, addJob, dropJob, abandonJobs, preview,
  jobs: () => jobs,
  els: () => els,
  setSpk: (v) => { spk = v; },
  expRes: () => $('expRes').style.display,
  expList: () => $('expList').textContent,
  expListTitle: () => $('expList').title,
  expCounts: () => $('expCounts').textContent,
  expWarnDisplay: () => $('expWarn').style.display,
  expWarnText: () => $('expWarn').textContent,
  expTip: () => $('expTip').textContent,
  expGoDisabled: () => $('expGo').disabled,
  actsChildren: () => ($('acts').__children || []).map(
    c => ({tag: c.tagName, text: c.textContent, hasOnclick: typeof c.onclick === 'function'})),
  actsInnerHtml: () => $('acts').innerHTML,
  actStyle: (i) => (($('acts').__children || [])[i] || {}).style || {},
};`;

const ctx = vm.createContext(globalThis);
vm.runInContext(code, ctx, { filename: 'ui.html' });

const P = globalThis.__probe;
if (!P) emit({ ok: false, fatal: 'ui.html 脚本没有跑起来（__probe 未定义）' });

async function waitFor(pred, ms = 3000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if (pred()) return true; } catch (e) { /* 还没就绪 */ }
    await new Promise((r) => realSetTimeout(r, 5));
  }
  try { return pred(); } catch (e) { return false; }
}

/* 文本类事实一律经它出 JSON：`textContent = undefined`（后端没给 warning 时
   渲染函数就会这么写）会让 JSON.stringify **整个键都消失**，Python 侧于是拿到
   KeyError 而不是「值不对」—— 断言照样红，但红得看不出原因。这里统一成
   `null` 或字符串，键永远在。 */
const text = (v) => (v === undefined || v === null ? null : String(v));

const cardIds = () => Object.keys(P.els());

async function enqueue() {
  P.setSpk({ job: 'j1', work: 'EP01' });
  document.getElementById('spkWork').value = WORK_NAME;
  await P.startExport();
}

/* 锁相关场景：面板要在**用户真点按钮**的路径上被驱动（不是直接调 startExport），
   因为「连点被挡住」这件事本身就发生在按钮的 disabled 语义上。 */
const exportPosts = () => fetchCalls.filter((u) => u.startsWith('/api/export')).length;
const settle = () => new Promise((r) => realSetTimeout(r, 40));

const facts = {
  ok: true, scenario,
  exportId: EXPORT_ID, listPath: LIST_PATH, warningSentinel: WARNING_SENTINEL,
  alertCalls,
};

try {
  if (scenario === 'export_lock' || scenario === 'export_job_gone'
      || scenario === 'export_abandon') {
    P.setSpk({ job: 'j1', work: 'EP01' });
    document.getElementById('spkWork').value = WORK_NAME;
    const btn = document.getElementById('expGo');

    btn.click();                       // 第一次点击：走真实的 onclick → startExport
    await waitFor(() => exportPosts() === 1);
    await settle();                    // 让 startExport 的 finally 跑完
    facts.postsAfterFirstClick = exportPosts();
    facts.disabledAfterEnqueue = P.expGoDisabled();
    facts.tipAfterEnqueue = text(P.expTip());

    if (scenario === 'export_lock') {
      btn.click();                     // 用户看没反应，又点了一下
      await settle();
      facts.postsAfterSecondClick = exportPosts();
      facts.disabledAfterSecondClick = P.expGoDisabled();
    } else if (scenario === 'export_job_gone') {
      jobGone = true;                  // 后端现在说这条任务没了
      facts.dropped = await waitFor(() => P.jobs().length === 0);
      facts.disabledAfterJobGone = P.expGoDisabled();
      btn.click();                     // 解锁之后必须**真的**能再入队，不是只变个颜色
      await settle();
      facts.postsAfterReclick = exportPosts();
    } else {
      // 用户点卡片上的「放弃」：abandonJobs → dropJob（同一条解锁路径）
      await P.abandonJobs([EXPORT_ID]);
      await settle();
      facts.confirmCalls = confirmCalls;
      facts.jobsAfterAbandon = P.jobs().length;
      facts.disabledAfterAbandon = P.expGoDisabled();
      btn.click();
      await settle();
      facts.postsAfterReclick = exportPosts();
    }
    facts.exportCalls = fetchCalls.filter((u) => u.startsWith('/api/export'));
    emit(facts);
  }

  if (scenario === 'preview_xss') {
    // 用户填的作品名会被后端放进 files[].name（`<作品名>.list`）。
    // 这里喂一个既含引号又含尖括号的名字：拼进 innerHTML 就会变成标签/属性。
    const hostile = `x');alert(1);('<img src=x onerror=alert(2)>.list`;
    P.addJob({ id: 'HOSTILE', name: '探针', status: 'done', logs: [],
               files: [{ name: hostile, path: 'D:\\x.list', kind: '训练集清单' }] });
    await P.preview('HOSTILE', hostile);
    facts.hostileName = hostile;
    facts.actsChildren = P.actsChildren();
    facts.actsInnerHtml = P.actsInnerHtml();
    facts.innerHtmlWrites = innerHtmlWrites.slice();
    facts.srtText = document.getElementById('srt').textContent;
    // 阳性对照：老写法（把名字拼进 innerHTML）会长成什么样 —— 用它证明
    // 这个负载**确实**带着可执行片段，所以「没有拼串」这条断言是有意义的。
    facts.legacyMarkup =
      `<button onclick="preview('HOSTILE','${hostile}')">${hostile}</button>`;
    emit(facts);
  }

  if(scenario === 'indextts_render') document.getElementById('expFormat').value = 'indextts';
  await enqueue();
  const job = P.jobs().find((j) => j.id === EXPORT_ID);
  facts.jobInJobs = !!job;
  facts.exportCalls = fetchCalls.filter((u) => u.startsWith('/api/export'));
  if (scenario === 'progress_tip') {
    await waitFor(() => String(P.expTip()).includes('42%'));
  } else if (scenario === 'error_tip') {
    await waitFor(() => String(P.expTip()).startsWith('导出失败'));
  } else if (scenario === 'warning_shown') {
    await waitFor(() => P.expWarnDisplay() === '' && P.expWarnText());
  } else {
    await waitFor(() => P.expList() !== '');
  }
  facts.cardIds = cardIds();
  facts.cardCreated = cardIds().includes(EXPORT_ID);
  facts.jobStatus = job && job.status;
  facts.expResDisplay = text(P.expRes());
  facts.expListText = text(P.expList());
  facts.expListTitle = text(P.expListTitle());
  facts.expCountsText = text(P.expCounts());
  facts.expWarnDisplay = text(P.expWarnDisplay());
  facts.expWarnText = text(P.expWarnText());
  facts.expTip = text(P.expTip());
  facts.expGoDisabled = P.expGoDisabled();
  if(scenario === 'indextts_render'){
    document.getElementById('expOpen').click();
    facts.openedFolder = fetchCalls.find(u => u.startsWith('/api/openfolder'));
    facts.exportFormat = exportPayload.format;
    facts.workName = document.getElementById('spkWork').value;
    facts.manifestPath = INDEX_PATH;
  }
} catch (err) {
  facts.ok = false;
  facts.error = String((err && err.stack) || err);
}
emit(facts);
