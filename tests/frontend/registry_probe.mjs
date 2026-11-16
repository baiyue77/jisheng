/*
 * 角色库面板的**可执行前端检查**：把 ui.html 里那段真脚本抽出来，在极简假 DOM 上
 * 走真实路径（openSpeakersByWork -> loadRegistry -> renderSpeakers -> 用户输入/点击），
 * 把观察到的事实以 JSON 吐给 Python 侧的 unittest。
 *
 * 为什么需要它（沿用 export_probe.mjs 的理由）：静态子串断言拦不住「建议被写进
 * 输入框 / suggest 档也预填 / 删除没确认 / 名字拼进 innerHTML」这一类只有跑起来
 * 才暴露的 bug —— Task 6 已经证明过一次（把 `if(j.kind==='export')` 改成
 * `if(false)` 后 617 条测试全绿）。
 *
 * 用法：node registry_probe.mjs <ui.html 路径> <场景>
 * 输出：一行 `__PROBE_JSON__ {...}`。
 *
 * 假 DOM 比 export_probe 多两样东西：**属性表**（setAttribute/getAttribute，
 * 用来断言 `list="spkNames"` 的补全接线）与**事件记录**（addEventListener 真的
 * 存下来，探针可以 fire('input')，从而测到「用户改名字后提示跟着变」）。
 */
import fs from 'node:fs';
import vm from 'node:vm';

const [, , htmlPath, scenario = 'prefill_auto'] = process.argv;

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
    tagName: tag, __id: id, __children: [], __attrs: {}, __listeners: {},
    style: {}, dataset: {}, files: [],
    value: '', checked: false, disabled: false, title: '', src: '', href: '',
    scrollTop: 0, scrollHeight: 0, clientHeight: 0, returnValue: '',
    textContent: '', _innerHTML: '', onclick: null, className: '',
    classList: { add() {}, remove() {}, contains() { return false; } },
    appendChild(c) { this.__children.push(c); return c; },
    removeChild(c) {
      const i = this.__children.indexOf(c);
      if (i >= 0) this.__children.splice(i, 1);
    },
    remove() {}, replaceChildren(...c) { this.__children = c; },
    addEventListener(type, fn) {
      (this.__listeners[type] = this.__listeners[type] || []).push(fn);
    },
    removeEventListener() {},
    /* 探针专用（真浏览器里没有）：把事件真的派发给注册过的处理函数 */
    fire(type, ev) {
      for (const fn of (this.__listeners[type] || [])) fn(ev || {});
    },
    scrollIntoView() {}, focus() {}, click() { if (this.onclick) this.onclick(); },
    setAttribute(k, v) { this.__attrs[k] = String(v); },
    getAttribute(k) { return (k in this.__attrs) ? this.__attrs[k] : null; },
    querySelector() { return mkEl('div'); },
    querySelectorAll() { return []; },
    insertAdjacentHTML() {},
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._innerHTML; },
    set(v) {
      this._innerHTML = String(v);
      innerHtmlWrites.push({ id: this.__id || '', tag: this.tagName,
                             value: String(v) });
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

globalThis.document = document;
globalThis.window = { open() {}, prompt: () => '', confirm: () => true };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
const alertCalls = [];
globalThis.alert = (s) => alertCalls.push(String(s));
const confirmCalls = [];
const promptCalls = [];
globalThis.URL = URL;
globalThis.Blob = class {};
globalThis.FormData = class { append() {} };

const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms) => realSetTimeout(fn, ms > 100 ? 0 : ms);

// --------------------------------------------------------------------------
// 场景数据
// --------------------------------------------------------------------------
const WORK = '无职转生';
const OTHER_WORK = '别的作品';
const ALICE = '艾丽丝';
const RUDEUS = '鲁迪乌斯';
const STRANGER = '路人';
/* 作品名与角色名一样是**用户数据**（面板的「作品名」输入框 → 库 → 角色库区
   渲染）。敌意 payload 必须同时覆盖两者，否则「不许 innerHTML 拼用户数据」
   这条只钉住了角色名那一半（Task 7 审查 T2：变异 F10 当时零红）。 */
const HOSTILE_WORK = `无职<img src=x onerror=alert(4)>转生`;
const HOSTILE_WORK2 = `"><script>alert(5)</script>`;
const HOSTILE = `x');alert(1);('<img src=x onerror=alert(2)>`;

function page(name, opts = {}) {
  return Object.assign({
    page: name, n_enroll: 2, n_episodes: 2, jobs: ['j1', 'j2'],
    total_s: 61.5, n_vectors: 4, vectors_usable: true, updated_at: '',
  }, opts);
}

function registry() {
  return {
    embedding_model: 'pyannote/speaker-diarization-3.1/wespeaker-voxceleb-resnet34-LM',
    dim: 256, thresholds: { auto: 0.65, suggest: 0.45 },
    n_works: 2, n_pages: 3, n_episodes: 4,
    works: [
      { work: WORK, model: 'm1', vectors_usable: true, n_pages: 2, n_episodes: 3,
        n_vectors: 6, total_s: 120.0, updated_at: '',
        pages: [page(ALICE), page(RUDEUS, { n_enroll: 1, n_episodes: 1,
                                            vectors_usable: false })] },
      { work: OTHER_WORK, model: 'm1', vectors_usable: true, n_pages: 1,
        n_episodes: 1, n_vectors: 2, total_s: 10.0, updated_at: '',
        pages: [page(STRANGER, { n_enroll: 1, n_episodes: 1 })] },
    ],
  };
}

/* `GET /api/registry/health` 的两种真实形态（数值刻意取成不规整的小数，
   这样 toFixed(3) 有没有被用、有没有被写死都能看出来）。
   · ok：有间隔，给得出建议阈值
   · thin：一个角色 / 3 条向量 —— 四个字段全是 **null 而不是 0**，这正是
     `speaker_registry.health_check` 在样本不够时的真实返回（空库同形）。 */
const HEALTH_OK = {
  ok: true, n_pages: 2, n_segments: 6, n_pairs_same: 4, n_pairs_diff: 12,
  same_min: 0.8123, same_min_page: RUDEUS,
  diff_max: 0.5011, diff_pair: [ALICE, STRANGER],
  gap: 0.3112, overlap: false,
  suggested: { auto: 0.7011, suggest: 0.6011 },
  thresholds: { auto: 0.65, suggest: 0.45 },
  model: 'probe', spaces: {},
  known_limit: '样本少时这两个数没有统计意义。',
};

const HEALTH_THIN = {
  ok: false, n_pages: 1, n_segments: 3, n_pairs_same: 0, n_pairs_diff: 0,
  same_min: null, same_min_page: '', diff_max: null, diff_pair: null,
  gap: null, overlap: null, suggested: null,
  thresholds: { auto: 0.65, suggest: 0.45 },
  model: 'probe', spaces: {},
  known_limit: '样本少时这两个数没有统计意义。',
};

function healthBody() {
  if (scenario === 'health_thin') return HEALTH_THIN;
  const h = JSON.parse(JSON.stringify(HEALTH_OK));
  if (scenario === 'hostile_names') {
    // 角色名也是用户数据：体检区的「最低的是「X」」同样要走 textContent
    h.same_min_page = HOSTILE;
    h.diff_pair = [HOSTILE, HOSTILE_WORK];
  }
  return h;
}

let CLUSTERS = [];
let SPK_WORK = WORK;          // /api/speakers 的 work_name（hostile 场景会换掉）
function cl(id, name, match) {
  return { id, label: '[说话人' + (id.slice(-1)) + ']', name, n_segments: 2,
           total_s: 10.0, segments: [], clips: [], match: match || null };
}
function speakers() {
  return { job: 'j1', work: 'EP01', work_name: SPK_WORK, source: '',
           unknown_segments: 0, unknown: { n: 0, segments: [] },
           others: { n: 0, segments: [] }, clusters: CLUSTERS };
}

const M_AUTO = { work: WORK, page: ALICE, score: 0.9123, band: 'auto', auto: ALICE };
const M_SUGGEST = { work: WORK, page: RUDEUS, score: 0.5123, band: 'suggest',
                    auto: null };
const M_AUTO_LOST = { work: WORK, page: ALICE, score: 0.8011, band: 'auto',
                      auto: null };

const SCENARIOS = {
  prefill_auto: () => { CLUSTERS = [cl('SPEAKER_00', '', M_AUTO)]; },
  suggest_only: () => { CLUSTERS = [cl('SPEAKER_00', '', M_SUGGEST)]; },
  /* 计划书里过时的 0.55：0.58 会被旧数字判成「可预填」，按库里现行阈值
     （auto=0.65）它只是候选 —— 面板必须用后端给的 band，不许自己写一套阈值。 */
  stale_threshold: () => {
    CLUSTERS = [cl('SPEAKER_00', '', { work: WORK, page: RUDEUS, score: 0.58,
                                       band: 'suggest', auto: null })];
  },
  named_kept: () => { CLUSTERS = [cl('SPEAKER_00', ALICE, null)]; },
  one_to_one: () => {
    CLUSTERS = [cl('SPEAKER_00', '', M_AUTO), cl('SPEAKER_01', '', M_AUTO_LOST)];
  },
  merge_hint: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  hostile_names: () => {
    CLUSTERS = [cl('SPEAKER_00', '', { work: HOSTILE_WORK, page: HOSTILE, score: 0.91,
                                       band: 'auto', auto: HOSTILE })];
  },
  rename_prompt: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  rename_cancel: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  /* 空名字（只有空白）与「输入的名字与原名相同」都必须拦在发请求之前 ——
     取消（null）走的就是前一条判据，因为 `(null||'')` 也是空串。 */
  rename_blank: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  rename_unchanged: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  forget_confirmed: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  forget_cancelled: () => { CLUSTERS = [cl('SPEAKER_00', '', null)]; },
  /* 体检：数据不够（四个字段都是 null）— 面板必须说「样本还不够」，
     绝不能渲染成 0.000 / 0 这种看着像「算过了」的分数。 */
  health_thin: () => { CLUSTERS = [cl('SPEAKER_00', '', M_AUTO)]; },
  /* 体检接口自己坏了（HTTP 500）：这一节要显示错误，页面别处照常。 */
  health_down: () => { CLUSTERS = [cl('SPEAKER_00', '', M_AUTO)]; },
};

if (!SCENARIOS[scenario]) emit({ ok: false, fatal: '未知场景：' + scenario });
SCENARIOS[scenario]();

if (scenario === 'hostile_names') {
  const reg = registry();
  // 两份用户数据都做成 payload：角色名（页）与**作品名**（两行作品）
  reg.works[0].work = HOSTILE_WORK;
  reg.works[0].pages[0].page = HOSTILE;
  reg.works[1].work = HOSTILE_WORK2;
  reg.works[1].pages[0].page = `"><script>alert(3)</script>`;
  globalThis.__HOSTILE_REG = reg;
  SPK_WORK = HOSTILE_WORK;     // 本集的作品名也是同一个 payload
}

// --------------------------------------------------------------------------
// fetch 桩
// --------------------------------------------------------------------------
const fetchCalls = [];
async function fakeFetch(url, opts) {
  const u = String(url);
  const method = (opts && opts.method) || 'GET';
  const call = { url: u, method, body: (opts && opts.body) || null };
  fetchCalls.push(call);
  let body = {};
  let status = 200;
  if (u.startsWith('/api/models')) {
    body = { asr: [], separation: [], diarize: [], default_output: '',
             sep_output: '', hint: '', separation_ready: true, diarize_ready: true };
  } else if (u.startsWith('/api/jobs')) {
    body = { jobs: [] };
  } else if (u.startsWith('/api/resumable')) {
    body = { items: [] };
  } else if (u.startsWith('/api/workdirs')) {
    body = { items: [{ work: 'EP01', work_name: WORK, named: 0 }] };
  } else if (u.startsWith('/api/speakers?')) {
    body = speakers();
  } else if (u.startsWith('/api/speakers/assign')) {
    body = { ok: true };
  } else if (u.startsWith('/api/registry/rename')) {
    body = { ok: true, work: WORK, from: 'from', to: 'to', moved: 3, merged: false };
  } else if (u.startsWith('/api/registry/forget')) {
    body = { ok: true, work: WORK, page: 'p', removed: 3 };
  } else if (u.startsWith('/api/registry/health')) {
    /* 必须排在 `/api/registry` 那条前面：健康接口的路径是它的前缀延伸，
       反过来的话这一支永远进不来（体检就永远测的是「库列表」那份数据）。 */
    if (scenario === 'health_down') {
      return { ok: false, status: 500,
               json: async () => ({ error: '【探针】体检接口炸了' }),
               text: async () => '' };
    }
    body = healthBody();
  } else if (u.startsWith('/api/registry')) {
    body = globalThis.__HOSTILE_REG || registry();
  } else if (u.startsWith('/api/result')) {
    body = {};
  }
  return {
    ok: status < 400, status,
    json: async () => body,
    text: async () => '',
  };
}
globalThis.fetch = fakeFetch;

// --------------------------------------------------------------------------
// 观察工具
// --------------------------------------------------------------------------
function findAll(root, pred, out = []) {
  if (pred(root)) out.push(root);
  for (const c of (root.__children || [])) findAll(c, pred, out);
  return out;
}
function clusterRow(i) {
  const spk = document.getElementById('spkList').__children[i];
  const input = findAll(spk, (e) => e.tagName === 'input'
                                  && e.className === 'nm')[0] || null;
  const sel = findAll(spk, (e) => e.tagName === 'select')[0] || null;
  const sg = findAll(spk, (e) => e.className === 'sg')[0] || null;
  const mg = findAll(spk, (e) => e.className === 'mg')[0] || null;
  return { root: spk, input, sel, sg, mg };
}
function options(sel) {
  return sel ? (sel.__children || []).map((o) => ({ value: o.value,
                                                    text: o.textContent })) : [];
}
function regRows() {
  const box = document.getElementById('spkReg');
  return findAll(box, (e) => e.className === 'regr').map((r) => ({
    text: (r.__children || []).map((c) => c.textContent),
    buttons: (r.__children || []).filter((c) => c.tagName === 'button')
      .map((b) => ({ text: b.textContent, el: b })),
  }));
}
function clickRegButton(rowIndex, label) {
  const rows = regRows();
  const row = rows[rowIndex];
  if (!row) return false;
  const btn = row.buttons.find((b) => b.text === label);
  if (!btn) return false;
  btn.el.click();
  return true;
}
function textOf(el) { return el ? String(el.textContent || '') : null; }
function dump(el) {
  return { tag: el.tagName, cls: el.className || '',
           text: textOf(el), value: (el.value === undefined ? null : el.value),
           attrs: Object.assign({}, el.__attrs),
           options: (el.tagName === 'select' || el.tagName === 'datalist')
             ? (el.__children || []).map((o) => ({ value: o.value, text: o.textContent }))
             : null,
           children: (el.__children || []).map(dump) };
}

// 把内部函数暴露给探针（只读观察 + 触发真实路径）
code += `
;globalThis.__probe = {
  openSpeakersByWork, openSpeakers, reloadSpeakers, renderSpeakers, renderRegistry,
  loadRegistry, fillNameDatalist, renameRegistryPage, forgetRegistryPage,
  mergeHintText, regPageInfo,
  setSpk: (v) => { spk = v; },
  getSpk: () => spk,
  getReg: () => spkReg,
};`;

const ctx = vm.createContext(globalThis);
vm.runInContext(code, ctx, { filename: 'ui.html' });

const P = globalThis.__probe;
if (!P) emit({ ok: false, fatal: 'ui.html 脚本没有跑起来（__probe 未定义）' });

globalThis.window.prompt = (...a) => {
  promptCalls.push(a.map(String));
  if (scenario === 'rename_cancel') return null;      // 用户按了取消
  if (scenario === 'rename_blank') return '   ';      // 只输入了空白
  if (scenario === 'rename_unchanged') return ALICE;  // 输入的名字与原名相同
  return scenario === 'rename_prompt' ? '新名字' : '';
};
globalThis.window.confirm = (...a) => {
  confirmCalls.push(a.map(String));
  return scenario === 'forget_confirmed' || scenario === 'hostile_names';
};

const facts = { ok: true, scenario, work: WORK, alice: ALICE, rudeus: RUDEUS,
                stranger: STRANGER, hostile: HOSTILE,
                hostileWork: HOSTILE_WORK, hostileWork2: HOSTILE_WORK2,
                alertCalls };

async function waitFor(pred, ms = 3000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if (pred()) return true; } catch (e) { /* 还没就绪 */ }
    await new Promise((r) => realSetTimeout(r, 5));
  }
  try { return pred(); } catch (e) { return false; }
}

try {
  await P.openSpeakersByWork('EP01');
  await waitFor(() => document.getElementById('spkNames').__children.length > 0);
  facts.rowCount = document.getElementById('spkList').__children.length;
  facts.registryText = textOf(document.getElementById('spkReg'));
  facts.datalist = (document.getElementById('spkNames').__children || [])
    .map((o) => o.value);

  // 「下一步」提示 + 跳转按钮：这两样是给「标注完不知道该干嘛」兜底的，
  // 光看源码子串看不出按钮点了到底滚不滚，所以这里真点一下并记录调用。
  // 两个坑：① 假 DOM 的 `textContent` 不聚合子节点，容器要 dump 成结构再递归取字；
  // ② 脚本里 `el.id = '...'` 不会注册进 byId（那张表只认 getElementById 自己造的），
  //   所以按钮必须**从子节点里找**，不能用 getElementById。
  const nextBox = document.getElementById('spkNext');
  facts.nextDump = nextBox ? dump(nextBox) : null;
  const goBtn = ((nextBox && nextBox.__children) || [])
    .find((c) => c.tagName === 'button') || null;
  facts.nextButton = goBtn ? { label: goBtn.textContent, tag: goBtn.tagName } : null;
  const exportBox = document.getElementById('spkExport');
  let exportScrolled = 0;
  if (exportBox) exportBox.scrollIntoView = () => { exportScrolled += 1; };
  if (goBtn) goBtn.click();
  facts.nextScrolledToExport = exportScrolled;

  if (scenario === 'merge_hint') {
    const r = clusterRow(0);
    const seen = [];
    const probeName = (v) => {
      r.input.value = v;
      r.input.fire('input');
      seen.push({ value: v, sg: textOf(r.sg), mg: textOf(r.mg) });
    };
    probeName(ALICE);          // 本作品已有 → 并入
    probeName(RUDEUS);         // 本作品已有（但向量的模型不同）
    probeName(STRANGER);       // 别的作品同名 → 新建，且要说明
    probeName('新人');          // 全新 → 新建
    facts.hints = seen;
    // 作品名清空后：不会入库，提示要说清楚
    document.getElementById('spkWork').value = '';
    document.getElementById('spkWork').fire('input');
    facts.hintNoWork = textOf(r.mg);
    facts.datalistNoWork = (document.getElementById('spkNames').__children || [])
      .map((o) => o.value);
    document.getElementById('spkWork').value = WORK;
    document.getElementById('spkWork').fire('input');
    facts.hintAfterRewrite = textOf(r.mg);
    facts.datalistAfterWorkChange = (document.getElementById('spkNames')
      .__children || []).map((o) => o.value);
  } else if (scenario === 'hostile_names') {
    const r = clusterRow(0);
    facts.inputValue = r.input.value;
    facts.inputIsSuggestion = textOf(r.sg);
    facts.mergeHint = textOf(r.mg);
    facts.spkWorkValue = document.getElementById('spkWork').value;
    facts.spkListDump = dump(document.getElementById('spkList'));
    facts.spkRegDump = dump(document.getElementById('spkReg'));
    facts.spkHealthDump = dump(document.getElementById('spkHealth'));
    facts.workHeaders = findAll(document.getElementById('spkReg'),
                                (e) => e.className === 'regwh')
      .map((e) => textOf(e));
    facts.datalist = (document.getElementById('spkNames').__children || [])
      .map((o) => o.value);
    facts.regRows = regRows().map((r) => ({ text: r.text,
                                            buttons: r.buttons.map((b) => b.text) }));
    facts.clicked = clickRegButton(0, '删除');
    await waitFor(() => fetchCalls.some((c) => c.url.startsWith('/api/registry/forget')));
    facts.registryCalls = fetchCalls.filter((c) => c.url.startsWith('/api/registry'));
  } else if (scenario.startsWith('rename_')) {
    facts.clicked = clickRegButton(0, '改名');
    await waitFor(() => fetchCalls.some((c) => c.url.startsWith('/api/registry/rename'))
                     || promptCalls.length > 0);
    await new Promise((r) => realSetTimeout(r, 30));
    facts.promptCalls = promptCalls;
    facts.registryCalls = fetchCalls.filter((c) => c.url.startsWith('/api/registry'));
  } else if (scenario.startsWith('forget_')) {
    facts.clicked = clickRegButton(0, '删除');
    await waitFor(() => confirmCalls.length > 0);
    await new Promise((r) => realSetTimeout(r, 30));
    facts.confirmCalls = confirmCalls;
    facts.registryCalls = fetchCalls.filter((c) => c.url.startsWith('/api/registry'));
  } else {
    const rows = [];
    for (let i = 0; i < document.getElementById('spkList').__children.length; i++) {
      const r = clusterRow(i);
      rows.push({
        inputValue: r.input.value,
        inputListAttr: r.input.getAttribute('list'),
        selectOptions: options(r.sel),
        suggestText: textOf(r.sg),
        mergeText: textOf(r.mg),
      });
    }
    facts.rows = rows;
    facts.spkListDump = dump(document.getElementById('spkList'));
    facts.spkRegDump = dump(document.getElementById('spkReg'));
    facts.spkHealthDump = dump(document.getElementById('spkHealth'));
    facts.regRows = regRows().map((r) => ({ text: r.text,
                                            buttons: r.buttons.map((b) => b.text) }));
    facts.innerHtmlWrites = innerHtmlWrites.slice();
    facts.registryCalls = fetchCalls.filter((c) => c.url.startsWith('/api/registry'));
    facts.healthCalls = fetchCalls.filter(
      (c) => c.url.startsWith('/api/registry/health'));

    if (scenario === 'prefill_auto') {
      // 用户把预填的名字改掉：提示必须跟着变（还说「库建议：X」就是假话）
      const r = clusterRow(0);
      r.input.value = '别人';
      r.input.fire('input');
      facts.suggestAfterEdit = textOf(r.sg);
      facts.mergeAfterEdit = textOf(r.mg);
      r.input.value = ALICE;
      r.input.fire('input');
      facts.suggestAfterRestore = textOf(r.sg);
    }
    if (scenario === 'suggest_only') {
      // 候选必须**能用**：在下拉里选中它 = 并入那个角色
      const r = clusterRow(0);
      const opt = (r.sel.__children || []).find((o) => o.value === RUDEUS);
      facts.candidateOptionFound = !!opt;
      r.sel.value = RUDEUS;
      await r.sel.onchange();
      await waitFor(() => fetchCalls.some(
        (c) => c.url.startsWith('/api/speakers/assign')));
      const call = fetchCalls.find((c) => c.url.startsWith('/api/speakers/assign'));
      facts.assignCall = call || null;
    }
  }
} catch (err) {
  facts.ok = false;
  facts.error = String((err && err.stack) || err);
}
facts.innerHtmlWrites = facts.innerHtmlWrites || innerHtmlWrites.slice();
facts.alertCalls = alertCalls;
emit(facts);
