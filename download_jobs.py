"""Share-link downloads through the installed open-source yt-dlp CLI."""
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

PROGRESS = '__ASR_PROGRESS__'
TITLE = '__ASR_TITLE__'


def downloader_command():
    if importlib.util.find_spec('yt_dlp') is not None:
        return [sys.executable, '-u', '-m', 'yt_dlp']
    executable = shutil.which('yt-dlp')
    return [executable] if executable else None


def status():
    ready = bool(downloader_command())
    return ready, ('使用 yt-dlp 解析平台链接；部分视频需要登录或受平台限制'
                   if ready else '未找到 yt-dlp；请在项目环境安装 requirements-download.txt')


def extract_url(text):
    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError('请粘贴视频链接或分享文案（不超过 16384 字符）')
    match = re.search(r'https?://[^\s<>"\u3000，。；！？、）】》〉」』]+', text, re.I)
    if not match:
        raise ValueError('分享内容中没有 HTTP(S) 视频链接')
    url = match.group(0).rstrip(".,;!?)]}，。；！？，）】》〉」』、'\"")
    parts = urlsplit(url)
    host = parts.hostname
    try:
        parts.port
    except ValueError as exc:
        raise ValueError('链接端口不合法') from exc
    if not host or parts.username or parts.password:
        raise ValueError('链接地址不合法，不支持带账号密码的地址')
    if host.casefold() == 'localhost' or host.casefold().endswith(('.localhost', '.local')):
        raise ValueError('请使用公开视频平台链接，不支持本机或局域网地址')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and not address.is_global:
        raise ValueError('不支持本机、局域网或保留地址')
    return url


def validate_public_host(url):
    parts = urlsplit(url)
    try:
        addresses = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme=='https' else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError('无法解析链接域名：' + str(exc)) from exc
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError('视频链接不能指向本机、局域网或保留地址')


def build_command(prefix, url, root, manifest, cookies_file=''):
    cmd = list(prefix) + [
        '--ignore-config', '--no-plugin-dirs', '--no-remote-components',
        '--no-playlist', '--playlist-items', '1', '--max-downloads', '1',
        '--match-filters', '!is_live & !is_upcoming',
        '--socket-timeout', '20', '--retries', '2', '--fragment-retries', '2',
        '--windows-filenames', '--trim-filenames', '160',
        '--newline', '--progress', '--progress-delta', '.5', '--no-colors',
        '--progress-template', 'download:' + PROGRESS + '%(progress)j',
        '--print', 'before_dl:' + TITLE + '%(title)j', '--no-simulate',
        '--print-to-file', 'after_move:%(filepath)j', str(manifest),
        '-P', str(root), '-o', '%(title).120B [%(id)s].%(ext)s',
        '-f', 'bv*[height<=1080]+ba/b[height<=1080]/b', '--merge-output-format', 'mp4',
    ]
    if shutil.which('node'):
        cmd += ['--js-runtimes', 'node']
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        cmd += ['--ffmpeg-location', str(Path(ffmpeg).parent)]
    if cookies_file:
        cmd += ['--cookies', str(cookies_file)]
    return cmd + ['--', url]


def read_outputs(manifest, root):
    root = Path(root).resolve()
    result = []
    if not Path(manifest).exists():
        return result
    for line in Path(manifest).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, str):
            raise ValueError('下载器返回的文件路径格式不合法')
        path = Path(value).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError('下载文件越出了所选目录')
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('下载未完成或文件为空')
        if path not in result:
            result.append(path)
    return result


class ProcessTree:
    """The stop endpoint must also stop ffmpeg children on Windows."""
    def __init__(self, process):
        self.process = process

    def poll(self):
        return self.process.poll()

    def wait(self, **kwargs):
        return self.process.wait(**kwargs)

    def terminate(self):
        if os.name == 'nt' and self.poll() is None:
            try:
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               capture_output=True, timeout=10,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if self.poll() is None:
                self.process.kill()
        elif self.poll() is None:
            self.process.terminate()

    kill = terminate


def enqueue(w, payload):
    try:
        if not isinstance(payload, dict):
            raise ValueError('请求体必须是对象')
        url = extract_url(payload.get('text'))
        ready, note = status()
        if not ready:
            return {'error': note}, 503
        raw = payload.get('output_dir') or ''
        if not isinstance(raw, str):
            raise ValueError('输出目录必须是文本路径')
        base = Path(raw.strip()) if raw.strip() else w.OUTPUT_ROOT
        if base.exists() and not base.is_dir():
            raise ValueError('输出路径不是文件夹')
        cookies = payload.get('cookies_file') or ''
        if not isinstance(cookies, str) or (cookies and not Path(cookies).is_file()):
            raise ValueError('Cookies 文件不存在，请选择本地 cookies.txt')
        ident = uuid.uuid4().hex[:12]
        job = dict(id=ident, kind='download', name=urlsplit(url).hostname,
                   media='', url=url, cookies_file=str(Path(cookies).resolve()) if cookies else '', status='wait', stage='解析链接',
                   out_dir=str(base / '下载' / ident), progress=0., eta=None,
                   logs=['分享链接已入队，下载后可执行所选功能。'], files=[], error='',
                   stop=False, abandon=False, opts={}, result=None)
        with w.JOBS_LOCK:
            w.JOBS[ident] = job
        w.TASK_QUEUE.put(job)
        return dict(id=ident, status='wait', kind='download'), 200
    except (ValueError, TypeError, OSError) as exc:
        return {'error': str(exc)}, 400


def run(w, job):
    def cancelled():
        return w.stop_requested(job) or w.abandon_requested(job)
    def finish_cancelled():
        w._set(job, status='abandoned' if w.abandon_requested(job) else 'stopped',
               stage='已放弃' if w.abandon_requested(job) else '已停止', eta=None)
    if cancelled():
        finish_cancelled()
        return
    url = extract_url(job['url'])
    validate_public_host(url)
    prefix = downloader_command()
    if not prefix:
        raise RuntimeError(status()[1])
    root = Path(job['out_dir']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / '.completed.jsonl'
    manifest.unlink(missing_ok=True)
    command = build_command(prefix, url, root, manifest, job.get('cookies_file') or '')
    if cancelled():
        finish_cancelled()
        return
    events = queue.Queue()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                            env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    tree = ProcessTree(proc)
    w._set(job, _proc=tree, stage='解析链接', progress=.01)
    def reader(pipe):
        for line in pipe:
            events.put(line.rstrip())
    readers = [threading.Thread(target=reader, args=(pipe,), daemon=True)
               for pipe in (proc.stdout, proc.stderr)]
    for thread in readers:
        thread.start()
    errors = []
    title = job['name']
    def harvest():
        nonlocal title
        while True:
            try:
                line = events.get_nowait()
            except queue.Empty:
                break
            if line.startswith(PROGRESS):
                try:
                    data = json.loads(line[len(PROGRESS):])
                    total = data.get('total_bytes') or data.get('total_bytes_estimate') or 0
                    fraction = min(.98, (data.get('downloaded_bytes') or 0)/total) if total else .03
                    w._set(job, stage='下载中', progress=max(.02, fraction), eta=data.get('eta'))
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            elif line.startswith(TITLE):
                try:
                    title = str(json.loads(line[len(TITLE):]))
                    w._set(job, name=title)
                except ValueError:
                    pass
            elif line:
                w._emit(job, line)
                if 'ERROR:' in line:
                    errors.append(line)
                if '[Merger]' in line or '[VideoRemuxer]' in line:
                    w._set(job, stage='合并音视频', progress=.98)
    try:
        while proc.poll() is None:
            harvest()
            if w.stop_requested(job) or w.abandon_requested(job):
                tree.terminate()
                break
            time.sleep(.2)
        for thread in readers:
            thread.join(timeout=3)
        harvest()
        if cancelled():
            finish_cancelled()
            return
        if proc.returncode not in (0, 101):
            raise RuntimeError('链接下载失败：' + (' | '.join(errors[-3:]) or f'退出码 {proc.returncode}'))
        outputs = read_outputs(manifest, root)
        if not outputs:
            raise RuntimeError('未下载到视频；该链接可能是直播、需要登录或暂不受支持')
        w._set(job, status='done', stage='完成', progress=1., eta=0.,
               media=str(outputs[0]),
               files=[dict(name=p.name, path=str(p), kind='视频') for p in outputs],
               result=dict(media_paths=[str(p) for p in outputs], title=title))
        w._emit(job, '下载完成，素材已可加入待处理列表。')
    finally:
        if proc.poll() is None:
            tree.terminate()
        for pipe in (proc.stdout, proc.stderr):
            pipe.close()
        w._set(job, _proc=None)
