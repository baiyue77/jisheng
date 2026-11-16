"""Regression tests for review findings; inference and media tools are stubbed."""
import contextlib
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Thread
import types
import subprocess
import unittest
from unittest.mock import patch
from urllib.parse import quote
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jisheng as A
import diarize_worker
import qwen_worker as Q
import webui as W


class OpenFolderRegressionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.server = W.ThreadingHTTPServer(("127.0.0.1", 0), W.Handler)
        thread = Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        launcher = patch.object(W.os, "startfile", create=True)
        self.launch = launcher.start()
        self.addCleanup(launcher.stop)

    def request(self, path, **headers):
        return self.request_route("GET", "/api/openfolder?path=" + quote(str(path)),
                                  headers=headers)

    def request_route(self, method, path, *, body=None, headers=None):
        conn = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            response.read()
            return response.status
        finally:
            conn.close()

    def test_existing_directory_opens_in_explorer(self):
        self.assertEqual(self.request(self.folder), 200)
        self.launch.assert_called_once_with(str(self.folder.resolve()), "explore")

    def test_regular_file_is_never_launched(self):
        file = self.folder / "sample.txt"
        file.write_text("sample", encoding="utf-8")
        self.assertEqual(self.request(file), 400)
        self.launch.assert_not_called()

    def test_external_request_is_rejected_before_opening(self):
        for headers in ({"Origin": "https://example.com"},
                        {"Sec-Fetch-Site": "cross-site"},
                        {"Host": "example.com"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(self.folder, **headers), 403)
        self.launch.assert_not_called()

    def test_local_job_api_rejects_foreign_host(self):
        self.assertEqual(self.request_route("GET", "/api/jobs"), 200)
        self.assertEqual(self.request_route("GET", "/api/jobs",
                                            headers={"Host": "example.com"}), 403)

    def test_foreign_origin_cannot_mutate_jobs(self):
        job = {"id": "check-origin", "status": "run", "stop": False, "logs": []}
        body = json.dumps({"id": job["id"]})
        headers = {"Origin": "https://example.com", "Content-Type": "text/plain"}
        with patch.dict(W.JOBS, {job["id"]: job}):
            self.assertEqual(self.request_route("POST", "/api/stop", body=body,
                                                headers=headers), 403)
            self.assertFalse(job["stop"])


class ModelDownloadPrivacyTests(unittest.TestCase):
    def test_authenticated_diarization_uses_hugging_face(self):
        with (patch.dict(os.environ, {"HF_HOME": "unused", "HF_ENDPOINT": "https://example.com",
                                      "HF_TOKEN": "placeholder"}),
              contextlib.redirect_stderr(io.StringIO())):
            diarize_worker._configure_hf_cache()
            self.assertEqual(os.environ.get("HF_ENDPOINT"), "https://huggingface.co")

    def test_public_model_download_disables_authentication(self):
        with tempfile.TemporaryDirectory() as folder:
            for filename in A.REQUIRED_MODEL_FILES:
                (Path(folder) / filename).touch()
            with (patch.object(A, "LOCAL_MODEL_DIR", Path(folder)),
                  patch("huggingface_hub.snapshot_download") as download,
                  patch.dict(os.environ, {"HF_TOKEN": "placeholder"}),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(A.download_model("https://example.com"), 0)
                self.assertIs(download.call_args.kwargs.get("token"), False)


class LauncherRegressionTests(unittest.TestCase):
    def test_check_reports_missing_ffprobe(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(A, "DEFAULT_HOME", Path(folder)),
                  patch.object(A, "register_cuda_dlls", return_value=[]),
                  patch.object(A.shutil, "which",
                               side_effect=lambda name: "ffmpeg.exe" if name == "ffmpeg" else None),
                  patch.object(sys, "argv", ["jisheng.py", "--check"]),
                  contextlib.redirect_stdout(io.StringIO()) as report):
                self.assertEqual(A.main(), 2)
            self.assertRegex(report.getvalue(), r"ffprobe\s+: 未找到")

    @unittest.skipUnless(os.name == 'nt', 'Windows launcher')
    def test_cli_help_uses_the_project_environment(self):
        launcher = Path(__file__).resolve().parent.parent / 'jisheng.cmd'
        result = subprocess.run(['cmd.exe', '/d', '/c', str(launcher), '--help'],
                                input='\n', text=True, encoding='utf-8', errors='replace',
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--no-vad', result.stdout)


class CheckpointRegressionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.media = self.root / 'EP01.mkv'
        self.media.write_bytes(b'original')
        self.calls = []
        self.job = dict(id='review', media=str(self.media), opts={}, logs=[],
                        out_dir=str(self.root / 'out'), status='run')
        self.extract = unittest.mock.Mock(side_effect=self.extract_audio)
        for p in (
            patch.object(W, 'WORK_DIR', self.root / 'work'),
            patch.object(W.shutil, 'which', return_value='ffmpeg'),
            patch.object(A, 'extract_audio', self.extract),
            patch.object(A, 'read_wav_float32', return_value=np.zeros(48000)),
            patch.object(A, 'vad_chunks', return_value=[
                dict(start=i * 16000, end=(i + 1) * 16000) for i in range(3)]),
            patch.object(W, '_cuda_ok', return_value=False),
            patch.object(W, '_get_model', return_value=None),
            patch.object(A, 'transcribe_chunk', side_effect=self.transcribe),
        ):
            p.start()
            self.addCleanup(p.stop)

    def extract_audio(self, media, dest):
        dest.write_bytes(b'fake wav')
        return 3

    def transcribe(self, model, audio, chunk, *args):
        i = chunk['start'] // 16000
        self.calls.append(i)
        if i == 0:
            return None
        if self.interrupt and i == 1:
            self.job['stop'] = True
        return dict(start=i, end=i + 1, text=f'line {i}')

    def test_resume_after_empty_chunk_does_not_repeat_subtitle(self):
        self.interrupt = True
        W.run_job(self.job)
        self.assertEqual(self.job['status'], 'stopped')
        self.interrupt = False
        self.job['stop'] = False
        self.calls.clear()
        W.run_job(self.job)
        self.assertEqual(self.calls, [2])
        self.assertEqual([s['start'] for s in W.load_segments(Path(self.job['work']))], [1, 2])

    def test_same_size_source_replacement_invalidates_audio(self):
        self.interrupt = False
        W.run_job(self.job)
        old = self.media.stat()
        self.media.write_bytes(b'replaced')
        os.utime(self.media, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))
        W.run_job(self.job)
        self.assertEqual(self.extract.call_count, 2)

    def test_webui_vad_switch_bypasses_detection(self):
        self.interrupt = False
        self.job['opts']['vad'] = False
        with patch.object(A, 'vad_chunks', return_value=[]) as vad:
            W.run_job(self.job)
        self.assertFalse(vad.called)
        self.assertEqual(self.calls, [0])

    def test_vad_toggle_invalidates_checkpoint_identity(self):
        self.assertNotEqual(W._opts_key({'vad': True}), W._opts_key({'vad': False}))

    def test_legacy_checkpoint_with_empty_chunks_resumes_by_time(self):
        self.interrupt = True
        W.run_job(self.job)
        work = Path(self.job['work'])
        import json
        state = W.load_state(work)
        state.pop('next_chunk', None)
        W.save_state(work, state)
        segments = W.load_segments(work)
        for segment in segments:
            segment.pop('chunk_index', None)
        (work / 'segments.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in segments), encoding='utf-8')
        self.interrupt = False
        self.job['stop'] = False
        self.calls.clear()
        W.run_job(self.job)
        self.assertEqual(self.calls, [2])

    def test_durable_segment_recovers_cursor_after_state_write_interruption(self):
        self.interrupt = True
        W.run_job(self.job)
        work = Path(self.job['work'])
        state = W.load_state(work)
        state['next_chunk'] = 1  # state still behind the fsynced subtitle row
        W.save_state(work, state)
        self.interrupt = False
        self.job['stop'] = False
        self.calls.clear()
        W.run_job(self.job)
        self.assertEqual(self.calls, [2])

    def test_pre_upgrade_options_key_keeps_existing_checkpoint(self):
        import json
        self.interrupt = True
        W.run_job(self.job)
        work = Path(self.job['work'])
        state = W.load_state(work)
        old_key = json.loads(state['opts_key'])
        old_key.pop('vad')
        state['opts_key'] = json.dumps(old_key, ensure_ascii=False, sort_keys=True)
        W.save_state(work, state)
        self.interrupt = False
        self.job['stop'] = False
        self.calls.clear()
        W.run_job(self.job)
        self.assertEqual(self.extract.call_count, 1, 'upgrade must not erase old checkpoints')
        self.assertEqual(self.calls, [2])


class DownloadRegressionTests(unittest.TestCase):
    def test_unicode_filename_has_valid_http_headers(self):
        with tempfile.TemporaryDirectory() as td:
            name = '日语字幕.ja.srt'
            dest = Path(td) / name
            dest.write_text('subtitle', encoding='utf-8')
            handler = object.__new__(W.Handler)
            handler.headers = {"Host": "127.0.0.1:8760"}
            handler.server = types.SimpleNamespace(server_address=("127.0.0.1", 8760))
            handler.path = '/api/download?id=review&name=' + quote(name)
            handler.request_version = 'HTTP/1.1'
            handler.command = 'GET'
            handler.wfile = io.BytesIO()
            handler.send_response = lambda *args: None
            with patch.dict(W.JOBS, review={'files': [dict(name=name, path=str(dest))]}):
                try:
                    handler.do_GET()
                except UnicodeEncodeError as exc:
                    self.fail(f'Unicode download raised {exc}')
            response = handler.wfile.getvalue()
            self.assertIn(b"filename*=UTF-8''" + quote(name).encode('ascii'), response)
            self.assertTrue(response.endswith(b'subtitle'))


class VadRegressionTests(unittest.TestCase):
    def test_no_vad_transcribes_the_full_audio_without_calling_vad(self):
        args = types.SimpleNamespace(vad_min_silence=500, beam=5, no_vad=True)
        model = types.SimpleNamespace(transcribe=lambda *a, **kw: (
            [types.SimpleNamespace(text='hello')], None))
        with patch.object(A, 'vad_chunks', return_value=[]) as vad:
            segments = A.vad_chunked_transcribe(model, np.zeros(32000), args, [])
        self.assertFalse(vad.called, '--no-vad must bypass speech detection')
        self.assertEqual(segments, [dict(start=0, end=2, text='hello')])


class QwenCacheRegressionTests(unittest.TestCase):
    def test_replaced_same_name_audio_is_fed_to_inference(self):
        heard = []

        class Inputs(dict):
            def to(self, *args):
                return self

        class Processor:
            @classmethod
            def from_pretrained(cls, *args, **kw):
                return cls()

            def apply_transcription_request(self, **kw):
                with wave.open(kw['audio'][0]) as stream:
                    heard.append(int(np.frombuffer(stream.readframes(1), dtype=np.int16)[0]))
                return Inputs(input_ids=np.zeros((1, 1), dtype=int))

            def decode(self, *args, **kw):
                return ['stub']

        class Model:
            device = 'cpu'
            dtype = None

            @classmethod
            def from_pretrained(cls, *args, **kw):
                return cls()

            def eval(self):
                pass

            def generate(self, **kw):
                return np.zeros((1, 2), dtype=int)

        torch = types.SimpleNamespace(bfloat16=None, no_grad=contextlib.nullcontext)
        transformers = types.SimpleNamespace(AutoProcessor=Processor, AutoModelForMultimodalLM=Model)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / 'src'
            src.mkdir()
            wav = src / '001.wav'
            argv = ['qwen_worker', str(src), '--out', str(root / 'out.jsonl'), '--tmp', str(root / 'cache')]
            with patch.dict(sys.modules, torch=torch, transformers=transformers), patch.object(sys, 'argv', argv):
                for value in (100, 900):
                    with wave.open(str(wav), 'wb') as stream:
                        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                        stream.writeframes(np.full(1600, value, dtype=np.int16).tobytes())
                    st = wav.stat()
                    os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + value * 1_000_000))
                    with contextlib.redirect_stdout(io.StringIO()):
                        Q.main()
            self.assertEqual(heard, [100, 900])


if __name__ == '__main__':
    unittest.main()
