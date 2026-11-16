import ast
import json
import io
from urllib.parse import quote
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stage_jobs
import webui as W


class LanguagesTests(unittest.TestCase):
    def test_all_local_processor_languages_are_available(self):
        import asr_languages
        source = Path('venv-diarize/Lib/site-packages/transformers/models/qwen3_asr/processing_qwen3_asr.py')
        if not source.exists():
            self.skipTest('local optional Transformers package not installed')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        expected = next(ast.literal_eval(n.value) for n in tree.body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='LANGUAGE_CODE_TO_NAME' for t in n.targets))
        self.assertEqual(set(asr_languages.LANGUAGES), set(expected))
        for language in [*expected, 'auto']:
            with self.subTest(language=language):
                self.assertEqual(stage_jobs.validate(dict(backend='qwen', language=language))[2], language)

    def test_unknown_language_does_not_fall_back_to_japanese(self):
        with self.assertRaises(ValueError):
            stage_jobs.validate(dict(backend='qwen', language='not-a-language'))


class DownloadTests(unittest.TestCase):
    def test_video_http_download_streams_without_loading_the_whole_file(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'video.mp4';dest.write_bytes(b'video payload')
            handler=object.__new__(W.Handler)
            handler.headers = {"Host": "127.0.0.1:8760"}
            handler.server = SimpleNamespace(server_address=("127.0.0.1", 8760))
            handler.path='/api/download?id=stream-test&name=video.mp4'
            handler.request_version='HTTP/1.1';handler.command='GET'
            handler.wfile=io.BytesIO();handler.send_response=lambda *args:None
            with patch.dict(W.JOBS, {'stream-test':dict(files=[dict(name=dest.name,path=str(dest))])}), \
                 patch.object(Path,'read_bytes',side_effect=AssertionError('large videos must stream')):
                handler.do_GET()
            self.assertTrue(handler.wfile.getvalue().endswith(b'video payload'))

    def test_share_text_extracts_the_url_without_chinese_punctuation(self):
        import download_jobs as D
        self.assertEqual(D.extract_url('看看这个视频 https://www.bilibili.com/video/BV123/，复制打开！'), 'https://www.bilibili.com/video/BV123/')

    def test_local_or_credential_urls_are_rejected(self):
        import download_jobs as D
        for value in ['file:///C:/secret', 'http://127.0.0.1/video', 'http://[::1]/video',
                      'http://192.168.1.1/video', 'http://user:pass@example.com/video',
                      'http://localhost/video', '--exec calc']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                D.extract_url(value)

    def test_command_uses_one_video_and_ignores_global_configuration(self):
        import download_jobs as D
        with tempfile.TemporaryDirectory() as td:
            cmd = D.build_command(['yt-dlp'], 'https://example.com/watch?v=1', Path(td), Path(td)/'result.jsonl')
        self.assertIn('--ignore-config', cmd)
        self.assertIn('--no-plugin-dirs', cmd)
        self.assertIn('--no-playlist', cmd)
        self.assertIn('--max-downloads', cmd)
        self.assertEqual(cmd[-2:], ['--', 'https://example.com/watch?v=1'])

    def test_result_cannot_escape_the_selected_download_directory(self):
        import download_jobs as D
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'root';root.mkdir()
            outside = Path(td)/'outside.mp4'; outside.write_bytes(b'video')
            manifest = root/'result.jsonl';manifest.write_text(json.dumps(str(outside))+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):
                D.read_outputs(manifest, root)

    def test_abandon_download_does_not_remove_asr_work(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)/'episode'; work.mkdir()
            with patch.object(W, 'WORK_DIR', Path(td)):
                W._discard_job(dict(id='download', kind='download', media='episode.mp4', work=str(work)))
            self.assertTrue(work.exists())

    def test_download_is_complete_only_after_a_real_output_is_registered(self):
        import download_jobs as D
        with tempfile.TemporaryDirectory() as td:
            job=dict(id='test',name='example.com',url='https://example.com/video',
                     out_dir=td,status='run',logs=[],files=[],stop=False)
            def fake_popen(cmd, **kwargs):
                output=Path(td)/'video.mp4';output.write_bytes(b'video')
                manifest=Path(cmd[cmd.index('--print-to-file')+2])
                manifest.write_text(json.dumps(str(output))+'\n',encoding='utf-8')
                class Process:
                    stdout=io.StringIO(D.TITLE+'"Test video"\n'+D.PROGRESS+'{"downloaded_bytes":50,"total_bytes":100,"eta":1}\n')
                    stderr=io.StringIO('')
                    returncode=0
                    def poll(self): return 0
                return Process()
            with patch.object(D,'validate_public_host'), patch.object(D,'downloader_command',return_value=['fake']), \
                 patch.object(D.subprocess,'Popen',side_effect=fake_popen), \
                 patch.object(W,'_get_model',side_effect=AssertionError('download must not invoke ASR')):
                D.run(W,job)
            self.assertEqual(job['status'],'done')
            self.assertEqual(job['result']['title'],'Test video')
            self.assertEqual(job['result']['media_paths'],[str(Path(td)/'video.mp4')])

    def test_stopped_download_does_not_launch_or_resolve_a_url(self):
        import download_jobs as D
        job=dict(id='test',stop=True,logs=[],status='run')
        with patch.object(D,'validate_public_host') as resolve, patch.object(D.subprocess,'Popen') as spawn:
            D.run(W,job)
        self.assertEqual(job['status'],'stopped')
        resolve.assert_not_called();spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
