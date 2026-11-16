import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import webui as W


def wav(path, seconds=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(b'\0\0' * (16000 * seconds))
    return path


class StageJobsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.media = wav(self.root / 'episode.wav')
        self.out = self.root / 'chosen'
        self.job = dict(id='stage-test', media=str(self.media), out_dir=str(self.out),
                        opts=dict(mode='extract'), status='run', logs=[], files=[])
        p = patch.object(W, 'WORK_DIR', self.root / 'work')
        p.start(); self.addCleanup(p.stop)

    def fake_convert(self, src, dest, **kw):
        return wav(dest)

    def test_extract_preserves_same_named_results(self):
        second = wav(self.root / 'other' / self.media.name, seconds=2)
        other_job = dict(self.job, id='other', media=str(second))
        with patch.object(W.ts, 'ffmpeg_convert',
                          side_effect=lambda src, dest, **kw: W.shutil.copyfile(src, dest)):
            W.run_job(self.job)
            first = Path(self.job['files'][0]['path'])
            original = first.read_bytes()
            W.run_job(other_job)
        self.assertNotEqual(first, Path(other_job['files'][0]['path']))
        self.assertEqual(first.read_bytes(), original)

    def test_extract_does_not_load_any_ai_model(self):
        with patch.object(W.ts, 'ffmpeg_convert', side_effect=self.fake_convert), \
             patch.object(W, '_get_model', side_effect=AssertionError('ASR must not run')), \
             patch.object(W.A, 'run_diarization', side_effect=AssertionError('diarization must not run')), \
             patch.object(W, 'separate_vocals', side_effect=AssertionError('separation must not run')):
            W.run_job(self.job)
        self.assertEqual(self.job['status'], 'done')
        self.assertTrue(self.job['files'])
        self.assertTrue(all(Path(f['path']).is_relative_to(self.out) for f in self.job['files']))

    def test_recognize_does_not_run_checked_pipeline_stages(self):
        self.job['opts'] = dict(mode='recognize', backend='whisper', language='ja',
                                separate=True, diarize=True, vad=False)
        fake_model = type('Model', (), {'transcribe': lambda self, *a, **kw:
                         ([type('S', (), {'text': 'テスト'})()], None)})()
        with patch.object(W.ts, 'ffmpeg_convert', side_effect=self.fake_convert), \
             patch.object(W, '_cuda_ok', return_value=False), \
             patch.object(W, '_get_model', return_value=fake_model), \
             patch.object(W.A, 'run_diarization', side_effect=AssertionError('wrong stage')), \
             patch.object(W, 'separate_vocals', side_effect=AssertionError('wrong stage')):
            W.run_job(self.job)
        self.assertEqual(self.job['status'], 'done')
        self.assertTrue((self.out / 'episode.ja.srt').exists())
        self.assertFalse(W._job_view(self.job)['can_label_speakers'])

    def test_invalid_stage_is_rejected(self):
        self.job['opts']['mode'] = 'typo'
        with self.assertRaises(ValueError):
            W.run_job(self.job)

    def test_qwen_chinese_output_stays_in_selected_directory(self):
        import stage_jobs
        self.job['opts'] = dict(mode='recognize', backend='qwen', language='zh', vad=False)
        def fake_qwen(w, job, chunks, work, on_segment):
            for i, chunk in enumerate(chunks):
                on_segment(i, dict(start=chunk['start']/16000, end=chunk['end']/16000, text='中文测试'))
        with patch.object(W.ts, 'ffmpeg_convert', side_effect=self.fake_convert), \
             patch.object(stage_jobs, 'run_qwen', side_effect=fake_qwen):
            W.run_job(self.job)
        self.assertTrue((self.out / 'episode.zh.srt').exists())
        self.assertFalse((self.out / 'episode.ja.srt').exists())

    def test_diarize_does_not_load_asr_and_keeps_new_embeddings(self):
        self.job['opts'] = dict(mode='diarize', backend='qwen', language='zh')
        def diarize(src, args, **kw):
            rows = [dict(start=0, end=1, speaker='SPEAKER_00')]
            Path(kw['out_embeddings']).write_bytes(b'new embeddings')
            Path(kw['out_json']).write_text(json.dumps(rows), encoding='utf-8')
            return rows
        with patch.object(W.ts, 'ffmpeg_convert', side_effect=self.fake_convert), \
             patch.object(W, '_cuda_ok', return_value=False), \
             patch.object(W.A, 'run_diarization', side_effect=diarize), \
             patch.object(W, '_store_cluster_embeddings'), \
             patch.object(W, '_get_model', side_effect=AssertionError('ASR must not run')):
            W.run_job(self.job)
        self.assertTrue((self.out / 'episode.diarization.json').exists())
        self.assertEqual((Path(self.job['work']) / W.EMBEDDINGS_FILE).read_bytes(), b'new embeddings')
        self.assertTrue(W._job_view(self.job)['can_label_speakers'])

    def test_different_stages_reuse_audio_but_not_asr_execution(self):
        self.job['opts'] = dict(mode='recognize', backend='qwen', language='zh', vad=False)
        import stage_jobs
        def fake_qwen(w, job, chunks, work, on_segment):
            for chunk in chunks:
                on_segment(chunk['index'], dict(start=0, end=1, text='测试'))
        with patch.object(W.ts, 'ffmpeg_convert', side_effect=self.fake_convert) as convert, \
             patch.object(stage_jobs, 'run_qwen', side_effect=fake_qwen):
            W.run_job(self.job)
            self.job['opts']['mode'] = 'diarize'
            with patch.object(W, '_cuda_ok', return_value=False), \
                 patch.object(W.A, 'run_diarization', return_value=[dict(start=0,end=1,speaker='SPEAKER_00')]), \
                 patch.object(W, '_store_cluster_embeddings'):
                W.run_job(self.job)
            self.assertEqual(convert.call_count, 1)
        self.assertIn('[说话人', (self.out / 'episode.zh.srt').read_text(encoding='utf-8'))


class FolderPickerTests(unittest.TestCase):
    def test_cancel_preserves_empty_selection(self):
        import folder_picker
        with patch.object(folder_picker.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='', stderr='')):
            self.assertEqual(folder_picker.choose_folder('D:\\chosen'), '')

    def test_selected_unicode_path_round_trips(self):
        import folder_picker
        with patch.object(folder_picker.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='D:\\输出目录', stderr='')):
            self.assertEqual(folder_picker.choose_folder(''), 'D:\\输出目录')


class StageSafetyTests(unittest.TestCase):
    def test_chinese_training_manifest_does_not_get_japanese_language_tag(self):
        import train_export
        self.assertEqual(train_export.list_line('sample.wav', '角色', '你好', language='zh').split('|')[2], 'ZH')

    def test_partial_utf8_result_is_not_decoded_until_complete(self):
        import stage_jobs
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'results.jsonl'
            dest.write_bytes(b'{"index":0}\n' + '{"text":"中'.encode('utf-8')[:-1])
            self.assertEqual(list(stage_jobs.read_results(dest)), [dict(index=0)])

    def test_abandon_old_task_keeps_checkpoint_used_by_running_task(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / 'shared'
            work.mkdir()
            active = dict(id='active', status='run', media='example.wav', work=str(work))
            old = dict(id='old', status='stopped', media='example.wav', work=str(work))
            with patch.object(W, 'WORK_DIR', root), patch.dict(W.JOBS, active=active, clear=True):
                W._discard_job(old)
            self.assertTrue(work.exists())


if __name__ == '__main__':
    unittest.main()
