import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node required for browser behavior probes')
class DownloadUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent.parent
        result=subprocess.run(['node',str(root/'tests/frontend/downloads_probe.mjs'),str(root/'ui.html')],
                              capture_output=True,text=True,encoding='utf-8',timeout=15)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.facts=json.loads(result.stdout)

    def test_backend_languages_are_restored_and_can_be_auto_detected(self):
        self.assertEqual(self.facts['restored'],'en')
        self.assertEqual(self.facts['auto'],'auto')
        self.assertEqual(self.facts['whisper'],['ja'])
        self.assertEqual(self.facts['qwen'],['auto','zh','ja','en'])

    def test_download_request_uses_the_chosen_output_and_explicit_cookies_file(self):
        self.assertEqual(self.facts['request'],dict(text='分享 https://example.com/watch?v=1',
                                                   output_dir='D:/chosen',cookies_file='D:/cookies.txt'))

    def test_completed_download_is_added_once_and_history_requires_manual_import(self):
        self.assertEqual(self.facts['pendingAfterRepeatedUpdates'],1)
        self.assertEqual(self.facts['pendingBeforeManual'],1)
        self.assertEqual(self.facts['manualCount'],2)

    def test_video_downloads_have_no_subtitle_preview_or_speaker_button(self):
        self.assertFalse(self.facts['binaryPreview'])
        self.assertEqual(self.facts['speakerDisplay'],'none')

    def test_download_error_is_shown_and_button_is_unlocked(self):
        self.assertIn('测试平台需要登录',self.facts['error'])
        self.assertFalse(self.facts['locked'])

    def test_download_button_tracks_removal_and_audio_can_be_queued(self):
        self.assertEqual(self.facts['removedDownload'], '加入待处理')
        self.assertEqual(self.facts['readdedDownload'], '已加入待处理')
        self.assertTrue(self.facts['queuedAudio'])

    def test_partial_submission_can_be_corrected_and_retried(self):
        self.assertEqual(self.facts['separateRequest']['sep_model'], 'MDX')
        self.assertEqual(self.facts['separateRequest']['mode'], 'separate')
        self.assertEqual(self.facts['retained'], [
            ['invalid.wav', '验证路径错误'], ['offline.wav', '验证网络错误']])
        self.assertEqual(self.facts['remaining'], ['offline.wav'])
        self.assertEqual(self.facts['duplicateUploadCount'], 2)
