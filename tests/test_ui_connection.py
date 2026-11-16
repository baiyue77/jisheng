import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from asr_languages import options


class CatalogTests(unittest.TestCase):
    def test_offline_catalog_matches_backend_language_options(self):
        source=(Path(__file__).resolve().parent.parent/'ui.html').read_text(encoding='utf-8')
        match=re.search(r'const DEFAULT_QWEN_LANGUAGES=(\[.*\]);',source)
        self.assertIsNotNone(match, 'offline language catalog must not fall back to two languages')
        self.assertEqual(json.loads(match.group(1)), options())


@unittest.skipUnless(shutil.which('node'), 'Node required')
class ConnectionTests(unittest.TestCase):
    def run_probe(self, mode):
        root=Path(__file__).resolve().parent.parent
        result=subprocess.run(['node',str(root/'tests/frontend/downloads_probe.mjs'),str(root/'ui.html'),mode],
                              capture_output=True,text=True,encoding='utf-8',timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_direct_html_open_explains_launcher_and_does_not_call_file_apis(self):
        data=self.run_probe('file')
        self.assertEqual(data['requests'],0)
        self.assertIn('jisheng-webui.cmd',data['notice'])
        self.assertEqual(data['href'],'http://127.0.0.1:8760/')
        self.assertTrue(data['disabled'])
        self.assertEqual(data['catalog'],31)

    def test_disconnected_backend_disables_execution_and_keeps_full_catalog(self):
        data=self.run_probe('offline')
        self.assertIn('jisheng-webui.cmd',data['notice'])
        self.assertTrue(data['disabled'])
        self.assertEqual(data['catalog'],31)
