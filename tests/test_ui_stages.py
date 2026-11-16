import json, shutil, subprocess, unittest
from pathlib import Path
class StageUITest(unittest.TestCase):
 @unittest.skipUnless(shutil.which('node'), 'Node unavailable')
 def test_stage_behavior(self):
  root=Path(__file__).resolve().parent.parent
  result=subprocess.run(['node',str(root/'tests/frontend/stages_probe.mjs'),str(root/'ui.html')],capture_output=True,text=True,encoding="utf-8",check=True)
  data=json.loads(result.stdout)
  self.assertEqual(data['language'],'zh')
  self.assertEqual(data['out'],'D:/chosen')
  self.assertIn('D%3A%2Fchosen',data['open'])
  self.assertEqual(data['req']['mode'],'extract')
  self.assertFalse(data['req']['separate'])
  self.assertEqual(data['button'],'抽取音频')
  self.assertTrue(data['disabled'])

