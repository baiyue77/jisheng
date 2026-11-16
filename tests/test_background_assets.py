import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import webui


class BackgroundAssetsTests(unittest.TestCase):
    def test_serves_only_named_backgrounds(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('megumin.webp', 'eris.webp', 'klee.webp', 'private.webp'):
                (root / name).write_bytes(b'test')
            with patch.object(webui, 'BACKGROUND_DIR', root, create=True):
                with ThreadingHTTPServer(('127.0.0.1', 0), webui.Handler) as server:
                    worker = threading.Thread(target=server.serve_forever)
                    worker.start()
                    base = f'http://127.0.0.1:{server.server_port}/assets/backgrounds/'
                    try:
                        for name in ('megumin.webp', 'eris.webp', 'klee.webp'):
                            with urlopen(base + name, timeout=2) as response:
                                self.assertEqual(response.headers.get_content_type(), 'image/webp')
                                self.assertEqual(response.read(), b'test')
                        for name in ('private.webp', '%2e%2e/ui.html'):
                            with self.assertRaises(HTTPError) as error:
                                urlopen(base + name, timeout=2)
                            self.assertEqual(error.exception.code, 404)
                    finally:
                        server.shutdown()
                        worker.join()
