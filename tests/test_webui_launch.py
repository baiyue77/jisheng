#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""webui_launch.py 的辅助逻辑测试（不真正启动 webui）。

关注点是「关窗口能不能杀干净」所依赖的两件事：
  1. 能正确解析祖先链，并挑出该监视的进程；
  2. 能建出 KILL_ON_JOB_CLOSE 的 Job Object。
"""

import ctypes
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import webui_launch as W  # noqa: E402


def _k32():
    return W._bind(ctypes.WinDLL("kernel32", use_last_error=True))


class TestProcessTable(unittest.TestCase):
    def test_table_contains_this_process(self):
        import os
        table = W._process_table(_k32())
        self.assertIn(os.getpid(), table)
        _ppid, name = table[os.getpid()]
        self.assertEqual(name, "python.exe")


class TestWatchList(unittest.TestCase):
    def test_returns_ancestors_of_this_process(self):
        k32 = _k32()
        table = W._process_table(k32)
        watched = W._watch_list(k32)

        self.assertTrue(watched, "祖先链不该为空")
        # 每个被监视的 PID 都必须真的存在，否则 watchdog 会立刻误触发
        for pid, name in watched:
            self.assertIn(pid, table, f"PID {pid} 不在进程表里")
            self.assertTrue(name)

    def test_does_not_go_above_innermost_terminal(self):
        """只取到最内层终端为止，避免监视到转瞬即逝的启动器。"""
        k32 = _k32()
        table = W._process_table(k32)
        watched = W._watch_list(k32)

        # 从本进程往上走，找到链条里第一个终端
        pid = __import__("os").getpid()
        chain = []
        seen = {pid}
        for _ in range(12):
            ppid, _n = table.get(pid, (0, ""))
            if not ppid or ppid in seen:
                break
            seen.add(ppid)
            chain.append((ppid, table.get(ppid, (0, ""))[1]))
            pid = ppid

        first_terminal = next(
            (i for i, (_p, n) in enumerate(chain) if n in W._TERMINAL_NAMES),
            None)
        if first_terminal is None:
            self.assertEqual(len(watched), 1)      # 没有终端就只监视直接父进程
        else:
            self.assertEqual(len(watched), first_terminal + 1)


class TestJobObject(unittest.TestCase):
    def test_kill_on_close_job_is_created(self):
        job = W._create_kill_on_close_job(_k32())
        self.assertTrue(job, "KILL_ON_JOB_CLOSE 的 Job Object 应能创建成功")


if __name__ == "__main__":
    unittest.main(verbosity=2)
