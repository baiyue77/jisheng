#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""启动 webui.py，并保证「终端一关，webui 跟着结束」。

要解决的问题
------------
Windows 上 ``venv\Scripts\python.exe`` 是个跳板（redirector stub）：它
拿到命令行后再起一个真正的解释器子进程。直接双击/在终端里跑时，
进程树会长这样::

    cmd.exe            <- 用户看到的窗口
      python.exe       <- venv 跳板
        python.exe     <- 真正的解释器（webui_launch.py）
          python.exe   <- 又是跳板
            python.exe <- 真正的 webui（占着 8760）

关掉窗口 / 强杀 cmd.exe 时，跳板常常先死，真正干活的解释器就变成孤儿，
继续占着 8760 端口，下次启动就报「端口已被占用」。

两道保险
--------
1. **祖先监视**（主要手段）：启动时解析祖先链，找出其中的终端进程
   （cmd / powershell / pwsh / bash / mintty / WindowsTerminal / conhost），
   用 ``WaitForMultipleObjects`` 等它们；**任何一个消失**就立刻结束 webui。
   这样无论用户关的是 cmd 窗口、还是 Git Bash 的终端窗口，都能收拾干净。

2. **Job Object**（兜底）：把 webui 放进一个带
   ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` 的 job。本进程若被强杀/崩溃，
   内核回收 job 句柄时会连带杀掉 job 内的整棵进程树
   （包括 webui 再起的 venv-sep 分离子进程）。

只用标准库（ctypes），不需要 pywin32。
"""

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEBUI = HERE / "webui.py"

# ---- Job Object -----------------------------------------------------------
_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

# ---- 进程枚举 / 等待 -------------------------------------------------------
_PROCESS_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TH32CS_SNAPPROCESS = 0x00000002
_WAIT_TIMEOUT = 0x00000102
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

# 视为「用户终端」的进程名；祖先链里最外层的那个是监视重点
_TERMINAL_NAMES = {
    "cmd.exe", "powershell.exe", "pwsh.exe",
    "bash.exe", "sh.exe", "zsh.exe", "mintty.exe",
    "windowsterminal.exe", "wt.exe", "conhost.exe",
}


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wt.DWORD),
        ("szExeFile", wt.WCHAR * 260),
    ]


def _bind(k32):
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.CreateJobObjectW.argtypes = [wt.LPVOID, wt.LPCWSTR]
    k32.SetInformationJobObject.restype = wt.BOOL
    k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int,
                                            wt.LPVOID, wt.DWORD]
    k32.OpenProcess.restype = wt.HANDLE
    k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k32.AssignProcessToJobObject.restype = wt.BOOL
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.CloseHandle.restype = wt.BOOL
    k32.CloseHandle.argtypes = [wt.HANDLE]
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    k32.Process32FirstW.restype = wt.BOOL
    k32.Process32FirstW.argtypes = [wt.HANDLE,
                                    ctypes.POINTER(_PROCESSENTRY32W)]
    k32.Process32NextW.restype = wt.BOOL
    k32.Process32NextW.argtypes = [wt.HANDLE,
                                   ctypes.POINTER(_PROCESSENTRY32W)]
    k32.WaitForMultipleObjects.restype = wt.DWORD
    k32.WaitForMultipleObjects.argtypes = [wt.DWORD, ctypes.POINTER(wt.HANDLE),
                                           wt.BOOL, wt.DWORD]
    return k32


def _create_kill_on_close_job(k32):
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return None
    return job


def _process_table(k32):
    """返回 {pid: (ppid, exe_name)}。"""
    snap = k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    table = {}
    if snap == _INVALID_HANDLE_VALUE:
        return table
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            table[entry.th32ProcessID] = (entry.th32ParentProcessID,
                                          entry.szExeFile.lower())
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return table


def _watch_list(k32):
    """祖先链里需要监视的进程：从直接父进程起，到**最内层**的终端为止。

    只取最内层终端，不往上传：
    真实场景双击 jisheng-webui.cmd 时链是
        [stub, cmd.exe, explorer.exe]        -> 监视 stub + cmd.exe
    而用脚本/计划任务拉起时链可能是
        [stub, cmd.exe, powershell.exe, ...] -> 只监视 stub + cmd.exe，
    因为再往上的 powershell 可能只是个转瞬即逝的启动器，
    监视它会误杀——之前就这么错过一次。
    """
    table = _process_table(k32)
    pid = os.getpid()
    chain, seen = [], {pid}
    for _ in range(12):
        parent_pid, _name = table.get(pid, (0, ""))
        if not parent_pid or parent_pid in seen:
            break
        seen.add(parent_pid)
        chain.append((parent_pid, table.get(parent_pid, (0, ""))[1]))
        pid = parent_pid

    for i, (_p, name) in enumerate(chain):
        if name in _TERMINAL_NAMES:
            return chain[:i + 1]
    return chain[:1]


def _start_console_watchdog(k32, on_lost):
    """控制台窗口消失就退出（对应「用户点了窗口右上角的 X」）。

    这条信号和「祖先进程退出」互补：某些情况下 cmd.exe 被杀掉、
    但控制台窗口因为还有进程挂着而没消失；反过来窗口关掉时
    祖先进程也未必立刻上报。

    注意 GetConsoleWindow 在 kernel32.dll，不在 user32.dll。
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    k32.GetConsoleWindow.restype = wt.HWND
    k32.GetConsoleWindow.argtypes = []
    user32.IsWindow.restype = wt.BOOL
    user32.IsWindow.argtypes = [wt.HWND]

    hwnd = k32.GetConsoleWindow()
    if not hwnd:
        return False      # 没有控制台（例如被当成服务跑），跳过

    def loop():
        while True:
            time.sleep(1.0)
            if not user32.IsWindow(hwnd):
                on_lost()
                return

    threading.Thread(target=loop, daemon=True).start()
    return True


def _start_watchdog(k32, watched, on_lost):
    """任一被监视进程退出时调用 on_lost（在守护线程里）。"""
    handles = []
    for pid, _name in watched:
        h = k32.OpenProcess(
            _PROCESS_SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION,
            False, pid)
        if h:
            handles.append(h)
    if not handles:
        return False

    arr = (wt.HANDLE * len(handles))(*handles)

    def loop():
        while True:
            r = k32.WaitForMultipleObjects(len(handles), arr, False, 1000)
            if r != _WAIT_TIMEOUT:
                on_lost()
                return

    threading.Thread(target=loop, daemon=True).start()
    return True


def main() -> int:
    # 输出被重定向到文件时，Python 会按区域编码（简体中文下是 GBK）写，
    # 遇到编码不了的字符不能直接崩；另外重定向时默认是块缓冲，
    # 日志要等进程退出才看得见，所以打开行缓冲。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace", line_buffering=True)
        except Exception:
            pass

    if not WEBUI.is_file():
        print(f"[X] 找不到 {WEBUI}")
        return 1

    k32 = _bind(ctypes.WinDLL("kernel32", use_last_error=True))
    job = _create_kill_on_close_job(k32)
    watched = _watch_list(k32)

    print("=" * 64)
    print("  集声")
    print("=" * 64)
    print("  [OK] 关闭本窗口 / 结束终端，会同时结束 webui 及其子进程")
    for pid, name in watched:
        print(f"       监视 {name} (PID {pid})")
    if not job:
        print("  [!] Job Object 不可用，强杀本进程时子进程可能残留")
    print("=" * 64)

    proc = subprocess.Popen([sys.executable, str(WEBUI)] + sys.argv[1:])

    if job:
        handle = k32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, proc.pid)
        if handle:
            k32.AssignProcessToJobObject(job, handle)
            k32.CloseHandle(handle)

    def on_lost():
        # 终端/窗口没了：把 webui 收掉，然后立刻退出。
        # 用 os._exit 避免在守护线程里做多余的清理、也避免被 join 卡住。
        try:
            proc.terminate()
        except Exception:
            pass
        os._exit(0)

    watched_ok = _start_watchdog(k32, watched, on_lost)
    console_ok = _start_console_watchdog(k32, on_lost)
    if watched_ok:
        print("  [OK] 已监视父进程，终端退出即结束服务")
    if console_ok:
        print("  [OK] 已监视控制台窗口，点 X 关闭窗口即结束服务")
    if not watched_ok and not console_ok:
        print("  [!] 无法监视父进程/控制台，关窗口后可能残留进程")

    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n收到中断，正在结束 webui ...")
        proc.terminate()
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 1


if __name__ == "__main__":
    sys.exit(main())
