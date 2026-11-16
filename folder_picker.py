"""Native folder dialog, invoked only by an explicit browser button click."""
import base64
import os
import subprocess
import threading

_lock = threading.Lock()


def choose_folder(initial=''):
    if os.name != 'nt':
        raise RuntimeError('此目录选择器需要 Windows；也可直接填写路径')
    script = r'''
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择集声的输出文件夹'
$dialog.ShowNewFolderButton = $true
if (Test-Path -LiteralPath $env:JISHENG_INITIAL_DIR -PathType Container) {
    $dialog.SelectedPath = $env:JISHENG_INITIAL_DIR
}
try {
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write($dialog.SelectedPath)
    }
} finally { $dialog.Dispose() }
'''
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    with _lock:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-STA', '-WindowStyle',
                                 'Hidden', '-EncodedCommand', encoded],
            env=dict(os.environ, JISHENG_INITIAL_DIR=initial), capture_output=True,
            encoding='utf-8', errors='replace', timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-800:] or '系统对话框启动失败')
    return result.stdout.strip()
