#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
rename_by_text.py —— 按 ASR 台词重命名切片音频（对齐 GPT-SoVITS 的「文本.wav」命名）

用法
----
    python rename_by_text.py <音频目录> --list xxx.list --dry-run
    python rename_by_text.py <音频目录> --list xxx.list

规则
----
* 文件名 = 该段台词 + .wav，保留全角标点
* 半角非法字符 \\ / : * ? " < > | 统一换成全角或删除
* 超长按 80 字截断，并保证完整路径 < 250 字符
* 重名自动加 _2 _3；空文本保持原名
* 同目录写 _rename_map.tsv（旧名 -> 新名），可用于回滚
"""

import argparse
import os
import re
import sys
from pathlib import Path

BAD = str.maketrans({
    "\\": "＼", "/": "／", ":": "：", "*": "＊", "?": "？",
    '"': "”", "<": "＜", ">": "＞", "|": "｜",
})


def sanitize(text: str, max_len: int = 80) -> str:
    t = (text or "").strip()
    t = t.translate(BAD)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .")
    if len(t) > max_len:
        t = t[:max_len].rstrip("，、, ")
    return t


def load_list(path: Path) -> dict:
    """解析 list：路径|说话人|语言|文本"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        out[os.path.basename(parts[0])] = parts[3]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--list", required=True, help="含台词的 list 文件")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-len", type=int, default=80)
    args = ap.parse_args()

    src = Path(args.src)
    texts = load_list(Path(args.list))
    files = sorted(src.glob("*.wav"))
    if not files:
        print(f"[X] 没有 wav：{src}", file=sys.stderr)
        return 2

    used, plan, skipped = {}, [], []
    for f in files:
        text = texts.get(f.name, "")
        name = sanitize(text, args.max_len)
        if not name:
            skipped.append(f.name)
            continue
        # 保证完整路径不超 Windows 限制
        while len(str(src / (name + ".wav"))) > 250 and len(name) > 10:
            name = name[:-5]
        full = name + ".wav"
        if full in used:
            used[full] += 1
            full = f"{name}_{used[full]}.wav"
        else:
            used[full] = 1
        plan.append((f, src / full, text))

    print(f"待改名 {len(plan)} 个 / 保持原名 {len(skipped)} 个")
    longest = max((len(p[1].name) for p in plan), default=0)
    print(f"最长文件名 {longest} 字符；最长完整路径 "
          f"{max((len(str(p[1])) for p in plan), default=0)} 字符")
    print("\n示例：")
    for old, new, _ in plan[:8]:
        print(f"  {old.name[-26:]}  ->  {new.name}")

    if args.dry_run:
        print("\n[dry-run] 未改动任何文件")
        return 0

    mp = src / "_rename_map.tsv"
    done = 0
    with open(mp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("old\tnew\n")
        for old, new, _ in plan:
            if old == new:
                continue
            try:
                old.rename(new)
                fh.write(f"{old.name}\t{new.name}\n")
                done += 1
            except OSError as exc:
                print(f"[!] 失败 {old.name}: {exc}", file=sys.stderr)
    print(f"\n已改名 {done} 个；映射表 -> {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
