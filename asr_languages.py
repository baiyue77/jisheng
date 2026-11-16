"""Qwen3-ASR 1.7B-hf language codes, matching the local Transformers processor.

Source: https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf
Chinese dialects without their own processor code are recognized under zh/auto.
"""
LANGUAGES = {
    'zh': '中文（含方言）', 'en': '英语', 'yue': '粤语', 'ja': '日语',
    'ko': '韩语', 'ar': '阿拉伯语', 'de': '德语', 'fr': '法语', 'es': '西班牙语',
    'pt': '葡萄牙语', 'id': '印度尼西亚语', 'it': '意大利语', 'ru': '俄语',
    'th': '泰语', 'vi': '越南语', 'tr': '土耳其语', 'hi': '印地语',
    'ms': '马来语', 'nl': '荷兰语', 'sv': '瑞典语', 'da': '丹麦语',
    'fi': '芬兰语', 'pl': '波兰语', 'cs': '捷克语', 'fil': '菲律宾语',
    'fa': '波斯语', 'el': '希腊语', 'hu': '匈牙利语', 'mk': '马其顿语',
    'ro': '罗马尼亚语',
}


def options():
    return [dict(value='auto', label='自动识别语言')] + [
        dict(value=code, label=f'{label} · {code}') for code, label in LANGUAGES.items()]
