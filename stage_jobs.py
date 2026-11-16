"""Independent media stages. Heavy Qwen inference stays in a child process."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import types
import wave

import numpy as np
from asr_languages import LANGUAGES

HERE = Path(__file__).resolve().parent
QWEN_MODEL = HERE / 'models' / 'Qwen3-ASR-1.7B-hf'
QWEN_PYTHON = HERE / 'venv-diarize' / 'Scripts' / 'python.exe'
MODES = ('extract', 'separate', 'recognize', 'diarize', 'asr')


def qwen_status():
    ready = (QWEN_MODEL / 'model.safetensors').is_file() and QWEN_PYTHON.is_file()
    return ready, ('已找到本地 Qwen3-ASR 1.7B 权重；首次执行时加载环境'
                   if ready else '缺少本地 Qwen 权重或 venv-diarize 运行环境')


def validate(opts):
    mode = opts.get('mode') or 'asr'
    backend = opts.get('backend') or 'whisper'
    language = opts.get('language') or ('zh' if backend == 'qwen' else 'ja')
    if mode not in MODES:
        raise ValueError('未知的执行功能：' + str(mode))
    if backend not in ('whisper', 'qwen'):
        raise ValueError('未知识别引擎：' + str(backend))
    if language not in (*LANGUAGES, 'auto') or (backend == 'whisper' and language != 'ja'):
        raise ValueError('anime-whisper 仅支持日语；Qwen 请选择支持的语言或自动识别')
    return mode, backend, language


def work_path(w, job):
    media = Path(job['media'])
    _, backend, language = validate(job['opts'])
    digest = hashlib.sha256(str(media.resolve()).casefold().encode('utf-8')).hexdigest()[:10]
    return w.WORK_DIR / (w.safe_name(media.stem)[:80] + '__' + digest + '_' + backend + '_' + language)


def chunks_for(w, audio, vad):
    spans = (w.A.vad_chunks(audio, vad_min_silence=500) if vad
             else [dict(start=0, end=len(audio))])
    result = []
    # The local Qwen processor supports at most 30 seconds. Bound all new
    # stage chunks to 25 seconds, including continuous speech and VAD-off input.
    for span in spans:
        for start in range(int(span['start']), int(span['end']), 25 * 16000):
            result.append(dict(start=start, end=min(start + 25 * 16000, int(span['end']))))
    return result


def read_results(path):
    raw = Path(path).read_bytes()
    # Decode only complete records: the writer may be midway through a UTF-8
    # character in the last row when the parent polls the file.
    for line in raw.split(b'\n')[:-1]:
        if line.strip():
            yield json.loads(line.decode('utf-8'))


def run_qwen(w, job, chunks, work, on_segment):
    ready, note = qwen_status()
    if not ready:
        raise RuntimeError(note)
    with w._model_lock:
        w._model_cache.update(key=None, model=None)
    gc.collect()
    audio = w.A.read_wav_float32(work / 'asr_input.wav')
    clip_dir = work / 'qwen_chunks'
    clip_dir.mkdir(exist_ok=True)
    manifest = []
    for chunk in chunks:
        if w.stop_requested(job):
            raise RuntimeError('__STOPPED__')
        index = chunk['index']
        dest = clip_dir / f'{index:06d}.wav'
        with wave.open(str(dest), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            piece = np.clip(audio[chunk['start']:chunk['end']], -1, 1)
            stream.writeframes((piece * 32767).astype('<i2').tobytes())
        manifest.append(dict(file=str(dest), index=index,
                             start=chunk['start']/16000, end=chunk['end']/16000))
    manifest_path = work / 'qwen_manifest.json'
    w.A.atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False))
    result_path = work / 'qwen_result.jsonl'
    result_path.unlink(missing_ok=True)
    cmd = [str(QWEN_PYTHON), '-u', str(HERE / 'qwen_worker.py'), str(clip_dir),
           '--manifest', str(manifest_path), '--model', str(QWEN_MODEL),
           '--out', str(result_path), '--language',
           job['opts'].get('language') or 'zh',
           '--batch', '1', '--tmp', str(work / 'qwen_cache')]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace', env=env)
    w._set(job, _proc=proc)
    errors = []
    def drain(pipe, error=False):
        for line in pipe:
            if error:
                errors.append(line.rstrip())
            elif line.strip():
                w._emit(job, line.strip())
    readers = [threading.Thread(target=drain, args=(proc.stdout,), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, True), daemon=True)]
    for reader in readers:
        reader.start()
    seen = set()
    expected = {row['index'] for row in manifest}
    def harvest():
        if not result_path.exists():
            return
        for row in read_results(result_path):
            index = row['index']
            if index in expected and index not in seen:
                on_segment(index, dict(start=row['start'], end=row['end'], text=row['text']))
                seen.add(index)
    try:
        while proc.poll() is None:
            harvest()
            if w.stop_requested(job):
                w.A._terminate(proc)
                harvest()
                raise RuntimeError('__STOPPED__')
            time.sleep(.25)
        harvest()
        if w.stop_requested(job):
            raise RuntimeError('__STOPPED__')
        if proc.returncode != 0:
            raise RuntimeError('Qwen 识别失败：' + ' | '.join(errors[-6:]))
        if seen != expected:
            raise RuntimeError('Qwen 结果不完整，已完成片段已保留，可继续')
    finally:
        if proc.poll() is None:
            w.A._terminate(proc)
        for reader in readers:
            reader.join(timeout=3)
        for pipe in (proc.stdout, proc.stderr):
            pipe.close()
        w._set(job, _proc=None)


def run(w, job):
    mode, backend, language = validate(job['opts'])
    opts = job['opts']
    media = Path(job['media'])
    out = Path(job.get('out_dir') or w.OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    work = work_path(w, job)
    work.mkdir(parents=True, exist_ok=True)
    w._set(job, work=str(work), out_dir=str(out))
    stat = media.stat()
    fingerprint = [str(media.resolve()), stat.st_size, stat.st_mtime_ns]
    state = w.load_state(work) or {}
    if state.get('fingerprint') != fingerprint:
        w._wipe_checkpoint(work)
        (work / 'asr_input.wav').unlink(missing_ok=True)
        state = dict(fingerprint=fingerprint)
    previous_out = state.get('subtitle_out_dir') or state.get('out_dir')
    if state.get('subtitle_stem') and previous_out:
        state['subtitle_out_dir'] = previous_out
    state.update(media=str(media), media_size=stat.st_size, media_mtime=stat.st_mtime,
                 opts=dict(opts), out_dir=str(out), status='running')
    w.save_state(work, state)
    log = lambda message: w._emit(job, message)
    stopped = lambda: w.stop_requested(job) or w.abandon_requested(job)

    def finish(files):
        if stopped():
            raise RuntimeError('__STOPPED__')
        state['status'] = 'done'
        w.save_state(work, state)
        w._set(job, status='done', stage='完成', progress=1., eta=0., files=files)

    def asset(path, kind):
        return dict(name=path.name, path=str(path), kind=kind)

    try:
        if stopped():
            raise RuntimeError('__STOPPED__')
        if mode == 'extract':
            w._set(job, stage='抽取音频', progress=.1)
            stem = w.available_output_stem(out, media.stem, ('.audio.wav',), media)
            dest = out / (stem + '.audio.wav')
            pending = work / 'extract.wav'
            w.ts.ffmpeg_convert(media, pending, sr=44100, should_stop=stopped, progress=log)
            shutil.copy2(pending, dest)
            finish([asset(dest, '音频')])
            return
        if mode == 'separate':
            w._set(job, stage='人声分离', progress=.1)
            raw = w.ts.ffmpeg_convert(media, work / 'raw44k.wav', sr=44100,
                                     should_stop=stopped, progress=log)
            _vocals, files = w.separate_vocals(raw, work / 'sep', media.stem,
                opts.get('sep_model') or w.SEP_MODEL_CHOICES[1]['value'], log, job)
            if not files:
                raise RuntimeError('分离完成但未成功导出音频，请检查输出目录')
            finish(files)
            return

        raw = work / 'raw16k.wav'
        if not raw.exists():
            w._set(job, stage='抽取音频', progress=.05)
            w.ts.ffmpeg_convert(media, raw, sr=16000, should_stop=stopped, progress=log)
        if stopped():
            raise RuntimeError('__STOPPED__')
        if mode == 'asr' and opts.get('separate'):
            _raw, files = w.separate_vocals(raw, work / 'sep', media.stem,
                opts.get('sep_model') or w.SEP_MODEL_CHOICES[1]['value'], log, job)
            shutil.copy2(_raw, work / 'vocals_16k.wav')
            job['files'] = files
            raw = work / 'vocals_16k.wav'

        if mode == 'diarize' or (mode == 'asr' and opts.get('diarize')):
            w._set(job, stage='说话人分割', progress=.2)
            w._wipe_annotation(work)
            dargs = types.SimpleNamespace(diarize_model=opts.get('diarize_model'),
                device='cuda' if w._cuda_ok() else 'cpu', num_speakers=opts.get('num_speakers'))
            dia = w.A.run_diarization(raw, dargs, out_json=work / 'diarization.json',
                out_embeddings=work / w.EMBEDDINGS_FILE,
                on_proc=lambda proc: w._set(job, _proc=proc), should_stop=stopped)
            if dia is None:
                raise RuntimeError('说话人分割失败，详见控制台；未启动识别')
            # Bootstrap without deleting the newly written embedding sidecar.
            labels = w.speaker_labels.bootstrap(work, job['id'], str(media),
                                                w._dia_speakers(dia), 'media')
            labels['language'] = language
            w.speaker_labels.save_labels(work, labels)
            w._store_cluster_embeddings(work, labels)
            if mode == 'diarize':
                stem = w.available_output_stem(out, media.stem, ('.diarization.json',), media)
                dest = out / (stem + '.diarization.json')
                w.A.atomic_write_text(dest, json.dumps(dia, ensure_ascii=False, indent=2))
                existing = w.load_segments(work)
                if existing:
                    w.A.assign_speakers(existing, dia, opts.get('diarize_min_overlap') or .3)
                    w.A.atomic_write_text(work / 'segments.jsonl', ''.join(json.dumps(s, ensure_ascii=False)+'\n' for s in existing))
                    stem = state.get('subtitle_stem')
                    if stem and previous_out and Path(previous_out).resolve() == out.resolve():
                        subtitle_out = Path(previous_out)
                        with w.SUBTITLE_OUTPUT_LOCK:
                            w.speaker_labels.rewrite_outputs(w.speaker_labels.load_labels(work), existing,
                                subtitle_out / f'{stem}.{language}.srt', subtitle_out / f'{stem}.{language}.txt')
                        w._set(job, subtitle_stem=stem, subtitle_out_dir=str(subtitle_out))
                finish([asset(dest, '分割结果')])
                return

        w._set(job, stage='识别中', progress=.25)
        audio = w.A.read_wav_float32(raw)
        chunks = chunks_for(w, audio, opts.get('vad', True))
        if not chunks:
            raise RuntimeError('未检测到语音；可关闭 VAD 重试')
        key = w._opts_key(opts)
        if state.get('recognition_key') != key:
            (work / 'segments.jsonl').unlink(missing_ok=True)
            state['next_chunk'] = 0
            state['recognition_key'] = key
        done = w.load_segments(work)
        start = max(state.get('next_chunk', 0), max((s.get('chunk_index', -1)+1 for s in done), default=0))
        state['total_chunks'] = len(chunks)
        state['duration'] = len(audio) / 16000
        glossary = w.A.load_glossary(Path(opts['glossary'])) if opts.get('glossary') else []
        dia_path = work / 'diarization.json'
        dia = json.loads(dia_path.read_text(encoding='utf-8')) if dia_path.exists() else None
        def accept(index, segment):
            if segment and segment.get('text', '').strip():
                segment['text'] = w.A.apply_glossary(segment['text'], glossary)
                segment['chunk_index'] = index
                if dia:
                    w.A.assign_speakers([segment], dia, opts.get('diarize_min_overlap') or .3)
                w.append_segment(work, segment)
                done.append(segment)
                log(segment['text'])
            state['next_chunk'] = index + 1
            w.save_state(work, state)
            w._set(job, progress=.25+.7*(index+1)/len(chunks))
        if backend == 'qwen' and start < len(chunks):
            shutil.copy2(raw, work / 'asr_input.wav')
            run_qwen(w, job, [dict(ch, index=i) for i, ch in enumerate(chunks) if i>=start], work, accept)
        elif backend == 'whisper':
            device = 'cuda' if w._cuda_ok() else 'cpu'
            model = w._get_model(w.A.resolve_model(opts.get('asr_model')), device,
                                 'float16' if device=='cuda' else 'int8', log)
            for index in range(start, len(chunks)):
                if stopped():
                    raise RuntimeError('__STOPPED__')
                accept(index, w.A.transcribe_chunk(model, audio, chunks[index], opts.get('beam') or 5, glossary))
        if stopped():
            raise RuntimeError('__STOPPED__')
        if not done:
            raise RuntimeError('没有识别到文字')
        stem = media.stem + ('.qwen' if backend == 'qwen' and language == 'ja' else '')
        labels = w.speaker_labels.load_labels(work)
        labels['language'] = language
        w.speaker_labels.save_labels(work, labels)
        stem, srt, txt = w.publish_subtitles(out, stem, language, media, labels, done)
        state['subtitle_stem'] = stem
        state['subtitle_out_dir'] = str(out)
        w._set(job, subtitle_stem=stem, subtitle_out_dir=str(out))
        finish(list(job.get('files', [])) + [asset(srt, '字幕'), asset(txt, '文本')])
    except (w.ts.SourceCancelled, RuntimeError) as exc:
        if stopped() or '__STOPPED__' in str(exc):
            if w.abandon_requested(job):
                w._discard_job(job)
                w._set(job, status='abandoned', stage='已放弃')
            else:
                state['status'] = 'stopped'
                w.save_state(work, state)
                w._set(job, status='stopped', stage='已停止')
            return
        raise
