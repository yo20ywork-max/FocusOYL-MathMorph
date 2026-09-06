"""Full public-task comparison of the original MiniCPM5 F16 and Prism GGUF.
No training, uploads, cloud Jobs, paid inference, or model-generated code execution.
Task prompts/scorers: lm-evaluation-harness 0.4.13. Chat transport: benchlib.py.
"""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache' / 'huggingface'))
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NLTK_DATA'] = str(ROOT / 'cache' / 'nltk')
Path(os.environ['NLTK_DATA']).mkdir(parents=True, exist_ok=True)
from benchlib import Server, canonical, json_default, make_adapter, save, sha

MODEL_HASHES = {
    'original': '68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d',
    'prism': '24f86e98d327afb5d17418486708b0dc653749d6eed67150c4f58f8c6591a41b',
}
TASKS = {
    'gsm8k_cot_zeroshot': ('gsm8k/gsm8k-cot-zeroshot.yaml', 'openai/gsm8k'),
    'ifeval': ('ifeval/ifeval.yaml', 'google/IFEval'),
}


def lock_run(folder: Path):
    path = folder / 'run.lock'
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('This run is already open in another terminal. Do not launch it twice.')
    return handle


def load_config(name, task_root, revision, maximum):
    from lm_eval.tasks._yaml_loader import load_yaml as load_yaml_config
    config = load_yaml_config(str(task_root / TASKS[name][0]))
    config['dataset_kwargs'] = {**(config.get('dataset_kwargs') or {}), 'revision': revision}
    config['num_fewshot'] = 0
    config['generation_kwargs'] = {'until': [], 'max_gen_toks': maximum,
                                   'do_sample': False, 'temperature': 0.0}
    return config


def evaluate_task(adapter, config, limit=None):
    from lm_eval import evaluator
    result = evaluator.simple_evaluate(model=adapter, tasks=[config], num_fewshot=0,
        batch_size=1, limit=limit, apply_chat_template=True, log_samples=True,
        random_seed=20260906, numpy_random_seed=20260906, torch_random_seed=None,
        fewshot_random_seed=20260906, bootstrap_iters=1000)
    if result is None or not result.get('results'):
        raise RuntimeError('The official evaluator did not return results')
    return result


def report(run: Path, smoke: int):
    results = {arm: {task: json.loads((run / arm / task / 'results.json').read_text(encoding='utf-8'))
                    for task in TASKS if (run / arm / task / 'results.json').is_file()}
               for arm in MODEL_HASHES}
    rows = []
    for task in TASKS:
        if any(task not in results[a] for a in results):
            continue
        left, right = (results[a][task]['results'][task] for a in ('original', 'prism'))
        for metric, first in left.items():
            if metric not in right or metric == 'alias' or 'stderr' in metric or not isinstance(first, (int, float)):
                continue
            second = right[metric]
            rows.append({'task': task, 'metric': metric, 'original': first,
                         'prism': second, 'delta_percentage_points': 100 * (second - first)})
    with (run / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'metric', 'original', 'prism', 'delta_percentage_points'])
        writer.writeheader()
        writer.writerows(rows)
    costs = {}
    for arm in MODEL_HASHES:
        costs[arm] = {}
        for task in TASKS:
            p = run / arm / task / 'cost.json'
            if p.is_file():
                costs[arm][task] = json.loads(p.read_text(encoding='utf-8'))
    scope = 'SMOKE TEST: first samples only; not full benchmark scores' if smoke else 'FULL DATASET SPLITS, zero-shot, think on, bounded generation'
    lines = ['# Original vs FocusOYL Prism', '', scope, '',
      'Uses lm-evaluation-harness 0.4.13 task prompts and scorers with explicitly overridden generation budget and stop strings.',
      'No cloud inference, leaderboard submission, third-party verification, or model training was performed by this script.',
      'GSM8K: primary flexible-extract; strict-match also reported. IFEval: primary prompt_level_strict_acc.',
      'The server returns reasoning separately. Only final content is scored. Official metrics may score a partial final answer;',
      'generation-limit counts are therefore reported separately. Timeouts/API errors stop the run instead of being silently scored.',
      'This is not a reproduction of an unspecified public leaderboard or proof of universal improvement.', '',
      '| Task | Metric | Original | Prism | Delta (pp) |', '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['task']} | {r['metric']} | {100*r['original']:.2f}% | {100*r['prism']:.2f}% | {r['delta_percentage_points']:+.2f} |")
    lines.extend(['', '## Actual generated tokens, including thinking', '',
                  '| Arm | Task | Requests | Tokens | Length stops |', '|---|---|---:|---:|---:|'])
    for a, entries in costs.items():
        for t, c in entries.items():
            lines.append(f"| {a} | {t} | {c['requests']} | {c['completion_tokens']} | {c['length_stops']} |")
    complete = all(len(v) == len(TASKS) for v in results.values())
    lines += ['', f'Complete: {complete}. Raw API records and official per-sample results are stored alongside this report.',
              'Existing recorded answers are reused on resume; this is checkpointing, not additional model trials.']
    (run / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    save(run / 'comparison.json', {'complete': complete, 'scope': scope, 'metrics': rows, 'cost': costs,
         'official_leaderboard_submission': False, 'verified_by_third_party': False})
    print('\nREPORT:', run / 'comparison.md', flush=True)
    for r in rows:
        print(f"{r['task']} {r['metric']}: original={100*r['original']:.2f}% prism={100*r['prism']:.2f}% delta={r['delta_percentage_points']:+.2f} pp")


def main():
    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=home / 'Documents/Codex/2026-08-30/new-chat-2/outputs/focusoyl-tools/models/MiniCPM5-1B-F16.gguf')
    p.add_argument('--prism', type=Path, default=home / 'Documents/FocusOYL-MathMorph-20260905-D/study/models/euclid_control/candidate.gguf')
    p.add_argument('--server', type=Path, default=home / 'llama/runtime/llama-server.exe')
    p.add_argument('--smoke', type=int, default=0, help='First N cases per task; 0 means full test splits')
    p.add_argument('--max-tokens', type=int, default=4096, help='Total reasoning plus final-answer budget')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--workers', type=int, choices=[1, 2, 4], default=2)
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--new-run', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--check', action='store_true', help='Check installation and local files without loading either model')
    args = p.parse_args()
    if args.self_test:
        from selftest import run_tests
        run_tests()
        return
    if args.smoke < 0 or not 256 <= args.max_tokens < args.context:
        p.error('smoke must be nonnegative, 256 <= max_tokens < context')
    if importlib.metadata.version('lm_eval') != '0.4.13':
        raise RuntimeError('Use start.py to install the pinned evaluator')
    models = {'original': args.source.resolve(strict=True), 'prism': args.prism.resolve(strict=True)}
    server = args.server.resolve(strict=True)
    print('Checking original and Prism hashes. Model files are read-only.', flush=True)
    for label, path in models.items():
        actual = sha(path)
        if actual != MODEL_HASHES[label]:
            raise RuntimeError(f'{label} SHA-256 mismatch. Refusing to evaluate the wrong candidate: {path}\nActual: {actual}')
    if args.check:
        from lm_eval import evaluator
        from lm_eval.tasks._yaml_loader import load_yaml as load_yaml_config
        import lm_eval
        base = Path(lm_eval.__file__).parent / 'tasks'
        for name in TASKS:
            assert load_config(name, base, 'main', args.max_tokens)['output_type'] == 'generate_until'
        print('CHECK PASSED: two exact GGUF files, evaluator imports, and both task definitions. No inference started.')
        return
    run_name = ('smoke-' + str(args.smoke) if args.smoke else 'full') + f'-think-on-{args.max_tokens}'
    if args.new_run:
        run_name += '-' + time.strftime('%Y%m%d-%H%M%S')
    run = ROOT / 'runs' / run_name
    run.mkdir(parents=True, exist_ok=True)
    lock = lock_run(run)
    try:
        import lm_eval
        import nltk
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        task_root = Path(lm_eval.__file__).parent / 'tasks'
        settings = {'source_hashes': MODEL_HASHES, 'server_sha256': sha(server),
            'engine_version': subprocess.run([str(server), '--version'], capture_output=True, text=True, errors='replace').stderr,
            'max_tokens': args.max_tokens, 'context_per_request': args.context, 'workers': args.workers,
            'gpu_layers': args.gpu_layers, 'thinking': True, 'temperature': 0, 'seed': 20260906,
            'few_shot': 0, 'custom_stop_strings': [], 'context_shift': False, 'smoke_limit': args.smoke,
            'task_yaml_sha256': {k: sha(task_root / v[0]) for k, v in TASKS.items()},
            'code_sha256': {n: sha(ROOT / n) for n in ['run_benchmark.py', 'benchlib.py']},
            'versions': {n: importlib.metadata.version(n) for n in ['lm_eval', 'datasets', 'evaluate', 'nltk', 'langdetect']},
            'paths': {k: str(v) for k, v in models.items()}}
        pp = run / 'protocol.json'
        if pp.exists():
            protocol = json.loads(pp.read_text(encoding='utf-8'))
            if protocol['settings'] != settings:
                raise RuntimeError('Settings/code/environment changed. Use --new-run instead of mixing runs.')
        else:
            from huggingface_hub import HfApi
            revisions = {name: HfApi().dataset_info(repo, token=False).sha for name, (_, repo) in TASKS.items()}
            protocol = {'settings': settings, 'dataset_revisions': revisions, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'methodology': 'Official task prompts/scorers; localhost chat adapter; generation until EOS or common budget; final content only'}
            save(pp, protocol)
            frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
            (run / 'requirements-lock.txt').write_text(frozen.stdout, encoding='utf-8')
        for resource in ['punkt', 'punkt_tab']:
            nltk.download(resource, download_dir=os.environ['NLTK_DATA'], quiet=True, raise_on_error=True)
        print('Mode:', 'FULL splits' if not args.smoke else f'SMOKE first {args.smoke} cases only', flush=True)
        print('Stop: Ctrl+C. Repeating the same command resumes saved requests. No scores are uploaded.', flush=True)
        for label, model_path in models.items():
            pending = [n for n in TASKS if not (run / label / n / 'results.json').is_file()]
            if not pending:
                print(label, 'already complete; reusing saved results', flush=True)
                continue
            with Server(server, model_path, label, run / label, args.context, args.workers, args.max_tokens, args.gpu_layers) as service:
                for name in pending:
                    folder = run / label / name
                    folder.mkdir(parents=True, exist_ok=True)
                    rev = protocol['dataset_revisions'][name]
                    cfg = load_config(name, task_root, rev, args.max_tokens)
                    adapter = make_adapter(service, MODEL_HASHES[label], name, folder, args.max_tokens, rev)
                    result = evaluate_task(adapter, cfg, limit=args.smoke or None)
                    cost = {'requests': len(adapter.rows),
                        'completion_tokens': sum(x['usage']['completion_tokens'] for x in adapter.rows),
                        'length_stops': sum(x['finish_reason'] == 'length' for x in adapter.rows),
                        'natural_stops': sum(x['finish_reason'] == 'stop' for x in adapter.rows),
                        'think_nonempty': sum(bool(x['reasoning_content']) for x in adapter.rows)}
                    save(folder / 'cost.json', cost)
                    save(folder / 'results.json', result)
                    report(run, args.smoke)
        for label, path in models.items():
            if sha(path) != MODEL_HASHES[label]:
                raise RuntimeError('Model file changed during evaluation')
        (run / 'INTERRUPTED.json').unlink(missing_ok=True)
        save(run / 'COMPLETE.json', {'complete': True, 'models_unchanged': True, 'cloud_jobs_submitted': 0})
        report(run, args.smoke)
    except BaseException as exc:
        save(run / 'INTERRUPTED.json', {'error': repr(exc), 'complete': False,
             'note': 'No partial run is labelled a complete benchmark. Same command resumes recorded requests.'})
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Owned model server closed. Saved responses remain for resume.')
        raise SystemExit(130)
