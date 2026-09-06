"""Local inference transport, provenance, and process cleanup. No benchmark scoring here."""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import json
import os
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
    temp.replace(path)


def json_default(x: Any) -> Any:
    if hasattr(x, 'item'):
        return x.item()
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


class Server:
    def __init__(self, executable: Path, model: Path, alias: str, output: Path,
                 context: int, workers: int, max_tokens: int, gpu_layers: int):
        import requests
        self.output, self.alias = output, alias
        self.context, self.workers, self.max_tokens = context, workers, max_tokens
        self.proc = None
        self.log = None
        self.key = secrets.token_urlsafe(32)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers['Authorization'] = 'Bearer ' + self.key
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        self.base = f'http://127.0.0.1:{port}'
        self.command = [str(executable), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                        '-ngl', str(gpu_layers), '-t', '4', '-tb', '4', '-c', str(context * workers),
                        '--parallel', str(workers), '--no-kv-unified', '--jinja', '--reasoning', 'on',
                        '--reasoning-format', 'deepseek', '--reasoning-budget', '-1',
                        '-n', str(max_tokens), '--no-context-shift', '--cache-ram', '0',
                        '--timeout', '600', '--alias', alias, '--api-key', self.key]

    def request(self, route: str, data: dict | None = None, timeout: int = 30):
        r = self.session.get(self.base + route, timeout=timeout) if data is None else self.session.post(self.base + route, json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        save(self.output / 'launch.json', {'command': self.command[:-1] + ['<ephemeral-local-key>']})
        self.log = (self.output / 'server.log').open('ab')
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        self.proc = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=self.log,
                                     stderr=subprocess.STDOUT, creationflags=flags)
        try:
            deadline = time.monotonic() + 180
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'llama-server exited. Read {self.output / "server.log"}')
                try:
                    self.request('/health', timeout=2)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('llama-server startup exceeded 180 seconds')
                    time.sleep(.5)
            identity = self.request('/v1/models')
            if self.alias not in [x['id'] for x in identity['data']]:
                raise RuntimeError('Unexpected server identity')
            save(self.output / 'identity.json', identity)
            messages = [{'role': 'user', 'content': 'Compute 13 * 7 - 4.'}]
            template = self.request('/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': True}})
            if not template['prompt'].endswith('<|im_start|>assistant\n<think>\n'):
                raise RuntimeError('Expected MiniCPM5 think-on template was not applied')
            save(self.output / 'template-probe.json', template)
            probe = self.request('/v1/chat/completions', {
                **generation_body(self.alias, messages, 128), 'stream': False}, timeout=120)
            msg = probe['choices'][0]['message']
            if not (msg.get('reasoning_content') or msg.get('reasoning')):
                raise RuntimeError('Thinking mode did not return a reasoning field')
            save(self.output / 'mode-probe.json', probe)
            print(f'CHECK OK: {self.alias}, think=on, localhost only', flush=True)
            return self
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.proc and self.proc.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.proc.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.log:
            self.log.close()
        self.session.close()
        if self.proc:
            save(self.output / 'cleanup.json', {'pid': self.proc.pid, 'exited': self.proc.poll() is not None,
                 'kv_release': 'process-local allocations released on exit', 'unrelated_processes_untouched': True})

    def __exit__(self, *unused):
        self.stop()


def generation_body(alias: str, messages: list, maximum: int) -> dict:
    return {'model': alias, 'messages': messages, 'temperature': 0.0, 'top_p': 1.0,
            'top_k': 0, 'min_p': 0.0, 'repeat_penalty': 1.0, 'seed': 20260906,
            'max_tokens': maximum, 'stream': True, 'stream_options': {'include_usage': True},
            'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': True}}


def collect_stream(response, cancel: threading.Event, checkpoint, wall_seconds=600) -> dict:
    text, reasoning = [], []
    finish, usage = None, {}
    started, last = time.monotonic(), time.monotonic()
    for raw in response.iter_lines(chunk_size=1024):
        now = time.monotonic()
        if cancel.is_set():
            raise InterruptedError('Stopped by user')
        if now - started > wall_seconds:
            raise TimeoutError('Observation wall-time limit reached; run is incomplete')
        if not raw or not raw.startswith(b'data:'):
            continue
        payload = raw[5:].strip()
        if payload == b'[DONE]':
            break
        obj = json.loads(payload)
        if obj.get('error'):
            raise RuntimeError(str(obj['error']))
        if obj.get('usage'):
            usage = obj['usage']
        for c in obj.get('choices', []):
            delta = c.get('delta') or {}
            if delta.get('content'):
                text.append(delta['content'])
            if delta.get('reasoning_content') or delta.get('reasoning'):
                reasoning.append(delta.get('reasoning_content') or delta['reasoning'])
            if c.get('finish_reason'):
                finish = c['finish_reason']
        if now - last > 15:
            checkpoint({'final_content': ''.join(text), 'reasoning_content': ''.join(reasoning),
                        'elapsed_seconds': now - started, 'status': 'RUNNING'})
            last = now
    if finish not in ('stop', 'length') or 'completion_tokens' not in usage:
        raise RuntimeError('Incomplete stream or missing usage; no score certified')
    return {'final_content': ''.join(text), 'reasoning_content': ''.join(reasoning),
            'finish_reason': finish, 'usage': usage, 'elapsed_seconds': time.monotonic() - started}


def make_adapter(server: Server, model_hash: str, task_name: str, task_dir: Path,
                 max_tokens: int, revision: str):
    from lm_eval.api.model import LM
    import requests

    class GGUFAdapter(LM):
        def __init__(self):
            super().__init__()
            self.cancel = threading.Event()
            self.rows = []

        @property
        def tokenizer_name(self):
            return 'MiniCPM5-embedded-template-' + model_hash[:12]

        def apply_chat_template(self, chat_history, add_generation_prompt=True, **kwargs):
            if not add_generation_prompt:
                raise ValueError('This transport does not support assistant prefill tasks')
            return json.dumps(chat_history, ensure_ascii=False)

        def loglikelihood(self, requests):
            raise NotImplementedError('This runner evaluates generative tasks only')

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError('Perplexity is not implemented by this transport')

        def generate_until(self, instances):
            def one(instance):
                if self.cancel.is_set():
                    raise InterruptedError('Cancelled')
                context, kwargs = instance.args
                messages = json.loads(context)
                if not isinstance(messages, list) or not messages:
                    raise ValueError('Expected harness chat messages')
                if kwargs.get('until'):
                    raise ValueError('Frozen think-enabled protocol requires until=[]')
                body = generation_body(server.alias, messages, max_tokens)
                doc_id = int(instance.doc_id)
                identity = {'model_sha256': model_hash, 'dataset_revision': revision, 'task': task_name,
                            'doc_id': doc_id, 'body': body, 'context': server.context}
                key = hashlib.sha256(canonical(identity).encode()).hexdigest()
                path = task_dir / 'raw' / (key + '.json')
                partial = task_dir / 'raw' / (key + '.partial.json')
                if path.is_file():
                    old = json.loads(path.read_text(encoding='utf-8'))
                    if old['identity'] != identity or old.get('status') != 'RECORDED':
                        raise RuntimeError('Invalid cached response')
                    return old
                session = requests.Session()
                session.trust_env = False
                session.headers['Authorization'] = 'Bearer ' + server.key
                try:
                    # Reject oversized requests; never silently truncate the prompt.
                    t = session.post(server.base + '/apply-template', json={'messages': messages,
                         'chat_template_kwargs': {'enable_thinking': True}}, timeout=30)
                    t.raise_for_status()
                    tok = session.post(server.base + '/tokenize', json={'content': t.json()['prompt'],
                          'add_special': True, 'parse_special': True}, timeout=30)
                    tok.raise_for_status()
                    if len(tok.json()['tokens']) + max_tokens > server.context:
                        raise ValueError(f'Prompt exceeds reserved context: {task_name} doc {doc_id}')
                    with session.post(server.base + '/v1/chat/completions', json=body, stream=True,
                                      timeout=(15, 120)) as response:
                        response.raise_for_status()
                        result = collect_stream(response, self.cancel,
                            lambda x: save(partial, {'identity': identity, **x}))
                    # Only final content goes to the official task scorer, never reasoning.
                    record = {'identity': identity, 'status': 'RECORDED', **result}
                    save(path, record)
                    partial.unlink(missing_ok=True)
                    return record
                except BaseException as exc:
                    save(task_dir / 'errors' / (key + '.json'), {'identity': identity, 'error': repr(exc)})
                    raise
                finally:
                    session.close()

            pool = cf.ThreadPoolExecutor(max_workers=server.workers)
            output = [None] * len(instances)
            futures = {pool.submit(one, x): i for i, x in enumerate(instances)}
            try:
                for count, f in enumerate(cf.as_completed(futures), 1):
                    record = f.result()
                    output[futures[f]] = record['final_content']
                    self.rows.append(record)
                    if count % 10 == 0 or count == len(instances):
                        trunc = sum(x['finish_reason'] == 'length' for x in self.rows)
                        tokens = sum(x['usage']['completion_tokens'] for x in self.rows)
                        print(f'{server.alias} | {task_name} | {count}/{len(instances)} | tokens={tokens} | capped={trunc}', flush=True)
                return output
            except BaseException:
                self.cancel.set()
                server.stop()
                for f in futures:
                    f.cancel()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    return GGUFAdapter()
