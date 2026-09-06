"""Synthetic transport and official-scoring tests. No GGUF inference or real benchmark score."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from benchlib import collect_stream, generation_body, save


class FakeStream:
    def __init__(self, rows):
        self.rows = rows
    def iter_lines(self, **unused):
        for obj in self.rows:
            yield b'data: ' + json.dumps(obj).encode()
        yield b'data: [DONE]'


class TransportTests(unittest.TestCase):
    def test_separate_reasoning(self):
        stream = FakeStream([
            {'choices': [{'delta': {'reasoning_content': 'Do not grade this: 99'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': 'The answer is 7.'}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': {'completion_tokens': 15}},
        ])
        r = collect_stream(stream, threading.Event(), lambda x: None)
        self.assertEqual(r['final_content'], 'The answer is 7.')
        self.assertIn('99', r['reasoning_content'])
        self.assertEqual(r['usage']['completion_tokens'], 15)

    def test_truncated_thinking_is_not_final(self):
        r = collect_stream(FakeStream([
            {'choices': [{'delta': {'reasoning_content': 'The answer is 7.'}, 'finish_reason': 'length'}]},
            {'choices': [], 'usage': {'completion_tokens': 4096}}]), threading.Event(), lambda x: None)
        self.assertEqual(r['final_content'], '')
        self.assertEqual(r['finish_reason'], 'length')

    def test_missing_usage_rejected(self):
        with self.assertRaises(RuntimeError):
            collect_stream(FakeStream([{'choices': [{'delta': {'content': '7'}, 'finish_reason': 'stop'}]}]), threading.Event(), lambda x: None)

    def test_cancel(self):
        event = threading.Event(); event.set()
        with self.assertRaises(InterruptedError):
            collect_stream(FakeStream([{'choices': []}]), event, lambda x: None)

    def test_payload(self):
        b = generation_body('prism', [{'role': 'user', 'content': 'hello'}], 4096)
        self.assertEqual(b['model'], 'prism')
        self.assertEqual(b['max_tokens'], 4096)
        self.assertTrue(b['chat_template_kwargs']['enable_thinking'])
        self.assertNotIn('api_key', b)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a' / 'record.json'
            save(p, {'a': 7})
            self.assertEqual(json.loads(p.read_text()), {'a': 7})
            self.assertFalse(p.with_suffix('.json.writing').exists())

    def test_official_task_scorers_with_synthetic_data(self):
        # Exercise the installed harness without downloading evaluation questions or loading a model.
        import lm_eval
        from lm_eval.api.model import LM
        from run_benchmark import load_config, evaluate_task
        class Stub(LM):
            @property
            def tokenizer_name(self): return 'synthetic-selftest-only'
            def apply_chat_template(self, history, add_generation_prompt=True, **kwargs): return json.dumps(history)
            def loglikelihood(self, requests): raise NotImplementedError
            def loglikelihood_rolling(self, requests): raise NotImplementedError
            def generate_until(self, requests): return ['The answer is 7.'] * len(requests)
        base = Path(lm_eval.__file__).parent / 'tasks'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gsm = d / 'gsm.jsonl'
            gsm.write_text(json.dumps({'question': 'Synthetic pipeline check: three plus four?', 'answer': '#### 7'}) + '\n', encoding='utf-8')
            cfg = load_config('gsm8k_cot_zeroshot', base, 'main', 4096)
            cfg['dataset_path'] = 'json'; cfg['dataset_name'] = None
            cfg['dataset_kwargs'] = {'data_files': {'test': str(gsm), 'train': str(gsm)}}
            r = evaluate_task(Stub(), cfg)
            self.assertEqual(r['results']['gsm8k_cot_zeroshot']['exact_match,flexible-extract'], 1.0)
            ife = d / 'ife.jsonl'
            ife.write_text(json.dumps({'key': 1, 'prompt': 'Synthetic pipeline check. Include the word answer.',
                 'instruction_id_list': ['keywords:existence'], 'kwargs': [{'keywords': ['answer']}]}) + '\n', encoding='utf-8')
            cfg = load_config('ifeval', base, 'main', 4096)
            cfg['dataset_path'] = 'json'; cfg['dataset_name'] = None
            cfg['dataset_kwargs'] = {'data_files': {'train': str(ife)}}
            r = evaluate_task(Stub(), cfg)
            self.assertEqual(r['results']['ifeval']['prompt_level_strict_acc,none'], 1.0)


def run_tests():
    print('SELF-TEST: synthetic data only. No actual model score will be produced.', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransportTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    run_tests()
