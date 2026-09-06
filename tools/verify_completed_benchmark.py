"""Recompute published aggregates from sanitized item metrics; no model or network."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def verify_data(result: dict, records: dict) -> dict:
    require(result.get('complete') is True, 'The report is incomplete')
    expected = result['sample_counts_per_model']
    require(expected == {'gsm8k_cot_zeroshot': 1319, 'ifeval': 541}, 'Invalid prompt counts')
    for task, count in expected.items():
        rows = records[task]
        require(len(rows) == count, f'{task}: missing item records')
        require(sorted(r['doc_id'] for r in rows) == list(range(count)), f'{task}: duplicate or missing IDs')
        for arm in ('original', 'prism'):
            runtime = [r[arm]['runtime'] for r in rows]
            cost = result['cost'][arm][task]
            require(sum(x['completion_tokens'] for x in runtime) == cost['completion_tokens'], 'Token total mismatch')
            require(sum(x['finish_reason'] == 'length' for x in runtime) == cost['length_stops'], 'Truncation total mismatch')
            require(sum(x['finish_reason'] == 'stop' for x in runtime) == cost['natural_stops'], 'Natural-stop total mismatch')
            require(cost['requests'] == count, 'Request count mismatch')
    require(len(result['metrics']) == 6, 'Expected all six metrics')
    checked = 0
    for m in result['metrics']:
        for arm in ('original', 'prism'):
            values = [r[arm]['metrics'][m['metric']] for r in records[m['task']]]
            flat = [x for group in values for x in group] if isinstance(values[0], list) else values
            require(all(v in (0, 1, False, True) for v in flat), 'Invalid indicator')
            expected_score = m[arm]
            require(sum(flat) == expected_score['correct'], 'Correct-count mismatch')
            require(len(flat) == expected_score['denominator'], 'Metric denominator mismatch')
            require(math.isclose(sum(flat)/len(flat), expected_score['score'], abs_tol=1e-12), 'Score mismatch')
            checked += 1
        delta = 100 * (m['prism']['score'] - m['original']['score'])
        require(math.isclose(delta, m['delta_percentage_points'], abs_tol=1e-12), 'Delta mismatch')
    require(result['total_scored_generations'] == 3720, 'Total generation count mismatch')
    return {'passed': True, 'model_metric_values_checked': checked, 'scored_generations': 3720, 'new_inference': False}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path(__file__).resolve().parents[1]/'evidence/localbench/full-think-on-4096')
    args = parser.parse_args()
    try:
        result = json.loads((args.evidence/'comparison.json').read_text(encoding='utf-8'))
        records = json.loads((args.evidence/'sample_metrics.json').read_text(encoding='utf-8'))
        print(json.dumps(verify_data(result, records), indent=2))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f'Verification failed: {exc}\n')
if __name__ == '__main__':
    main()
