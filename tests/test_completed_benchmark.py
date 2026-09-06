"""Publication evidence checks, not new language-model evaluations."""
import copy
import json
import unittest
from pathlib import Path
from tools.verify_completed_benchmark import verify_data
ROOT = Path(__file__).resolve().parents[1]/'evidence/localbench/full-think-on-4096'
class CompletedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads((ROOT/'comparison.json').read_text(encoding='utf-8'))
        self.rows = json.loads((ROOT/'sample_metrics.json').read_text(encoding='utf-8'))
    def test_published_aggregates(self):
        self.assertEqual(verify_data(self.result,self.rows)['model_metric_values_checked'],12)
    def test_counts_are_not_percentages(self):
        self.result['sample_counts_per_model']['ifeval'] = 54100
        with self.assertRaises(ValueError): verify_data(self.result,self.rows)
    def test_missing_item_is_rejected(self):
        self.rows['ifeval'].pop()
        with self.assertRaises(ValueError): verify_data(self.result,self.rows)
    def test_wrong_score_is_rejected(self):
        self.result['metrics'][0]['prism']['score'] = 1.0
        with self.assertRaises(ValueError): verify_data(self.result,self.rows)
    def test_instruction_denominator_is_not_prompt_count(self):
        row = next(m for m in self.result['metrics'] if m['metric']=='inst_level_strict_acc,none')
        row['original']['denominator'] = 541
        with self.assertRaises(ValueError): verify_data(self.result,self.rows)
if __name__ == '__main__':
    unittest.main()
