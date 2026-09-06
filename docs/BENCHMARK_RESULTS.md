# MiniCPM5-1B F16 vs FocusOYL Prism-1B F16

## Completed benchmark report

Completed local, full-split evaluation using public benchmark task definitions. Both models use Think mode, zero-shot prompts, greedy decoding, and a common 4,096-token total generation cap. This is a self-reported experiment, not an independently verified leaderboard submission.

Published 2026-09-07 (Asia/Taipei). Integrity audit: `2026-09-06T19:47:48.942275+00:00`. Run ID: `full-think-on-4096`.

## Results

| Benchmark / metric | MiniCPM5-1B F16 | FocusOYL Prism-1B F16 | Change (pp) |
|---|---:|---:|---:|
| GSM8K: flexible numeric extraction | 70.43% (929/1,319) | 72.71% (959/1,319) | +2.27 |
| GSM8K: strict-format extraction | 0.23% (3/1,319) | 0.53% (7/1,319) | +0.30 |
| IFEval: prompt-level strict | 72.64% (393/541) | 68.58% (371/541) | -4.07 |
| IFEval: instruction-level strict | 74.70% (623/834) | 71.34% (595/834) | -3.36 |
| IFEval: prompt-level loose | 74.12% (401/541) | 71.35% (386/541) | -2.77 |
| IFEval: instruction-level loose | 75.78% (632/834) | 73.50% (613/834) | -2.28 |

Prism answers 30 more GSM8K questions correctly under flexible numeric extraction, but passes all strict IFEval instructions on 22 fewer prompts. All four IFEval metrics decline. This is a task trade-off, not an across-the-board capability upgrade.

There are 1,860 prompts per model and 3,720 scored generations across both models.

## Generation usage

| Task | Original generated tokens | Prism generated tokens | Token change | Original length stops | Prism length stops |
|---|---:|---:|---:|---:|---:|
| GSM8K | 2,379,397 | 2,178,602 | -8.44% | 258 / 1,319 | 212 / 1,319 |
| IFEval | 917,527 | 934,800 | +1.88% | 84 / 541 | 87 / 541 |

Tokens include thinking and final answers. Auxiliary mode probes are not included in these scored-request totals.

## Shared protocol

| Setting | Both models |
|---|---|
| Reasoning | Think enabled; recorded reasoning fields checked |
| Generation limit | 4,096 total generated tokens per question, including thinking and answer |
| Context | 8,192 tokens per request; context shifting disabled |
| Sampling | temperature 0; top_p 1; top_k 0; min_p 0; repeat_penalty 1 |
| Seed / repetitions | 20260906; one scored generation per question |
| Few-shot examples | 0 |
| Concurrency / GPU offload setting | 2 requests; gpu_layers=99 |
| Engine | llama.cpp build 10672, commit 511f9c137; Windows x86_64 |
| Evaluation framework | lm-evaluation-harness 0.4.13 via a local chat API adapter |
| Task definitions | gsm8k_cot_zeroshot version 3.0; ifeval version 4.0 |
| Stop policy | EOS or common token cap; custom stop strings are empty |
| Scored output | Final-answer content only; separate reasoning content never used as a substitute |

## Extraction and scoring notes

**GSM8K strict-format extraction is not a second independent mathematics test.** Both GSM8K filters score the same generations. The saved strict regex expects `The answer is` followed by a numeric answer; it failed to parse 1,316 original outputs and 1,311 Prism outputs. Its very low 0.23% / 0.53% scores therefore cannot be interpreted as overall mathematical ability. Flexible extraction uses the existing task's last-numeric-match filter. It can itself mis-extract an answer, so it is reported by its exact filter name rather than called perfect semantic grading. Neither prompts nor filters were changed after seeing these results.

**IFEval prompt-level accuracy** requires every checked instruction in a prompt to pass. **Instruction-level accuracy** sums individual passing checks across all prompts: 834 checks per model, not the unweighted mean of per-prompt percentages. Strict and loose use the task's two original matching conventions; both are disclosed.

**Truncation remains part of the result.** Length-stopped generations are not discarded from the denominator. The harness scores whatever final content was returned, including a partial final answer if one exists; empty final answers fail. This differs from the earlier internal study's additional natural-completion requirement. These full-split results must not be merged with the 156-item study.

## Console diagnostics and audit scope

The terminal displayed `sample_len` values as `131900.00%` and `54100.00%`. That is a presentation error: the underlying result files store integer counts of 1,319 and 541. Published tables separate counts from percentages. The repeated terminal report is not evidence of a second run; this publication checks 3,720 scored response records in total.

The terminal also emitted language-detection warnings for text without detectable language features and a Git-context warning. They are disclosed rather than removed from the historical run. All four task result files exist and the combined report has `complete=true`. The publication audit recomputed all six score pairs from saved item metrics, checked question/prompt/target alignment, verified identical generation settings, and matched every scored response to its recorded final content. This is an integrity and aggregation audit, not an independent rerun of inference or of the scorers.

The original and Prism GGUF files were rehashed during publication and matched the model identities recorded in the run. No weights, test prompts, scores, generation budgets, or source logs were modified. No new model inference or paid cloud job was started.

## Interpretation and limitations

The observed GSM8K flexible-extraction improvement is +2.27 percentage points, with fewer generated tokens on that task. The strict IFEval prompt metric declines by 4.07 percentage points, and its other three metrics decline too. Lower token consumption does not by itself establish a wall-clock speedup. No pooled or selectively weighted overall score is presented.

This is one fixed-checkpoint, fixed-configuration local run. No new significance or non-inferiority claim is made. Results do not establish universal improvement, a model-architecture ceiling breakthrough, coding/tool-use capability, long-context reliability, or performance on unseen benchmarks. The 4,096-token cap is an experimental condition, not an unlimited-thinking result. Pretraining contamination has not been ruled out. Independent reproduction remains outstanding; the checkpoint stays Experimental.

## Model and dataset identities

| Artifact | SHA-256 / revision |
|---|---|
| MiniCPM5-1B F16 GGUF | `68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d` |
| FocusOYL Prism-1B F16 GGUF | `24f86e98d327afb5d17418486708b0dc653749d6eed67150c4f58f8c6591a41b` |
| GSM8K `openai/gsm8k`, main/test | `740312add88f781978c0658806c59bc2815b9866` |
| IFEval `google/IFEval`, train-named evaluation split | `966cd89545d6b6acfd7638bc708b98261ca58e84` |

## Evidence and reproduction

The [complete evidence directory](https://github.com/yo20ywork-max/FocusOYL-MathMorph/tree/main/evidence/localbench/full-think-on-4096) includes `comparison.json`, four task-configuration/result snapshots, `protocol.json`, `sample_metrics.json`, diagnostics, original-file digests, and `AUDIT.json`. Item records expose metrics and identity hashes without redistributing full dataset prompts, responses, or reasoning traces. The earlier `latest_incomplete/` snapshot is retained as historical evidence, not the current status.

From the repository root, recompute the published aggregates without loading a model:

```shell
python tools/verify_completed_benchmark.py
```

This validates published records; a fresh inference reproduction still requires the pinned GGUF files and the documented evaluation environment.

Sources: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [IFEval](https://huggingface.co/datasets/google/IFEval).
