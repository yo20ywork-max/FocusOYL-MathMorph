# Evaluation protocol and evidence boundaries

## What was measured

The internal experiments measured correct final delivery under fixed resource and formatting constraints, not an abstract intelligence score. A successful internal observation required the final-answer field to be parseable and correct, with a natural completion. A correct option mentioned only inside a reasoning trace was not sufficient.

For the v0.4 internal comparison: Think on; temperature 0; top_p 1; top_k 0; min_p 0; repeat_penalty 1; seed 20260905; at most 2,048 generated tokens including reasoning and answer; 4,096-token context; four concurrent requests; 120-second per-item observation limit; no context shifting; llama.cpp b10672 / 511f9c137. This is not the upstream model's maximum attainable score under all sampling policies.

## Predeclared acceptance gates

The first v0.4 study required at least +5 percentage points overall, two-comparison adjusted paired p<0.05, known token cost no greater than 1.15 times baseline, no observed protected-category drop exceeding 2 percentage points, no increase in loop aborts or token-limit stops, and no transport/time/cancellation contamination. These gates combine finite-sample observations and a paired test; category gates are not formal non-inferiority confidence certificates.

For wins $w$ and losses $l$, the exact paired sign/McNemar-binomial two-sided calculation used

$$
p=\min\left(1,2\sum_{k=0}^{\min(w,l)}{w+l\choose k}2^{-(w+l)}\right).
$$

For two planned comparisons, the reported adjusted value is $\min(1,2p)$. No discordances gives p=1. A nonsignificant result does not prove equivalence or absence of an effect; it means the declared evidence threshold was not met.

## Development, holdout, and duplicate prompts

Weight calculations did not use task examples or labels. Development scores did influence research selection, so the end-to-end research process is not data-independent. The first holdout contained 160 requests, four of which matched a repeated development prompt. The exclusion rule was fixed before holdout inference and before candidate selection completed. Both raw 160-item and non-overlap 156-item analyses were retained. Shared procedural templates and possible pretraining overlap remain limitations.

The second study prepared 320 fresh holdout items but never evaluated them because no development candidate passed. This repository does not report scores for that unused set.

## Why the loop guard is not an improvement mechanism

A lexical guard checks persistent duplicate 12-word windows in the separate reasoning field. When it aborts a request, that observation is a failure of successful delivery, not a correct answer. It does not inject an end-of-thinking token, retrieve a gold answer, change repetition penalties, or modify the GGUF. Some aborted requests lack final usage metadata; missing token counts must not be replaced with zero or presented as a complete cost estimate.

The earlier uncapped run had a 900-second observation guard and sliding context. Censored observations are not known semantic errors. Its user-paused partial sample cannot be treated as a full representative benchmark.

## Public-task evaluation is a distinct protocol

The combined `full-think-on-4096` run is now **COMPLETE**: both models completed 1,319 GSM8K and 541 IFEval prompts, totaling 3,720 scored generations. The earlier `latest_incomplete/` snapshot remains a historical intermediate record.

### Completed public-benchmark comparison

Completed local, full-split evaluation using public benchmark task definitions. Both models use Think mode, zero-shot prompts, greedy decoding, and a common 4,096-token total generation cap. This is a self-reported experiment, not an independently verified leaderboard submission.

| Benchmark / metric | MiniCPM5-1B F16 | FocusOYL Prism-1B F16 | Change (pp) |
|---|---:|---:|---:|
| GSM8K: flexible numeric extraction | 70.43% (929/1,319) | 72.71% (959/1,319) | +2.27 |
| GSM8K: strict-format extraction | 0.23% (3/1,319) | 0.53% (7/1,319) | +0.30 |
| IFEval: prompt-level strict | 72.64% (393/541) | 68.58% (371/541) | -4.07 |
| IFEval: instruction-level strict | 74.70% (623/834) | 71.34% (595/834) | -3.36 |
| IFEval: prompt-level loose | 74.12% (401/541) | 71.35% (386/541) | -2.77 |
| IFEval: instruction-level loose | 75.78% (632/834) | 73.50% (613/834) | -2.28 |

Prism answers 30 more GSM8K questions correctly under flexible numeric extraction, but passes all strict IFEval instructions on 22 fewer prompts. All four IFEval metrics decline. This is a task trade-off, not an across-the-board capability upgrade.

Each model was evaluated on **1,319 GSM8K questions and 541 IFEval prompts**. IFEval contains **834 individual instruction checks**; those are not 834 separate prompts. Changes are calculated from unrounded scores.

The two GSM8K extraction filters score the same generated answers, not separate runs. The strict filter is highly format-sensitive; read the extraction diagnostics before interpreting its near-zero scores.

Unlike the earlier internal study, this harness protocol scores the returned final-answer field even on length-stopped outputs. It never substitutes text from the reasoning field. All denominators and failed final answers are retained.

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

The [completed report](BENCHMARK_RESULTS.md) records token usage, strict/flexible extraction, dataset and model identities, warnings, and evidence locations. The publication audit re-sums existing item metrics and checks stored responses; it does not rerun the scorers or model. No leaderboard submission or independent verification is claimed.

## Evidence publication

Public files contain scores, status labels, resource counts, case identifiers or hashes, source identities, and protocol summaries. Full dataset prompts, model reasoning, personal paths, live credentials, server log dumps, and model binaries are excluded. The sanitized representation has its own digest; a local source digest is not a promise that its unredacted bytes are in this repository.

A rigorous follow-up should register a fixed model identity, compare matching Think modes and actual resource budgets, retain all failed cases, use complete recognized tasks, and obtain independent reproduction. Neither payment for cloud compute nor a repository commit verification badge verifies model capability.

Sources: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [IFEval](https://huggingface.co/datasets/google/IFEval), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
