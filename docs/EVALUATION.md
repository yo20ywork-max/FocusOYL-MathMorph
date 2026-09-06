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

`benchmarks/localbench/` archives the local runner using EleutherAI `lm-evaluation-harness` 0.4.13, GSM8K `gsm8k_cot_zeroshot` task version 3.0, and IFEval task version 4.0. The generator uses a local llama.cpp API. Chat generation is not a substitute for token log-likelihood scoring tasks. Task-definition versions, sampling settings, total generation limits, chat templates, and final-answer extraction must accompany any published result.

The available saved combined run is marked **INCOMPLETE**. It contains paired GSM8K results and an unpaired Prism IFEval result. GSM8K flexible extraction is 72.7066% original versus 70.4321% Prism; the raw task files retain strict extraction too. These are not a leaderboard submission, a third-party certificate, or a claim that all IFEval work finished.

No full benchmark was restarted for this publication. The snapshots are existing observations, identified by source hashes and publication timestamps. Their runtime counts may include auxiliary mode probes. Read `evidence/localbench/latest_incomplete/STATUS.json` before interpreting the aggregate.

## Evidence publication

Public files contain scores, status labels, resource counts, case identifiers or hashes, source identities, and protocol summaries. Full dataset prompts, model reasoning, personal paths, live credentials, server log dumps, and model binaries are excluded. The sanitized representation has its own digest; a local source digest is not a promise that its unredacted bytes are in this repository.

A rigorous follow-up should register a fixed model identity, compare matching Think modes and actual resource budgets, retain all failed cases, use complete recognized tasks, and obtain independent reproduction. Neither payment for cloud compute nor a repository commit verification badge verifies model capability.

Sources: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [IFEval](https://huggingface.co/datasets/google/IFEval), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
