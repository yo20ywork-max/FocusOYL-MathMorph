# Experiment log and negative results

The underlying records date the main experiments to 2026-09-05 and the following publication/evaluation period. This document is a retrospective index, not an invented contemporaneous laboratory notebook. Raw local records remain unchanged. Public structured copies omit full prompts, generated traces, and private machine details; source hashes and publication hashes are retained in `provenance/ARCHIVE_MANIFEST.json`.

## Precursor: quantization-channel geometry

The first hypothesis represented each linear channel as an outer-product feature and estimated a noise covariance from quantization scales. A 700-case synthetic study tested several operator structures. Some heterogeneous cases improved, while exactly representable operators were harmed by nonzero noise estimates. This was synthetic reconstruction evidence, not language-model capability evidence. The precursor's conditional risk expression and counterexample are documented in the mathematics report; the real-GGUF implementation subsequently used different candidate operators.

## v0.1: real GGUF conversion and GPEP

`archive/v0.1/` contains the initial reader/writer, proxy geometry, immutable-file checks, and evaluation machinery. A MiniCPM5 F16 candidate and a MiniMind Q4_0 candidate were generated. The 24-item smoke screen gave MiniCPM5 8/24 before and after. MiniMind scored 0/24 with answers truncated at the short 96-token cap. The result established the plumbing, not a stronger model. Early quantizer invocation issues were corrected with explicit type checks; file creation was not counted as capability improvement.

## v0.2: NPSR and apparent arithmetic gains

NPSR redistributed energy between a projected subspace and its complement while preserving the Frobenius norm in ideal arithmetic. The broad experiment modified attention-output and FFN-down matrices across the 24 blocks. On 264 mixed items, MiniCPM5 scored 145 originally, 71 for flatten, and 133 for sharpen; MiniMind scored 5, 6, and 5 respectively.

An arithmetic follow-up froze the sharpen artifact before new confirmation questions. On 96 multiplication/subtraction items, correct delivery increased from 37 to 90, with 56 paired wins and 3 losses. The candidate consumed 5,531 output tokens versus 575 for the original. On 48 composed arithmetic items the scores were 11 and 46. When both versions were explicitly asked to work step by step, a 48-item control was 47 versus 48. These results support altered use of existing capability, not a demonstrated increase in intrinsic knowledge or equal-compute reasoning power.

The original development/parser results were not rewritten to improve the story. Later parser changes were accompanied by new confirmation questions. See `evidence/v0.2/`.

## Think-mode controls

Three conditions used the same inputs: original Think on, old sharpen Think on, and old sharpen Think off. On the 264-item mixed set their scores were 214, 168, and 158. Across 456 scored requests per arm, total generation was 212,828, 417,972, and 25,578 tokens; token-limit stops were 19, 91, and 1. The common total generation cap was 2,048 tokens.

This control changed the interpretation of the earlier arithmetic advantage. The original with native thinking already approached the arithmetic ceiling. More generated reasoning did not reliably mean more successful delivery. These records belong to **old NPSR sharpen**, not to the public Prism artifact. See `evidence/think_modes/`.

## Unlimited-generation attempt and pause audit

The next experiment removed the generation-token cap but retained an 8,192-token sliding context, eight concurrent requests, and a 900-second per-item observation limit. This was not infinite compute or unlimited retained history.

The original arm recorded all 456 observations: 455 natural completions, one censored observation, and 410 correct deliveries. When the user paused the sharpen arm, 159 observations had been recorded: 122 natural completions, 37 censored observations, and 89 correct deliveries. Eight requests were active and 289 had not started. The completion order was not a random sample, so 89/159 is not a final model accuracy estimate.

All 37 censored traces lacked a final answer. In their final text windows, 36 had at least 50% duplicate 12-word windows; the median was 74.60%, versus 1.95% among 122 naturally completed traces. Some repeatedly announced a final option without emitting it; others repeated invalid objections or merely restated the question. Lexical repetition is not a direct percentage of semantic uselessness.

The experiment was stopped on request; active traces were saved and marked user-interrupted. Dedicated inference processes were closed and their KV allocations released. The records do not prove that all longer thinking is useless, or identify a unique mechanistic cause. See `evidence/unlimited/AUDIT_SUMMARY.json`.

## v0.3: local row-tangent editing and release gates

The next candidate started again from the unmodified original and edited only FFN-down tensors in blocks 8 and 15. It preserved individual row norms in ideal arithmetic and enforced row/tensor update budgets after encoding. Actual tensor changes were approximately 0.7503% each.

On a 24-item scoped regression screen, original and new candidate both scored 22, while old sharpen scored 18. The new candidate used 7,815 tokens versus 7,034 for the original. All original/new observations naturally completed; old sharpen had four loop-guard aborts and one token-limit stop. Recovery to baseline was not classified as a breakthrough.

A Q4 attempt was rejected because post-encoding row-norm drift exceeded the fixed threshold. The loop guard also detected 31 of 37 old censored traces in retrospective replay, with no triggers among the 122 old natural completions. That replay is not an independent detector-generalization test, and the external guard does not become a model ability. See `evidence/v0.3/`.

## v0.4, first study: nonlinear feature geometry versus Euclidean control

Four recipes were fixed: cubic last-layer mild, cubic last-layer stronger, cubic cross-layer, and Euclidean control. On 48 development requests, original scored 38; all three geometric candidates scored 39; the Euclidean control scored 37. The prescribed tie-break selected the less costly cubic mild candidate. The Euclidean arm was a preplanned comparison, not introduced after seeing held-out outcomes.

Before held-out inference, a prompt audit found repeated procedural `range(4)` cases shared with development. All four matching held-out requests were declared ineligible for certification before selection completed and before held-out inference. The 160 raw requests remain; the non-overlap analysis contains 156.

| Group | Original | Cubic mild | Euclidean / Prism |
|---|---:|---:|---:|
| ARC-Challenge subset | 38/64 | 41/64 | 48/64 |
| ARC-Easy subset | 49/64 | 49/64 | 50/64 |
| Synthetic arithmetic | 15/16 | 15/16 | 16/16 |
| Synthetic code reading | 8/12 | 8/12 | 7/12 |
| Total | 110/156 | 113/156 | 121/156 |

The Euclidean control's 17 wins and 6 losses gave raw paired two-sided p=0.0346897, or 0.0693793 after the two-comparison correction. It used fewer tokens but regressed in the code category. No complete acceptance pass was issued. The stored `FINAL_DECISION` is not retroactively changed just because a candidate was named and published as experimental.

## v0.4, second study: strong projections and readout protection

The follow-up used 47 deduplicated development items and six declared variants. Scores were: original 37; cubic rank128 full 33; cubic rank128 protected 30; Euclidean rank512 protected 36; Euclidean rank512 unprotected 36; Euclidean rank32 protected 24; cubic rank32 protected 24.

The strongest projections changed much of **one final FFN-down matrix**, not 90% of the whole model. The protected output-contrast basis had rank 116 in this checkpoint. Low measured leakage did not establish preserved autoregressive behavior. None qualified for advancement, so the separately prepared 320-item holdout was not evaluated. The complete v0.4 program recorded 1,049 scored requests.

An initial tokenizer-array read failed before model scoring; the reader fallback was fixed without changing recipes or questions. Failed attempts remain in the local provenance. See `evidence/v0.4/`.

## Publication: FocusOYL Prism-1B

The Euclidean control was named FocusOYL Prism-1B F16 Experimental-01. It was not relabeled as NPSR sharpen or as the full high-dimensional operator. Weights were hosted on Hugging Face; documentation was subsequently translated into English. Gating and model-card language are hosting settings, not evidence about model intelligence. The current canonical account is `yoxia`; older records retain the historical `yooo19y` identifier.

## Subsequent standard-task run: incomplete combined report

The local benchmark setup uses the actual `lm-evaluation-harness` GSM8K and IFEval task definitions, with a documented local API adapter and common 4,096-token generation setting. During this publication task, an existing saved run was found. **No new model inference was started for publication.**

Its aggregate report is `INCOMPLETE`. The stored paired GSM8K flexible-extraction score is 0.7270659591 for original and 0.7043214556 for Prism, a difference of -2.27445 percentage points. Strict-extraction scores are near zero, which must remain visible as a formatting/extraction distinction rather than being silently replaced. A Prism IFEval result exists, but a completed paired original IFEval result is not certified in the available aggregate.

We publish sanitized result snapshots and per-item metric/hash data, not a fabricated completed comparison. Runtime request counts can include auxiliary probes and must not be treated as the number of benchmark questions. See `evidence/localbench/latest_incomplete/` and the evaluation report.

## Interpretation

The experiments support a narrow conclusion: weight-only editing changes behavior and can improve some observed subsets, while also causing regressions, inefficiency, or failure to terminate. This archive does not establish stable global improvement, a universal converter, or a new mathematical law of intelligence. Negative results are first-class outputs of the research.
