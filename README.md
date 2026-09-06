# FocusOYL MathMorph

### Training-free GGUF weight editing, from spectral geometry to reproducible experiments

[Model weights](https://huggingface.co/yoxia/FocusOYL-Prism-1B) · [Mathematics](docs/MATHEMATICS.md) · [Experiment history](docs/EXPERIMENT_LOG.md) · [Completed benchmarks](docs/BENCHMARK_RESULTS.md) · [Evaluation protocol](docs/EVALUATION.md) · [Reproduction](docs/REPRODUCIBILITY.md) ? [Euclidean converter](docs/EUCLIDEAN_CONVERTER.md)

**MathMorph is the research framework. FocusOYL Prism-1B is one specific experimental checkpoint.** This repository documents the hypotheses, implementations, controls, negative results, and publication lineage behind that checkpoint. It does not claim a universal GGUF upgrade, a newly pretrained foundation model, or a proven breakthrough beyond an architecture's capability ceiling.

> **Current status: completed local GSM8K + IFEval evaluation; checkpoint remains Experimental.** Prism improves GSM8K flexible-extraction accuracy from 70.43% to 72.71%, while IFEval prompt-level strict accuracy decreases from 72.64% to 68.58%. All six metric pairs are disclosed below. This is not a universal upgrade or independently verified leaderboard result. The earlier internal study still failed its original acceptance criteria.

## 1. What is actually released?

| Field | Released Prism checkpoint |
|---|---|
| Hugging Face publisher | [yoxia](https://huggingface.co/yoxia) |
| Base model | [OpenBMB / MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) |
| Artifact | `FocusOYL-Prism-1B-F16-Experimental-01.gguf` |
| Recipe | MathMorph v0.4 `euclid_control` |
| Changed tensor | `blk.23.ffn_down.weight`, zero-based block numbering |
| Projector dimension | 512 |
| Relative update budget | 0.12, with a 0.98 pre-encoding reserve |
| Representation | F16 GGUF; 2,166,551,936 bytes |
| Additional training | None: no gradient training, LoRA, distillation, or teacher model |
| Inference changes | No extra layers, parameters, retrieval system, or external agent |
| Unchanged locally | Other tensor bytes, tokenizer, chat template, layout, and file length |

**Prism is not the earlier NPSR sharpen checkpoint. It does not use the cubic-feature metric or the readout-protection operator below.** The word `rank` describes an internal projector, not a claim that the whole language model has rank 512.

## 2. The formula used by Prism

Let $W\in\mathbb{R}^{m\times n}$ be the selected FFN down matrix. Its singular value decomposition defines the leading left singular subspace:

$$
W=U\Sigma V^\top,\qquad P_r=U_rU_r^\top,\qquad P_r^2=P_r=P_r^\top.
$$

Split the matrix into retained and residual components:

$$
W=P_rW+(I-P_r)W,\qquad D=(I-P_r)W.
$$

Prism applies **partial residual shrinkage**, not unconditional rank truncation:

$$
\boxed{W'=W-\alpha D=P_rW+(1-\alpha)(I-P_r)W}
$$

$$
\alpha=\min\left(1,\frac{0.98  \varepsilon  \lVert W\rVert_F}{\lVert D\rVert_F}\right),\qquad r=512,\quad\varepsilon=0.12.
$$

The zero-residual case returns the original matrix. The pre-encoding bound follows directly:

$$
\frac{\lVert W'-W\rVert_F}{\lVert W\rVert_F}\leq0.98\varepsilon.
$$

For an exact singular projector, the singular values obey:

$$
\sigma_i'=
\begin{cases}
\sigma_i,&i\leq r,\\
(1-\alpha)\sigma_i,&i>r.
\end{cases}
$$

For $\alpha<1$, rank is preserved in exact arithmetic. At $\alpha=1$, rank can decrease. F16 re-encoding is a separate numerical operation and is audited afterward. A weight-space error bound is **not** a task-accuracy or stopping-behavior guarantee.

## 3. The broader geometric research program

The research progressed beyond isolated weight magnitudes into three interacting spaces: nonlinear channel features, downstream readout geometry, and protected logit contrasts. These branches are retained even where they failed.

For a gated FFN,

$$
h_j(x)=\mathrm{SiLU}(g_j^\top x)(u_j^\top x),\qquad y=Wh(x),
$$

we constructed a cubic Hermite surrogate under an explicit Gaussian proxy, producing the uncentered feature Gram matrix

$$
K=\mathbb{E}[\widehat h(x)\widehat h(x)^\top].
$$

A positive-definite downstream reader metric $H$ gives $C=H^{1/2}$. With $P_r$ selected from the leading eigenspace of $CWKW^\top C$, the weighted approximation is

$$
W_r=C^{-1}P_rCW\in\underset{\mathrm{rank}(Z)\leq r}{\arg\min}\quad \lVert C(Z-W)K^{1/2}\rVert_F^2.
$$

A separate branch restricts the update to the orthogonal complement of a fixed readout-contrast subspace, whose orthonormal basis is $Q_A$:

$$
\boxed{W'=W-\alpha(I-Q_AQ_A^\top)\left[W-H^{-1/2}P_rH^{1/2}W\right].}
$$

**This boxed expression is an experimental family, not the released Prism recipe.** Prism is the specialization $K=I$, $H=I$, and $Q_AQ_A^\top=0$. The full derivation, assumptions, limitations, and conditional readout-invariance argument are in [MATHEMATICS.md](docs/MATHEMATICS.md). More dimensions did not automatically produce a better model in these experiments.

## 4. Results, including the failures

### Internal held-out comparison: v0.4

The original 160 requests are retained in the evidence. Four duplicate procedural prompts were identified and excluded from certification **before** held-out inference. The resulting 156 non-overlapping items were evaluated with Think enabled, greedy decoding, and a common 2,048-token total generation cap.

| Metric | Original F16 | Cubic-geometry candidate | Euclidean control / Prism |
|---|---:|---:|---:|
| Correct final delivery | 110 / 156 | 113 / 156 | 121 / 156 |
| Accuracy | 70.51% | 72.44% | 77.56% |
| Generated tokens | 107,200 | 108,138 | 97,038 |
| Token-limit terminations | 17 | 17 | 11 |
| Paired wins / losses vs. original | N/A | 11 / 8 | 17 / 6 |
| Two-comparison adjusted p-value | N/A | 1.0 | 0.06938 |

Prism's observed gain was **7.05 percentage points** with **9.48% fewer tokens**, but its code subset declined from 8/12 to 7/12. It therefore failed both the adjusted significance threshold and the protected-category criterion. These are custom generated-answer subsets, not official complete ARC leaderboard scores.

### Stronger projection was not consistently better

In a separate 47-item development screen, the original scored 37. The six stronger/readout-protected variants scored 33, 30, 36, 36, 24, and 24. No candidate qualified for advancement; the prepared 320-item holdout was **not evaluated**. The full v0.4 program performed 1,049 scored requests, not 1,049 independent tasks.

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

The GSM8K strict-format filter is a formatting-sensitive diagnostic, not an independent mathematics test. See [full results, token usage, and extraction notes](docs/BENCHMARK_RESULTS.md). The earlier incomplete snapshot remains archived; the completed comparison is in `evidence/localbench/full-think-on-4096/`.

## 5. Research timeline

| Stage | Question | Outcome |
|---|---|---|
| Quantization-channel prototype | Can cross-channel noise estimates improve an operator without training? | Synthetic improvements and explicit counterexamples; not a language-model gain certificate |
| v0.1 GPEP | Can proxy-path projection improve a real GGUF? | Pipeline ran; small smoke tests did not show improvement |
| v0.2 NPSR | Does energy-preserving spectral redistribution help? | Arithmetic gains were accompanied by much longer output; mixed-task regression |
| Think-mode controls | Is the gain still present when the base model thinks? | Original Think mode performed better overall |
| Unlimited-generation diagnosis | Do very long traces continue making useful progress? | Many censored traces repeated statements without producing a final answer |
| v0.3 bounded local edits | Can damage and runaway evaluation be reduced? | Scoped regression recovered baseline, not a capability gain |
| v0.4 geometric / Euclidean comparison | Do richer feature and reader geometries select better edits? | Euclidean control gave the strongest internal observed total; no full acceptance pass |
| Post-release task evaluation | Does the published candidate retain its advantage? | Completed paired GSM8K and IFEval full splits; GSM8K gains and instruction-following regressions |

Read [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for the complete documented sequence and the distinction between archival results, retrospective diagnosis, and unevaluated plans.

## 6. Use and reproduce

### A. Reproduce the published Prism

Python 3.12 is the original Windows environment. `convert_prism.py` is the fixed entry point for the **known Prism baseline**, not a promise to optimize arbitrary GGUF files:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe convert_prism.py --source "C:\models\MiniCPM5-1B-F16.gguf" --out "C:\models\prism-reproduction"
```

It writes to a new directory, verifies source identity, and does not publish, replace a production model, or start a paid cloud job. Exact output hashes can depend on the numerical library and floating-point implementation; compare the recorded recipe and audited changes as well as the input identity.

### B. Run an explicit Euclidean experiment

`convert_euclidean.py` shares the unchanged v0.4 core but exposes layer selection, rank, and the per-tensor update budget. It always disables high-dimensional geometry and readout protection. Start with a metadata-only preflight:

```powershell
.\.venv\Scripts\python.exe convert_euclidean.py --source "C:\models\MiniCPM5-1B-F16.gguf" --layers 23 --rank 512 --budget 0.12 --dry-run
```

Then create a separate candidate in a new directory:

```powershell
.\.venv\Scripts\python.exe convert_euclidean.py --source "C:\models\MiniCPM5-1B-F16.gguf" --out "C:\models\euclidean-experiment-01" --layers 23 --rank 512 --budget 0.12
```

Use `--layers last` or an explicit list such as `--layers "21,23"`. `--expected-sha256` optionally pins the source. Only supported dense GGUF layouts and native floating-point target tensors are accepted; **this is not an arbitrary-GGUF or guaranteed-improvement converter**. Read the [usage guide and rejection rules](docs/EUCLIDEAN_CONVERTER.md).

### C. Run synthetic checks

```bash
python -m unittest discover -s tests -v
```

These are synthetic mathematical and release-contract checks, **not fresh language-model scores**. Archived benchmark runners have their own dependencies and historical path assumptions; consult [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before executing them.

## 7. Repository map

```text
convert_prism.py          fixed, source-pinned entry for the released recipe
convert_euclidean.py      configurable Euclidean experiments with a dry-run preflight
archive/v0.1/            GPEP and GGUF read/write machinery
archive/v0.2/            NPSR and arithmetic confirmation experiments
archive/v0.3/            bounded row-tangent edits and stability screens
archive/v0.4/            cubic geometry, reader metrics, and protected projections
benchmarks/              think-mode and standard-task evaluation runners
evidence/                sanitized scores, conditions, decisions, and diagnostics
provenance/              source and publication hashes; availability notes
docs/                    derivations, timeline, evaluation, and reproduction
tests/                   small synthetic publication checks
```

The archive is a **curated research publication**, not a binary dump of a working computer. Model weights, credentials, full prompt/response traces, executable runtimes, and private machine paths are not published here. Original local records remain intact. Structured evidence is sanitized and its source/publication hashes are recorded; some historical local files were unavailable and are listed, not fabricated. See [DATA_AND_PRIVACY.md](docs/DATA_AND_PRIVACY.md).

## 8. Lineage, license, and scope

The upstream model is OpenBMB's MiniCPM5-1B. This repository and the derived research release use Apache-2.0 subject to retained applicable upstream notices. Third-party datasets and dependencies retain their own licenses. No upstream institution endorses the reported results.

The expected released artifact SHA-256 is

```text
24f86e98d327afb5d17418486708b0dc653749d6eed67150c4f58f8c6591a41b
```

The original comparison artifact SHA-256 is

```text
68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d
```

**Weight editing is not equivalent to learning new knowledge. A successful conversion is not a successful benchmark, and a higher observed score is not a universal capability theorem.**

References: [LASER](https://arxiv.org/abs/2312.13558), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [llama.cpp](https://github.com/ggml-org/llama.cpp), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
