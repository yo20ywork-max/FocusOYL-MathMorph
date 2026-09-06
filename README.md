# FocusOYL MathMorph

### Training-free GGUF weight editing, from spectral geometry to reproducible experiments

[Model weights](https://huggingface.co/yoxia/FocusOYL-Prism-1B) · [Mathematics](docs/MATHEMATICS.md) · [Experiment history](docs/EXPERIMENT_LOG.md) · [Evaluation protocol](docs/EVALUATION.md) · [Reproduction](docs/REPRODUCIBILITY.md)

**MathMorph is the research framework. FocusOYL Prism-1B is one specific experimental checkpoint.** This repository documents the hypotheses, implementations, controls, negative results, and publication lineage behind that checkpoint. It does not claim a universal GGUF upgrade, a newly pretrained foundation model, or a proven breakthrough beyond an architecture's capability ceiling.

> **Current status: experimental; retain the original model as the default.** On one internal 156-item comparison, Prism improved observed correct delivery from 110 to 121 while using fewer generated tokens. It failed the predeclared overall acceptance criteria. A later public-task evaluation snapshot also contains a GSM8K regression; the combined evaluation remains marked incomplete. Both are documented rather than selectively omitted.

## 1. What is actually released?

| Field | Released Prism checkpoint |
|---|---|
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
\alpha=\min\!\left(1,\frac{0.98\,\varepsilon\,\lVert W\rVert_F}{\lVert D\rVert_F}\right),\qquad r=512,\quad\varepsilon=0.12.
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
h_j(x)=\operatorname{SiLU}(g_j^\top x)(u_j^\top x),\qquad y=Wh(x),
$$

we constructed a cubic Hermite surrogate under an explicit Gaussian proxy, producing the uncentered feature Gram matrix

$$
K=\mathbb{E}[\widehat h(x)\widehat h(x)^\top].
$$

A positive-definite downstream reader metric $H$ gives $C=H^{1/2}$. With $P_r$ selected from the leading eigenspace of $CWKW^\top C$, the weighted approximation is

$$
W_r=C^{-1}P_rCW\in\underset{\operatorname{rank}(Z)\leq r}{\operatorname{argmin}}\;\lVert C(Z-W)K^{1/2}\rVert_F^2.
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

### Subsequent public-task evaluation snapshot

The existing local benchmark uses `lm-evaluation-harness` task definitions for GSM8K and IFEval. The saved paired GSM8K flexible-extraction metric is **72.7066% for the original versus 70.4321% for Prism**. The aggregate file is explicitly `INCOMPLETE`; there is no certified paired final IFEval comparison. These results must not be merged with the earlier custom benchmark, and there is no public leaderboard rank or third-party verification claim. See [the evaluation notes](docs/EVALUATION.md) and `evidence/localbench/latest_incomplete/`.

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
| Post-release task evaluation | Does the published candidate retain its advantage? | Saved GSM8K comparison regressed; combined evaluation incomplete |

Read [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for the complete documented sequence and the distinction between archival results, retrospective diagnosis, and unevaluated plans.

## 6. Use and reproduce

Python 3.12 is the original Windows environment. The root converter is an explicit-path entry point for the **known Prism baseline**, not a promise to optimize arbitrary GGUF files:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe convert_prism.py --source "C:\models\MiniCPM5-1B-F16.gguf" --out "C:\models\prism-reproduction"
```

It writes to a new directory, verifies source identity, and does not publish, replace a production model, or start a paid cloud job. Exact output hashes can depend on the numerical library and floating-point implementation; compare the recorded recipe and audited changes as well as the input identity.

```bash
python -m unittest discover -s tests -v
```

These are synthetic mathematical and release-contract checks, **not fresh language-model scores**. Archived benchmark runners have their own dependencies and historical path assumptions; consult [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before executing them.

## 7. Repository map

```text
convert_prism.py          explicit-path entry point for the released recipe
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
