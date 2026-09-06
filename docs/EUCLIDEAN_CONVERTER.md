# Euclidean projection converter

`convert_euclidean.py` is a configurable **experimental entry point**, not a new mathematical method. It delegates weight editing and byte auditing to the unchanged `archive/v0.4/convert_geometry.py`. It does not download models, run inference, train, publish, or start cloud jobs.

Publisher on Hugging Face: [yoxia](https://huggingface.co/yoxia). Code repository: [yo20ywork-max/FocusOYL-MathMorph](https://github.com/yo20ywork-max/FocusOYL-MathMorph). These are different platform identifiers.

## Choose the correct entry point

| Entry point | Purpose | Source identity | Parameters |
|---|---|---|---|
| `convert_prism.py` | Reproduce the published Prism recipe | Mandatory known baseline SHA-256 | Fixed block 23, rank 512, budget 0.12 |
| `convert_euclidean.py` | Explore Euclidean edits on structurally supported inputs | Optional expected SHA-256; the core records the actual hash | Explicit layer selection, rank, and per-tensor budget |

The fixed entry point and the archived source remain unchanged. Using the experimental entry with the Prism parameters does not turn another source model into the published Prism checkpoint. Floating-point libraries can also affect output byte identity.

## Install once

Clone or download the **entire repository**, not just the entry-point file. From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The editing path uses the CPU and NumPy. It does not require CUDA or a Hugging Face Jobs balance. Matrix sizes still determine memory and runtime.

## Check before converting

```powershell
.\.venv\Scripts\python.exe convert_euclidean.py --source "C:\models\MiniCPM5-1B-F16.gguf" --layers 23 --rank 512 --budget 0.12 --dry-run
```

`--dry-run` parses the file layout and checks the architecture, targets, storage types, parameter ranges, and conservative workspace estimate. It prints a JSON plan and creates no candidate or output directory. It does **not** inspect tensor values, run the eigendecomposition, load the model in an inference engine, or certify model quality. Supplying `--expected-sha256` additionally reads and hashes the source.

## Create a separate candidate

```powershell
.\.venv\Scripts\python.exe convert_euclidean.py --source "C:\models\MiniCPM5-1B-F16.gguf" --out "C:\models\euclidean-experiment-01" --layers 23 --rank 512 --budget 0.12
```

The output directory must not already exist. The source is never overwritten. To modify several layers, use `--layers "21,23"`; the budget applies **separately to each selected matrix**, not to the complete network. To select only the final block, use `--layers last`, which is the default.

For the exact published baseline, pin the input identity:

```powershell
.\.venv\Scripts\python.exe convert_euclidean.py --source "C:\models\MiniCPM5-1B-F16.gguf" --out "C:\models\euclidean-pinned-01" --layers 23 --rank 512 --budget 0.12 --expected-sha256 68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d
```

These are path examples, not automatic model downloads. Use your actual original file, not an already edited candidate, when comparing with a baseline.

## Operation and limits

For each selected down-projection matrix $W$, define the leading left singular subspace projector $P_r=U_rU_r^\top$ and residual $D=(I-P_r)W$:

$$
W'=P_rW+(1-\alpha)(I-P_r)W,
\qquad
\alpha=\min\left(1,\frac{0.98\,\varepsilon\|W\|_F}{\|D\|_F}\right).
$$

The zero-residual limit is the original matrix; ineffective encoded edits are rejected by the core. This entry always sets `geometry=False` and does not apply the cubic feature metric, downstream reader weighting, or readout protection. The mathematical formula and numerical implementation are the existing Euclidean control, not an additional invention.

Accepted inputs are limited to single-file, little-endian GGUF v2/v3 with compatible dense `llama`, `qwen2`, or `qwen3` metadata and selected `blk.<index>.ffn_down.weight` matrices. Selected tensors must use **F16, F32, or BF16** storage. Architecture labels are structural checks, not proof of behavioral compatibility with every model in those families.

MoE metadata, sharded files, selected biased FFNs, missing or non-matrix targets, quantized target writes, invalid ranks, and unsupported workspace sizes are rejected. Other unedited tensors are copied byte-for-byte; the entire file does not have to use one storage type. A mixed GGUF still requires each selected target to satisfy the native-floating-point rule.

`--rank` must be positive and **strictly below the smaller selected matrix dimension**, with no silent clamping. `--budget` must be finite and in `(0, 0.30]`. Shape-based workspace checks are conservative estimates, not a guarantee that a machine has enough free RAM. BF16/F16 rounding may exceed a requested very small budget; that is a rejection, not permission to bypass the guard.

## Output and audit

A successful conversion produces:

```text
candidate.gguf       separate weight-edited file
plan.json            archived-core recipe and source identity
report.json          candidate hash, encoded changes, and full-file byte audit
entrypoint.json      experimental CLI identity and explicit Euclidean settings
```

The historical core uses a generic method-family label in its report. Read `recipe.geometry=false`, the selected tensor records, and `entrypoint.json` to identify this Euclidean path. No historical report schema or archived implementation was silently rewritten.

On a numerical failure, the core records `REJECTED.json` and removes its partial GGUF. After a failure or interruption, inspect the output directory and choose a **new** destination before retrying. The entry does not delete your existing directories to make a rerun succeed.

## Tests and interpretation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The additional CLI tests use tiny synthetic GGUF fixtures for native float encoding, byte preservation, parameter validation, and rejection behavior. They are not usable language models and produce no language-model benchmark scores.

**Conversion success means an audited candidate was written, not that it became more capable.** Compare original and candidate with the same engine, prompts, thinking mode, generation budget, and scorers. Keep development selection separate from final evaluation. Current research status and historical results remain in [EVALUATION.md](EVALUATION.md); this entry-point release adds no new capability claim.
