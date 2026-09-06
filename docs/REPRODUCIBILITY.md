# Reproducibility and safe operation

## Supported entry point

The root `convert_prism.py` reproduces the declared **Euclidean control recipe**, starting from the known original F16 GGUF. It does not auto-detect a best candidate, train, download weights, start a server, or upload results. Supply explicit local source and a new output directory.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe convert_prism.py --source 'C:\models\MiniCPM5-1B-F16.gguf' --out 'C:\models\prism-reproduction'
```

The expected baseline SHA-256 is `68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d`. Do not substitute a newer file with the same name and assume it is identical. The released candidate SHA-256 is `24f86e98d327afb5d17418486708b0dc653749d6eed67150c4f58f8c6591a41b`.

Conversion requires additional disk space approximately equal to one output GGUF plus reserve, and working RAM for the selected matrices and decomposition. Existing outputs are rejected. Only declared tensor regions may change. Use the original source, not a previously edited sharpen checkpoint. The released filename is a label; renaming must not change its content digest.

## Numerical reproducibility

The experiment used Windows Python 3.12 and NumPy 1.26.4; GGUF tooling included 0.19.0. NumPy eigensolvers and floating-point kernels may differ across BLAS builds or hardware. A repeated singular value also makes a chosen truncated subspace non-unique. The code reproduces the algorithm and enforces byte/format guards, but does not promise cross-platform bitwise identity of the new weights. Preserve library versions and compare edit measurements as well as hashes.

## Archive versus supported wrapper

`archive/` is versioned research source, not a promise that every old script is a polished cross-platform application. Source home paths have been generalized to `C:\Users\YOUR_USER`. Some scripts expect sibling study folders from the original experiment; recreate the documented data layout or supply appropriate paths. Do not execute archival runners blindly. Files not recovered from the live research folders are listed in the archive manifest.

For the old v0.4 study runners, obtain a compatible `llama-server` separately, read the accepted arguments with `--help`, and review all context, timeout, GPU, and concurrency settings. Do not use historical results as the output of a new run. The root converter's minimal requirements do not install all benchmark extras.

## Standard-task runner

The archived local benchmark needs its own isolated environment and the packages used in that release, including `lm-eval[api,ifeval]==0.4.13`. Read `start.py` and `run_benchmark.py` before running; verify model paths and SHA checks. It uses local compute but consumes electricity, disk, RAM, and GPU time. No HF Jobs credit or paid API is needed for local inference. This publication does not schedule a run.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The publication tests use small synthetic matrices. They verify the projection decomposition, error bound, energy relation, protected-subspace algebra, and release recipe contract. They do not measure language-model accuracy. Earlier engineering test counts in the experiment log refer to historical test suites, not a new re-execution of those entire suites during publication.

## Re-evaluation checklist

Record immutable source/candidate hashes, source code commit, runtime build, task revisions, exact prompts/templates, Think state, sampling policy, generation and context budgets, failures, and actual usage. Keep development and final holdout separate. Confirm that reasoning is not silently used as the final answer. Never merge scores from old sharpen, high-dimensional variants, and the published Prism checkpoint.
