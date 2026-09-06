# Data, licensing, and publication boundaries

This repository is a curated account of the complete documented research progression, not a mirror of the author's home directory. It includes core source for v0.1-v0.4, evaluation runners, compact results, protocol and failure summaries, mathematical derivations, and provenance hashes.

## Intentionally excluded

Model weights and runtime binaries belong outside GitHub. Credentials, private keys, `.env` files, virtual environments, shared caches, raw server output, full benchmark prompts, and complete generated reasoning/answers are not published. Original local research records and source model files are not modified by publication.

Structured snapshots remove free-text prompt/response fields and generalize private home-directory paths. Numeric results, expected/parsed answer codes where already present, case identifiers, hashes, usage, status, and configuration are retained. Any transformed copy has a separate published SHA-256, with its original source hash recorded when available. The public archive is not falsely described as the exact raw byte-for-byte evaluation logs.

Missing historical files are recorded in `provenance/ARCHIVE_MANIFEST.json`; their contents or successful completion are not reconstructed by guessing. The earliest synthetic prototype is described mathematically in the documentation; the real-GGUF source archive begins at v0.1.

## Dataset and dependency rights

ARC is attributed to AllenAI; its source revision in the main experiment is `210d026faf9955653af8916fad021475a3f00453`. The dataset page supplies its applicable licensing terms. GSM8K and IFEval retain their upstream terms. This repository does not relicense third-party datasets as Apache-2.0 or reproduce complete upstream question sets. Reproduction scripts obtain data from the upstream sources.

The base model is OpenBMB MiniCPM5-1B, whose applicable model license and notices must accompany derivatives. `LICENSE` applies to this research code and documentation except separately licensed material; dependencies are not vendored wholesale. Preserve any existing notices in archived source. Names identify provenance and do not imply endorsement.

## Public interpretation

A public GitHub repository, an English model card, rendered equations, a valid file hash, or a verified commit does not certify capability. Experimental results are limited to their stated checkpoints and protocols. Please report reproduction findings with configuration and failure cases, not only the highest score.
