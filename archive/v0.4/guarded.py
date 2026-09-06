"""Guarded, local, data-free weight surgery. Structural bounds are not capability bounds."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse, hashlib, json, math, shutil, sys, tempfile, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import numpy as np
from mathmorph.ggufio import GGUF, GGUFError, decode, sha256, assert_only_patches
from mathmorph.mathcore import leading_basis
from mathmorph.pipeline import encode_same_type
VERSION = '0.3.0'


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


@dataclass(frozen=True)
class Policy:
    profile: str = 'middle-pair'
    total_budget: float = .02
    row_budget: float = .01
    rank: int = 16
    strength: float = .35
    seed: int = 20260905
    workspace_gib: float = 2.
    def validate(self):
        if self.profile not in ('middle-pair', 'middle-single'):
            raise ValueError('Unknown local profile')
        for name, lo, hi in [('total_budget', 0., .05), ('row_budget', 0., .02),
                              ('strength', 0., .5), ('workspace_gib', .05, 64.)]:
            v = getattr(self, name)
            if not math.isfinite(v) or not lo < v <= hi:
                raise ValueError(f'Invalid {name}')
        if not isinstance(self.rank, int) or not 1 <= self.rank <= 128 or self.seed < 0:
            raise ValueError('Invalid rank or seed')


def select_tensors(g, policy):
    policy.validate()
    arch = g.metadata.get('general.architecture')
    if arch not in ('llama', 'qwen2', 'qwen3'):
        raise GGUFError('UNSUPPORTED architecture; a shared extension is not semantic compatibility')
    if g.metadata.get(arch + '.expert_count', 0):
        raise GGUFError('MoE is not covered by this adapter')
    count = g.metadata.get(arch + '.block_count')
    if not isinstance(count, int) or isinstance(count, bool) or count < 6:
        raise GGUFError('Requires a dense model with at least six blocks')
    lo, hi = count // 3, (2 * count) // 3 - 1
    layers = [count // 2] if policy.profile == 'middle-single' else sorted({lo, hi})
    names = []
    for i in layers:
        prefix = f'blk.{i}.ffn_'
        keys = [prefix + k + '.weight' for k in ('down', 'up', 'gate')]
        if any(k not in g.tensors for k in keys):
            raise GGUFError('Incomplete dense gated FFN')
        if any(prefix + k + '.bias' in g.tensors for k in ('down', 'up', 'gate')):
            raise GGUFError('Biased FFN adapter not validated')
        d, u, a = (g.tensors[k] for k in keys)
        if len(d.shape) != 2 or u.shape != a.shape or u.shape != tuple(reversed(d.shape)):
            raise GGUFError('Incompatible FFN shapes')
        if d.qtype not in (0, 1, 2, 8, 30):
            raise GGUFError('Unsupported write codec; no implicit high-precision replacement')
        if math.prod(d.shape) * 80 > policy.workspace_gib * 1024**3:
            raise MemoryError('Estimated single-tensor workspace exceeds limit')
        names.append(keys[0])
    return names


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')


def row_tangent(w, budget, rank=16, seed=0, strength=.35):
    """Rotate each nonzero row toward a projected tangent while preserving its norm.

    Per-row distance <= budget before encoding. This does NOT preserve arbitrary
    functions, matrix rank, attention, termination, or performance.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or not w.size or not np.isfinite(w).all():
        raise ValueError('Finite, nonempty 2-D weight required')
    if not math.isfinite(budget) or not 0 <= budget <= .1:
        raise ValueError('budget outside [0,.1]')
    if not isinstance(rank, int) or rank < 1 or not 0 <= strength <= .5 or not math.isfinite(strength):
        raise ValueError('Invalid rank or strength')
    if not np.any(w) or budget == 0 or strength == 0:
        return w.copy(), {'max_row_change': 0., 'relative_change': 0.}
    q = leading_basis(w, min(rank, *w.shape), seed, power_iters=2, oversampling=8)
    a = w.astype(np.float64)
    p = (q @ (q.T @ w)).astype(np.float64)
    ns = np.einsum('ij,ij->i', a, a)
    n = np.sqrt(ns)
    c = np.divide(np.einsum('ij,ij->i', p, a), ns, out=np.zeros_like(ns), where=ns > 0)
    v = p - c[:, None] * a
    vn = np.linalg.norm(v, axis=1)
    ratio = np.divide(vn, n, out=np.zeros_like(n), where=n > 0)
    angle = np.minimum(np.arctan(strength * ratio), 2 * np.arcsin(budget / 2))
    scale = np.divide(n * np.sin(angle), vn, out=np.zeros_like(n), where=vn > n * 1e-10)
    z = np.cos(angle)[:, None] * a + scale[:, None] * v
    z[n == 0] = 0
    out = z.astype(np.float32)
    stats = distortion(w, out)
    if stats['max_row_change'] > budget + 2e-6:
        raise ArithmeticError('Numerical row-bound failure')
    return out, stats


def distortion(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid tensor comparison')
    n, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    diff = np.linalg.norm(b-a, axis=1)
    if np.any((n == 0) & (diff > 0)):
        raise GGUFError('A zero row changed')
    rel = np.divide(diff, n, out=np.zeros_like(n), where=n > 0)
    drift = np.divide(abs(bn-n), n, out=np.zeros_like(n), where=n > 0)
    total = float(np.linalg.norm(a))
    return {'max_row_change': float(rel.max(initial=0)),
            'max_row_norm_drift': float(drift.max(initial=0)),
            'relative_change': float(np.linalg.norm(b-a)/total) if total else 0.}


def convert(source: Path, out: Path, policy=Policy(), expected_sha=None, quantizer=None):
    started = time.monotonic(); policy.validate()
    source, out = Path(source).resolve(strict=True), Path(out).resolve()
    if not source.is_file() or out.exists():
        raise ValueError('Existing source and NEW output directory required')
    before = sha256(source)
    if expected_sha and before != expected_sha:
        raise GGUFError('SOURCE_HASH_MISMATCH: never stack onto an unexpected checkpoint')
    g = GGUF(source); names = select_tensors(g, policy)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < g.size + 2**30:
        raise OSError('Insufficient disk reserve')
    out.mkdir()
    allocation = min(policy.row_budget, policy.total_budget / len(names))
    report = {'version': VERSION, 'method': 'local-row-tangent-NPSR',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input': str(source), 'source_sha256': before,
        'policy': asdict(policy), 'names': names, 'per_tensor_allocation': allocation,
        'training': False, 'evaluation_data_used_for_weights': False, 'parameters_added': 0,
        'protected': 'all unselected bytes, including ALL attention projections, embedding/output, norms and chat template',
        'protection_note': 'Byte preservation is not a functional termination guarantee',
        'stability_status': 'UNTESTED', 'capability_status': 'UNVERIFIED', 'release': 'RETAIN_ORIGINAL',
        'source_code_sha256': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).parent/'src/mathmorph/ggufio.py',
            Path(__file__).parent/'src/mathmorph/mathcore.py', Path(__file__).parent/'src/mathmorph/pipeline.py']}}
    save(out/'plan.json', report)
    partial, patches, tensors = out/'candidate.gguf.partial', {}, []
    try:
        with tempfile.TemporaryDirectory(prefix='encode-', dir=out) as tmp:
            scratch = Path(tmp)
            for i, name in enumerate(names):
                t = g.tensors[name]; w = g.matrix(name); original = g.raw(name)
                control = encode_same_type(g, name, w, quantizer, scratch, f'{i}.control', 4)
                if control != original:
                    raise GGUFError('ROUNDTRIP_DRIFT: refusing an unisolated conversion')
                # Reserve 25% for encoding error, then check ACTUAL decoded changes.
                z, ideal = row_tangent(w, allocation * .75, policy.rank,
                    stable_seed(policy.seed, name), policy.strength)
                raw = encode_same_type(g, name, z, quantizer, scratch, f'{i}.candidate', 4)
                actual = distortion(w, decode(raw, t.qtype, t.shape))
                if (len(raw) != t.nbytes or actual['max_row_change'] > allocation or
                    actual['max_row_norm_drift'] > .001 or raw == original):
                    raise GGUFError(f'ENCODED_GUARD_REJECTED: {name}: {actual}')
                patches[name] = raw
                tensors.append({'name': name, 'shape': t.shape, 'type': t.type_name, 'ideal': ideal,
                    'actual': actual, 'patch_sha256': hashlib.sha256(raw).hexdigest()})
                print('PATCH', name, json.dumps(actual), flush=True)
            measured = sum(x['actual']['relative_change'] for x in tensors)
            if measured > policy.total_budget:
                raise GGUFError('MODEL_WEIGHT_BUDGET_EXCEEDED')
            shutil.copyfile(source, partial)
            with partial.open('r+b') as f:
                for name, raw in patches.items():
                    f.seek(g.tensors[name].offset); f.write(raw)
                f.flush(); os.fsync(f.fileno())
            audit = assert_only_patches(source, partial, patches)
            if sha256(source) != before:
                raise GGUFError('Source changed while converting')
            output_sha = sha256(partial)
            final = out/'candidate.gguf'; partial.rename(final)
            report.update(tensors=tensors, measured_weight_budget=measured,
                candidate={'path': str(final), 'sha256': output_sha, 'bytes': final.stat().st_size},
                structural_status='PASS', source_unchanged=True, byte_audit=audit,
                elapsed_seconds=time.monotonic()-started)
            save(out/'report.json', report)
            print('CONVERSION_COMPLETE', output_sha, flush=True)
            return report
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        final = out/'candidate.gguf'
        if final.exists(): final.rename(out/'candidate.gguf.rejected')
        save(out/'FAILED.json', {'status': 'REJECTED', 'error': repr(exc), 'source_sha256': before})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--expected-source-sha256'); p.add_argument('--quantizer', type=Path)
    p.add_argument('--profile', choices=['middle-pair', 'middle-single'], default='middle-pair')
    p.add_argument('--total-budget', type=float, default=.02); p.add_argument('--row-budget', type=float, default=.01)
    a = p.parse_args()
    convert(a.source, a.out, Policy(profile=a.profile, total_budget=a.total_budget, row_budget=a.row_budget),
            a.expected_source_sha256, a.quantizer)
