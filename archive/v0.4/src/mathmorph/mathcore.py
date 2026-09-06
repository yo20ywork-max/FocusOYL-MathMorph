"""Gaussian-path-energy projector (GPEP), an unvalidated research operator.

No training samples, gradients, labels, teachers, or optimizer updates.
The quadrature is exact only in the quadrature limit under its Gaussian
input assumption. That assumption is not a statement about LLM activations.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    rank: int = 64
    strength: float = 0.05
    max_change: float = 0.02
    quadrature: int = 32
    power_iters: int = 2
    oversampling: int = 8
    seed: int = 20260905
    def validate(self):
        if self.rank < 1 or self.quadrature < 8 or self.quadrature > 128:
            raise ValueError("rank >= 1 and 8 <= quadrature <= 128 required")
        if not np.isfinite(self.strength) or not 0 <= self.strength <= 10:
            raise ValueError("strength must be finite and in [0,10]")
        if not np.isfinite(self.max_change) or not 0 <= self.max_change <= 0.25:
            raise ValueError("max_change must be finite and in [0,0.25]")
        if not 0 <= self.power_iters <= 8 or not 0 <= self.oversampling <= 64:
            raise ValueError("Invalid subspace parameters")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def _valid(a: np.ndarray):
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("Finite nonempty 2-D matrix required")


def path_moments(up: np.ndarray, gate: np.ndarray, norm: np.ndarray | None = None,
                 nodes: int = 32) -> np.ndarray:
    """Estimate E[(SiLU(g_j x) (u_j x))^2], x ~ N(0,I).

    With optional norm gamma, g and u are multiplied by gamma first.
    Let vg=||g||^2, vu=||u||^2, c=g.u. Conditional on a=g.x:
       E[b^2|a] = vu-c^2/vg + (c^2/vg^2)*a^2.
    Only 1-D Gauss-Hermite integration is needed for each channel.
    Channel cross-covariances are NOT included in the resulting metric.
    """
    _valid(up); _valid(gate)
    if up.shape != gate.shape:
        raise ValueError("Up/gate shape mismatch")
    u=np.asarray(up,dtype=np.float64)
    g=np.asarray(gate,dtype=np.float64)
    if norm is not None:
        n=np.asarray(norm,dtype=np.float64).reshape(-1)
        if n.size!=u.shape[1] or not np.isfinite(n).all():
            raise ValueError("Invalid FFN normalization scale")
        u=u*n; g=g*n
    vg=np.einsum('ij,ij->i',g,g)
    vu=np.einsum('ij,ij->i',u,u)
    c=np.einsum('ij,ij->i',u,g)
    tiny=np.finfo(np.float64).tiny
    safe=np.maximum(vg,tiny)
    conditional=np.maximum(vu-c*c/safe,0)
    beta2=(c/safe)**2
    z,w=np.polynomial.hermite.hermgauss(nodes)
    a=np.sqrt(2*vg[:,None])*z[None,:]
    # Stable sigmoid without exponent overflow.
    sig=np.exp(-np.logaddexp(0.0,-a))
    s2=(a*sig)**2
    m=np.sum(s2*(conditional[:,None]+beta2[:,None]*a*a)*w[None,:],axis=1)/np.sqrt(np.pi)
    m[vg==0]=0
    if not np.isfinite(m).all():
        raise ValueError("Non-finite path moment")
    return np.maximum(m,0)


def leading_basis(a: np.ndarray, rank: int, seed: int, power_iters=2, oversampling=8):
    """Deterministic-seed randomized range finder with reorthogonalization.

    This is linear-algebra iteration, not learning. For a fixed orthogonal
    basis the proximal solution below is exact, even when this range finder
    approximates the leading singular subspace.
    """
    _valid(a)
    m,n=a.shape
    r=min(rank,m,n)
    if r==m:
        return np.eye(m,dtype=np.float32)
    k=min(r+oversampling,m,n)
    rng=np.random.default_rng(seed)
    omega=rng.standard_normal((n,k)).astype(np.float32)
    q=np.linalg.qr(a@omega,mode='reduced')[0]
    for _ in range(power_iters):
        z=np.linalg.qr(a.T@q,mode='reduced')[0]
        q=np.linalg.qr(a@z,mode='reduced')[0]
    small=q.T@a
    # Eigenproblem is only k by k.
    gram=np.asarray(small,dtype=np.float64)@np.asarray(small.T,dtype=np.float64)
    eig,v=np.linalg.eigh(gram)
    order=np.argsort(eig)[::-1][:r]
    basis=q@v[:,order].astype(np.float32)
    return np.linalg.qr(basis,mode='reduced')[0]


def transform(down: np.ndarray, up: np.ndarray, gate: np.ndarray,
              norm: np.ndarray | None, config: Config, mode="gpep") -> tuple[np.ndarray,dict]:
    config.validate(); _valid(down); _valid(up); _valid(gate)
    if down.shape[1]!=up.shape[0] or up.shape!=gate.shape or down.shape[0]!=up.shape[1]:
        raise ValueError("Requires a compatible dense gated FFN")
    if mode not in ("gpep","unweighted","roundtrip"):
        raise ValueError("Unknown operator")
    w=np.asarray(down,dtype=np.float32)
    stats={"operator":mode,"configuration":asdict(config),"research_status":"UNVALIDATED_CAPABILITY_HYPOTHESIS"}
    if mode=="roundtrip" or config.strength==0 or config.max_change==0:
        return w.copy(), {**stats,"eta":0.0,"ideal_relative_change":0.0}
    moment=path_moments(up,gate,norm,config.quadrature) if mode=="gpep" else np.ones(w.shape[1])
    if not np.any(moment>0) or not np.any(w):
        return w.copy(),{**stats,"eta":0.0,"ideal_relative_change":0.0,"reason":"zero_energy"}
    scale=np.sqrt(moment)
    med=float(np.median(scale[scale>0]))
    # Numerical regularization of the proxy, not inferred semantics.
    scale=np.clip(scale/max(med,1e-30),1e-3,1e3).astype(np.float32)
    a=w*scale[None,:]
    q=leading_basis(a,config.rank,config.seed,config.power_iters,config.oversampling)
    preserved=q@(q.T@w)
    residual=w-preserved
    wn=float(np.linalg.norm(w.astype(np.float64)))
    rn=float(np.linalg.norm(residual.astype(np.float64)))
    eta=min(config.strength/(1+config.strength),config.max_change*wn/max(rn,1e-30))
    candidate=w-np.float32(eta)*residual
    delta=candidate-w
    measured=float(np.linalg.norm(delta.astype(np.float64))/max(wn,1e-30))
    if measured>config.max_change*(1+1e-4)+1e-7:
        raise ArithmeticError("Perturbation bound violated")
    if not np.isfinite(candidate).all():
        raise ArithmeticError("Non-finite candidate")
    stats.update({"eta":eta,"ideal_relative_change":measured,"effective_rank":int(q.shape[1]),
        "proxy_moment_min":float(moment.min()),"proxy_moment_max":float(moment.max()),
        "assumption":"isotropic_Gaussian_before_learned_norm; diagonal_channel_moment_proxy",
        "orthogonality_error":float(np.linalg.norm(q.T@q-np.eye(q.shape[1]))),
        "projected_component_relative_change":float(np.linalg.norm(q.T@delta)/max(wn,1e-30)),
        "unselected_component_multiplier":1-eta})
    return candidate,stats
