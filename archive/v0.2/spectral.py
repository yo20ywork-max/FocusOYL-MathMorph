"""Norm-preserving spectral redistribution. A research hypothesis, not a capability theorem."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
from mathmorph.mathcore import leading_basis


def components(array: np.ndarray, rank=16, seed=20260905):
    a=np.asarray(array,dtype=np.float32)
    if a.ndim<2 or not np.isfinite(a).all() or min(a.shape)<1:
        raise ValueError('Finite nonempty tensor with ndim >= 2 required')
    # Explicit matricization, not a claim about convolution/MoE semantics.
    w=a.reshape(-1,a.shape[-1])
    norm=float(np.linalg.norm(w.astype(np.float64)))
    if norm==0: return w,np.zeros_like(w),0.0,norm,a.shape
    q=leading_basis(w,min(rank,*w.shape),seed,power_iters=2,oversampling=8)
    p=q@(q.T@w)
    e=float(np.sum(p.astype(np.float64)**2)/(norm*norm))
    e=float(np.clip(e,0,1))
    # Remove numerical longitudinal component; this leaves the direction tangent.
    v=p-np.float32(e)*w
    longitudinal=float(np.sum(v.astype(np.float64)*w)/(norm*norm))
    v=v-np.float32(longitudinal)*w
    return w,v,e,norm,a.shape


def transform_from_components(comp, direction=1, strength=0.35, budget=0.05):
    if direction not in (-1,1) or not 0<=strength<=0.5 or not 0<=budget<=0.2:
        raise ValueError('direction +/-1; strength [0,.5]; budget [0,.2] required')
    w,v,e,norm,shape=comp
    vn=float(np.linalg.norm(v.astype(np.float64)))
    if norm==0 or vn<norm*1e-8 or strength==0 or budget==0:
        return w.copy().reshape(shape),{'e':e,'lambda':0.,'relative_change':0.,'norm_ratio':1.,'no_effect':True}
    import math
    max_lambda=math.tan(2*math.asin(budget/2))*norm/vn
    lam=direction*min(strength,max_lambda)
    z=w-np.float32(lam)*v
    # Measured normalization accounts for finite precision and approximate P.
    z*=np.float32(norm/np.linalg.norm(z.astype(np.float64)))
    change=float(np.linalg.norm(z.astype(np.float64)-w)/norm)
    if change>budget+2e-6 or not np.isfinite(z).all():
        raise ArithmeticError('Numerical bound failure')
    return z.reshape(shape),{'e':e,'lambda':lam,'relative_change':change,
        'norm_ratio':float(np.linalg.norm(z.astype(np.float64))/norm),'no_effect':False,
        'theoretical_relative_change':math.sqrt(max(0,2-2/math.sqrt(1+lam*lam*e*(1-e))))}


def transform(array,rank=16,seed=20260905,direction=1,strength=.35,budget=.05):
    return transform_from_components(components(array,rank,seed),direction,strength,budget)
