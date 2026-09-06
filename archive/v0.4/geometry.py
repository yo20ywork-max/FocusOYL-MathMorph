"""Data-free cubic-feature geometry. Proxy moments are NOT model activations."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(k, '4')
import numpy as np
from numpy.polynomial.hermite import hermgauss

def _matrix(a):
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Nonempty finite 2D matrix required')
    return a

def hermite_coefficients(scales, nodes=48):
    s=np.asarray(scales,dtype=np.float64)
    if s.ndim!=1 or not np.isfinite(s).all() or np.any(s<0):
        raise ValueError('Finite nonnegative gate scales required')
    x,w=hermgauss(nodes); z=np.sqrt(2)*x; w=w/np.sqrt(np.pi)
    a=s[:,None]*z[None,:]
    sig=np.exp(-np.logaddexp(0.,-a)); phi=a*sig
    basis=np.stack((np.ones_like(z),z,(z*z-1)/np.sqrt(2)))
    return (phi*w)@basis.T

def cubic_kernel(g,u, *, dtype=np.float32):
    g,u=_matrix(g),_matrix(u)
    if g.shape!=u.shape: raise ValueError('Gate/up shape mismatch')
    s=np.linalg.norm(g,axis=1)
    v=np.divide(g,s[:,None],out=np.zeros_like(g),where=s[:,None]>0)
    coeff=hermite_coefficients(s); a,b,c=coeff.T
    t=np.einsum('ij,ij->i',v,u)
    linear=a[:,None]*u+np.sqrt(2)*c[:,None]*t[:,None]*v
    mean=b*t
    v,u,linear=(np.asarray(x,dtype=dtype) for x in (v,u,linear))
    vv=v@v.T; uu=u@u.T; vu=v@u.T
    cross=vu*vu.T
    k=linear@linear.T+np.outer(mean,mean).astype(dtype)
    k+=np.outer(b,b).astype(dtype)*(vv*uu+cross)
    k+=np.outer(c,c).astype(dtype)*(vv*vv*uu+2*vv*cross)
    k=(k+k.T)*.5
    scale=float(np.mean(np.diag(k)))
    if scale>0: k/=scale
    return k, {'quadrature_nodes':48,'coefficient_orders':[0,1,2],
       'feature_chaoses':[0,1,2,3],'kernel_scale_removed':scale,
       'assumption':'isotropic Gaussian after absorbing learned norm; cubic SiLU surrogate',
       'true_activation_covariance':False}

def reader_metric(readers, dimension, ridge=.25):
    if dimension<1 or not np.isfinite(ridge) or ridge<=0:
        raise ValueError('Invalid metric dimension/ridge')
    h=np.zeros((dimension,dimension),dtype=np.float64)
    for r in readers:
        r=_matrix(r)
        if r.shape[1]!=dimension: raise ValueError('Reader dimension mismatch')
        norm=np.linalg.norm(r,axis=1)
        r=np.divide(r,norm[:,None],out=np.zeros_like(r),where=norm[:,None]>0)
        h+=r.T@r/max(1,r.shape[0])
    tr=np.trace(h)
    if tr>0: h*=dimension/tr
    h+=ridge*np.eye(dimension)
    vals,vecs=np.linalg.eigh((h+h.T)*.5)
    root=(vecs*np.sqrt(vals))@vecs.T
    inv=(vecs/np.sqrt(vals))@vecs.T
    return root.astype(np.float32),inv.astype(np.float32),{
        'ridge':ridge,'min_eigenvalue':float(vals.min()),'max_eigenvalue':float(vals.max())}

def metric_tail_direction(w,k,root,inv,rank):
    w=np.asarray(_matrix(w),dtype=np.float32)
    m,n=w.shape
    if not isinstance(rank,int) or not 1<=rank<=min(m,n): raise ValueError('Invalid rank')
    if root.shape!=(m,m) or inv.shape!=(m,m): raise ValueError('Metric shape mismatch')
    a=root@w
    if k is None: s=a@a.T
    else:
        if k.shape!=(n,n) or not np.isfinite(k).all(): raise ValueError('Kernel shape/value mismatch')
        s=(a@k)@a.T
    s=np.asarray((s+s.T)*.5,dtype=np.float64)
    values,q=np.linalg.eigh(s)
    if values[-1]<0 or values[0]<-1e-4*max(1.,values[-1]):
        raise ArithmeticError('Proxy covariance not numerically PSD')
    q=q[:,-rank:].astype(np.float32)
    retained=inv@(q@(q.T@a))
    tail=w-retained
    return tail, {'rank':rank,'proxy_energy_retained':float(np.maximum(values[-rank:],0).sum()/max(np.maximum(values,0).sum(),1e-30)),
        'covariance_eigen_min':float(values[0]),'covariance_eigen_max':float(values[-1]),
        'tail_relative_norm':float(np.linalg.norm(tail)/max(np.linalg.norm(w),1e-30))}

def bounded_shrink(w,tail,budget):
    w,tail=_matrix(w),_matrix(tail)
    if w.shape!=tail.shape or not np.isfinite(budget) or not 0<=budget<=.3:
        raise ValueError('Invalid budget or shape')
    wn=float(np.linalg.norm(w));tn=float(np.linalg.norm(tail))
    alpha=min(1.,budget*wn/tn) if tn and wn else 0.
    z=(w-alpha*tail).astype(np.float32)
    return z,{'alpha':alpha,'relative_change':float(np.linalg.norm(z-w)/max(wn,1e-30)),
               'weight_bound_only':True}
