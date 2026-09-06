"""Create both predeclared candidates. Never read evaluation prompts/answers."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import sys, json, shutil, argparse, re, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
from mathmorph.ggufio import GGUF, GGUFError, sha256, decode, assert_only_patches
from mathmorph.pipeline import encode_same_type
from spectral import components,transform_from_components


def run(source:Path,out:Path,quantizer:Path|None=None,select='residual',rank=16):
    source=source.resolve(strict=True);out=out.resolve()
    if out.exists(): raise FileExistsError('New output directory required')
    g=GGUF(source)
    pattern=r'blk\.\d+\.(ffn_down|attn_output)\.weight'
    names=[k for k,t in g.tensors.items() if len(t.shape)>=2 and
        ((select=='all-matrix' and k.endswith('.weight')) or re.fullmatch(pattern,k))]
    if not names: raise GGUFError('No eligible tensor; this is UNSUPPORTED, not an improvement')
    if any(np.prod(g.tensors[k].shape)>100_000_000 for k in names):
        raise MemoryError('Tensor exceeds this implementation workspace limit')
    out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free<g.size*3+2**30: raise OSError('Insufficient free disk')
    out.mkdir(); scratch=out/'scratch';scratch.mkdir()
    start=time.monotonic();before=sha256(source)
    plan={'version':'0.2.0','method':'NPSR','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'input':str(source),'source_sha256':before,'size_bytes':g.size,
        'architecture':g.metadata.get('general.architecture'),'select':select,
        'rank':rank,'strength':.35,'budget':.05,'seed':20260905,'names':names,
        'directions':{'flatten':1,'sharpen':-1},'training':False,'evaluation_data_used':False,
        'novelty_status':'new_project_candidate_not_established_historical_novelty',
        'capability_status':'UNTESTED','universal_compatibility':False,
        'source_code_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).parent/'spectral.py']}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    patchsets={k:{} for k in ('flatten','sharpen')};controls={};rows=[]
    try:
        for idx,name in enumerate(names):
            t=g.tensors[name];w=g.matrix(name)
            comp=components(w,rank,20260905+idx)
            control=encode_same_type(g,name,w,quantizer,scratch,f'{idx}.control',4)
            if control!=g.raw(name): controls[name]=control
            for label,sgn in plan['directions'].items():
                z,stats=transform_from_components(comp,sgn,.35,.05)
                raw=encode_same_type(g,name,z,quantizer,scratch,f'{idx}.{label}',4)
                actual=decode(raw,t.qtype,t.shape)
                n=max(float(np.linalg.norm(w.astype(np.float64))),1e-30)
                relative=float(np.linalg.norm(actual.astype(np.float64)-w)/n)
                norm_ratio=float(np.linalg.norm(actual.astype(np.float64))/n) if np.any(w) else 1.
                if not np.isfinite(actual).all() or relative>.062 or abs(norm_ratio-1)>.01:
                    raise GGUFError(f'Encoded bound rejected {label}/{name}: {relative}, {norm_ratio}')
                patchsets[label][name]=raw
                rows.append({'name':name,'candidate':label,'type':t.type_name,**stats,
                    'encoded_relative_change':relative,'encoded_norm_ratio':norm_ratio,
                    'changed_bytes':sum(a!=b for a,b in zip(raw,g.raw(name))) if len(raw)<1_000_000 else None,
                    'raw_sha256':hashlib.sha256(raw).hexdigest()})
            print(f'TENSOR {idx+1}/{len(names)} {name} e={comp[2]:.5f}',flush=True)
        report={**plan,'tensors':rows,'outputs':{},'roundtrip_modified_tensors':list(controls)}
        if controls: patchsets['roundtrip-control']=controls
        for label,patches in patchsets.items():
            partial=out/(label+'.gguf.partial');shutil.copyfile(source,partial)
            with partial.open('r+b') as f:
                for name,data in patches.items():f.seek(g.tensors[name].offset);f.write(data)
                f.flush();os.fsync(f.fileno())
            audit=assert_only_patches(source,partial,patches)
            final=out/(label+'.gguf');partial.rename(final)
            report['outputs'][label]={'path':str(final),'sha256':sha256(final),**audit}
        if sha256(source)!=before: raise GGUFError('Source changed')
        report.update(source_unchanged=True,structural_status='PASS',elapsed_seconds=time.monotonic()-start)
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        shutil.rmtree(scratch)
        print(json.dumps({'status':'PASS','outputs':report['outputs'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        return report
    except BaseException as exc:
        (out/'FAILED.json').write_text(json.dumps({'status':'REJECTED','error':str(exc)}),encoding='utf-8')
        for p in out.glob('*.gguf'):p.rename(p.with_suffix('.gguf.rejected'))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--quantizer',type=Path);p.add_argument('--select',choices=['residual','all-matrix'],default='residual')
    p.add_argument('--rank',type=int,default=16);a=p.parse_args();run(a.input,a.out,a.quantizer,a.select,a.rank)
