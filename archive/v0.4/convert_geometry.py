"""Immutable GGUF conversion using cubic channel geometry and downstream readers."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
import argparse,hashlib,json,math,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
from mathmorph.ggufio import GGUF,GGUFError,sha256,encode_native,decode,assert_only_patches
from geometry import cubic_kernel,reader_metric,metric_tail_direction,bounded_shrink

def save(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def recipes(blocks):
    if blocks<8:raise ValueError('At least 8 dense blocks required')
    return [
      {'label':'lifted_last_mild','layers':[blocks-1],'rank':512,'budget':.04,'geometry':True},
      {'label':'lifted_last_strong','layers':[blocks-1],'rank':512,'budget':.12,'geometry':True},
      {'label':'lifted_cross','layers':[blocks-4,blocks-2],'rank':512,'budget':.04,'geometry':True},
      {'label':'euclid_control','layers':[blocks-1],'rank':512,'budget':.12,'geometry':False}]

def validate(g):
    arch=g.metadata.get('general.architecture')
    if arch not in ('llama','qwen2','qwen3') or g.metadata.get(arch+'.expert_count',0):
        raise GGUFError('Unsupported architecture; no universal compatibility claim')
    return arch,int(g.metadata[arch+'.block_count'])

def readers_for(g,layer,blocks,d):
    if layer+1<blocks:
        norm=g.matrix(f'blk.{layer+1}.ffn_norm.weight').reshape(-1)
        return [g.matrix(f'blk.{layer+1}.ffn_{k}.weight')*norm for k in ('gate','up')], 'next_ffn_gate_up'
    name='output.weight' if 'output.weight' in g.tensors else 'token_embd.weight'
    t=g.tensors[name]
    if t.qtype not in (0,1):raise GGUFError('Terminal reader requires native FP16/FP32 head')
    if len(t.shape)!=2 or t.shape[1]!=d:raise GGUFError('Output head shape mismatch')
    a=np.memmap(g.path,dtype='<f4' if t.qtype==0 else '<f2',mode='r',offset=t.offset,shape=t.shape)
    ix=np.linspace(0,t.shape[0]-1,min(4096,t.shape[0]),dtype=np.int64)
    r=np.array(a[ix],dtype=np.float32);del a
    r*=g.matrix('output_norm.weight').reshape(-1)
    return [r],'4096_uniform_vocabulary_rows'

def make_direction(g,l,blocks,rank,geometry):
    key=f'blk.{l}.ffn_down.weight';w=g.matrix(key);d,f=w.shape
    if max(d,f)>8192 or d*d*32+f*f*36+w.size*24>3*1024**3:
        raise MemoryError('Analytic geometry workspace exceeds 3 GiB')
    if any(f'blk.{l}.ffn_{x}.bias' in g.tensors for x in ('gate','up','down')):
        raise GGUFError('Biased FFN not supported')
    if geometry:
        norm=g.matrix(f'blk.{l}.ffn_norm.weight').reshape(-1)
        gate=g.matrix(f'blk.{l}.ffn_gate.weight')*norm
        up=g.matrix(f'blk.{l}.ffn_up.weight')*norm
        if gate.shape!=(f,d) or up.shape!=(f,d):raise GGUFError('Gated FFN shape mismatch')
        k,ks=cubic_kernel(gate,up); del gate,up
        readers,rtype=readers_for(g,l,blocks,d)
        root,inv,hs=reader_metric(readers,d);del readers
    else:
        k=None;root=inv=np.eye(d,dtype=np.float32);ks={};hs={};rtype='identity'
    tail,ts=metric_tail_direction(w,k,root,inv,min(rank,d,f))
    return w,tail,{'kernel':ks,'reader_type':rtype,'metric':hs,'projection':ts}

def convert(source,out,recipe,cache=None,expected_sha=None):
    source,out=Path(source).resolve(strict=True),Path(out).resolve()
    if out.exists():raise FileExistsError('New output directory required')
    digest=sha256(source)
    if expected_sha and digest!=expected_sha:raise GGUFError('Unexpected source checkpoint')
    g=GGUF(source);arch,blocks=validate(g)
    if any(not 0<=l<blocks for l in recipe['layers']):raise GGUFError('Layer out of range')
    if not 1<=recipe['rank']<=8192 or not 0<recipe['budget']<=.3:raise ValueError('Invalid recipe')
    out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free<g.size+2**30:raise OSError('Insufficient disk reserve')
    out.mkdir();start=time.monotonic();patches={};detail=[]
    plan={'method':'cubic-feature-reader-metric-tail','version':'0.4.0','recipe':recipe,'source':str(source),
       'source_sha256':digest,'architecture':arch,'started_utc':datetime.now(timezone.utc).isoformat(),
       'training':False,'calibration_examples':False,'answers_in_conversion':False,'added_parameters':0,
       'capability_status':'UNVERIFIED','release':'RETAIN_ORIGINAL',
       'code_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('geometry.py')]}}
    save(out/'plan.json',plan);partial=out/'candidate.gguf.partial'
    try:
        for l in recipe['layers']:
            name=f'blk.{l}.ffn_down.weight';t=g.tensors[name]
            if t.qtype not in (0,1,30):raise GGUFError('This release writes native floats only')
            ck=(digest,l,recipe['rank'],recipe['geometry'])
            if cache is not None and ck in cache:w,tail,stats=cache[ck]
            else:
                w,tail,stats=make_direction(g,l,blocks,recipe['rank'],recipe['geometry'])
                if cache is not None:cache[ck]=(w,tail,stats)
            z,ds=bounded_shrink(w,tail,recipe['budget']*.98)
            raw=encode_native(z,t.qtype);actual=decode(raw,t.qtype,t.shape)
            change=float(np.linalg.norm(actual-w)/max(np.linalg.norm(w),1e-30))
            if not np.isfinite(actual).all() or change>recipe['budget'] or len(raw)!=t.nbytes:
                raise GGUFError('Encoded change guard failed')
            if raw==g.raw(name):raise GGUFError('No effective weight edit')
            patches[name]=raw;detail.append({'name':name,'geometry':stats,'shrink':ds,'encoded_relative_change':change})
            print('GEOMETRY',recipe['label'],name,round(change,6),flush=True)
        shutil.copyfile(source,partial)
        with partial.open('r+b') as f:
            for name,raw in patches.items():f.seek(g.tensors[name].offset);f.write(raw)
            f.flush();os.fsync(f.fileno())
        audit=assert_only_patches(source,partial,patches)
        if sha256(source)!=digest:raise GGUFError('Source changed during conversion')
        final=out/'candidate.gguf';partial.rename(final)
        plan.update(tensors=detail,byte_audit=audit,candidate={'path':str(final),'sha256':sha256(final),'bytes':final.stat().st_size},
           source_unchanged=True,seconds=time.monotonic()-start,structural_status='PASS')
        save(out/'report.json',plan);return final
    except BaseException as e:
        partial.unlink(missing_ok=True);save(out/'REJECTED.json',{'error':repr(e),'source_unchanged':sha256(source)==digest});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recipe',choices=[x['label'] for x in recipes(24)],default='lifted_last_mild')
    p.add_argument('--expected-sha256');a=p.parse_args();g=GGUF(a.source);_,b=validate(g)
    r=next(x for x in recipes(b) if x['label']==a.recipe);print(convert(a.source,a.out,r,expected_sha=a.expected_sha256))
