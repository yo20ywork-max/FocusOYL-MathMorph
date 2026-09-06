"""Second predeclared development sweep; fresh holdout only for a promising candidate.
No old held-out labels enter the geometry or recipe selection.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import json,random,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq
import experiment as exp
from convert_geometry import GGUF,sha256,save,validate,make_direction,encode_native,decode,assert_only_patches
from screen import run_arm
from stability import summarize
from anchor_protection import contrast_basis,protect_delta

RECIPES=[
 {'label':'cubic_last128_full','layer':23,'rank':128,'alpha':1.,'geometry':True,'protect':False},
 {'label':'cubic_last128_protected','layer':23,'rank':128,'alpha':1.,'geometry':True,'protect':True},
 {'label':'euclid_last512_protected','layer':23,'rank':512,'budget':.12,'geometry':False,'protect':True},
 {'label':'euclid_last512_unprotected','layer':23,'rank':512,'budget':.12,'geometry':False,'protect':False},
 {'label':'euclid_last32_protected','layer':23,'rank':32,'alpha':1.,'geometry':False,'protect':True},
 {'label':'cubic_last32_protected','layer':23,'rank':32,'alpha':1.,'geometry':True,'protect':True}]

def fingerprint(c):return ' '.join(c['prompt'].casefold().split())

def prepare(project,out):
    seen=set();dev=[];exclude=set();study=project/'study'
    old_paths=[study/'data/development.json',study/'data/holdout.json',
      project.parent/'FocusOYL-Think3Way-20260905/suite.json',
      project.parent/'FocusOYL-MathMorph-20260905-C/screen-suite.json']
    all_previous=[]
    for path in old_paths:
        rows=json.loads(path.read_text(encoding='utf-8'))['cases'];all_previous+=rows
        exclude.update(c['id'] for c in rows)
    for c in json.loads(old_paths[0].read_text(encoding='utf-8'))['cases']:
        h=fingerprint(c)
        if h not in seen:dev.append(c);seen.add(h)
    seen={fingerprint(c) for c in all_previous};rng=random.Random(2026090573);test=[]
    for cat in ('ARC-Challenge','ARC-Easy'):
        rows=pq.read_table(study/'data'/f'{cat}-test.parquet').to_pylist();rng.shuffle(rows)
        picked=0
        for r in rows:
            cid=cat+':'+r['id'];labs=r['choices']['label'];texts=r['choices']['text']
            if cid in exclude or len(labs)>5:continue
            prompt='Answer the multiple-choice question. Write the answer as FINAL: followed by one option letter.\nQuestion: '+r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(texts))+'\nAnswer:'
            c={'id':cid,'category':cat,'prompt':prompt,'check':'mcq','expected':chr(65+labs.index(r['answerKey']))}
            if fingerprint(c) in seen:continue
            test.append(c);seen.add(fingerprint(c));picked+=1
            if picked==128:break
        assert picked==128
    for cat in ('arithmetic','code'):
        for i in range(32):
            while True:
                a,b=rng.randint(3,35),rng.randint(3,35);c,d=rng.randint(1,9),rng.randint(1,9);k=i%4
                if cat=='arithmetic':
                    text,answer=[(f'({a} * {b}) - {c}',a*b-c),(f'({a} + {b}) * {c}',(a+b)*c),
                      (f'{a} * {b} + {c} * {d}',a*b+c*d),(f'({a} - {b}) * ({c} + {d})',(a-b)*(c+d))][k]
                    prompt=f'Compute {text}. Write FINAL: followed by the integer.'
                else:
                    if k==0:text=f'a=[{a},{b}]; b=a; b.append({c}); print(sum(a))';answer=a+b+c
                    elif k==1:
                        n=rng.randint(3,16);text=f'print(sum(i*i for i in range({n})))';answer=sum(j*j for j in range(n))
                    elif k==2:text=f'a=[{a},{b},{c}]; b=a[:]; b.append({d}); print(len(a)+sum(b))';answer=3+a+b+c+d
                    else:text=f'print(sum(range({a},{a+2*b},2)))';answer=sum(range(a,a+2*b,2))
                    prompt=f'What integer does this Python code print? {text}\nWrite FINAL: followed by the integer.'
                row={'id':f'mm04-sweep-holdout-{cat}-{i}','category':cat,'prompt':prompt,'check':'integer','expected':str(answer)}
                if fingerprint(row) not in seen:break
            test.append(row);seen.add(fingerprint(row))
    assert len(dev)==47 and len(test)==320 and len({fingerprint(c) for c in test})==320
    save(out/'development.json',{'purpose':'development_after_deduplicating_first_study','cases':dev})
    save(out/'holdout.json',{'purpose':'fresh_prompt_disjoint_confirmation_for_single_selected_candidate','cases':test})
    save(out/'data_audit.json',{'n_dev':len(dev),'n_holdout':len(test),'prompt_overlap':0,
      'development_sha256':sha256(out/'development.json'),'holdout_sha256':sha256(out/'holdout.json'),
      'seed':2026090573,'excluded_id_count':len(exclude),'previous_source_sha256':{str(p):sha256(p) for p in old_paths},
      'shared_task_templates':True,'unseen_pretraining_content_guaranteed':False})
    return dev,test

def convert_one(source,out,recipe,cache):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);g=GGUF(source);arch,blocks=validate(g)
    name=f"blk.{recipe['layer']}.ffn_down.weight";tensor=g.tensors[name]
    if tensor.qtype not in (0,1,30):raise ValueError('Native float weights required')
    key=(recipe['layer'],recipe['rank'],recipe['geometry'])
    if key not in cache:cache[key]=make_direction(g,key[0],blocks,key[1],key[2])
    w,tail,stats=cache[key];protect_info=None;q=None
    if recipe['protect']:
        if recipe['layer']!=blocks-1:raise ValueError('Readout protection requires final FFN only')
        if 'contrast_basis' not in cache:cache['contrast_basis']=contrast_basis(g)
        q,protect_info=cache['contrast_basis'];tail=protect_delta(tail,q)
    alpha=min(1.,.98*recipe['budget']*float(np.linalg.norm(w))/max(float(np.linalg.norm(tail)),1e-30)) if 'budget' in recipe else recipe['alpha']
    if not 0<alpha<=1:raise ValueError('alpha must be in (0,1]')
    z=(w-alpha*tail).astype(np.float32);raw=encode_native(z,tensor.qtype);actual=decode(raw,tensor.qtype,tensor.shape)
    change=float(np.linalg.norm(actual-w)/np.linalg.norm(w))
    if not np.isfinite(actual).all() or change>1.5 or raw==g.raw(name):raise ValueError('Invalid or ineffective strong projection')
    leakage=float(np.linalg.norm(q.T@(actual-w))/max(np.linalg.norm(actual-w),1e-30)) if q is not None else None
    if leakage is not None and leakage>.005:raise ValueError('Encoded protected-subspace leakage exceeds 0.5 percent of update norm')
    if shutil.disk_usage(out).free<g.size+2**30:raise OSError('Disk reserve')
    partial=out/'candidate.gguf.partial';shutil.copyfile(source,partial)
    try:
        with partial.open('r+b') as f:f.seek(tensor.offset);f.write(raw);f.flush();os.fsync(f.fileno())
        audit=assert_only_patches(source,partial,{name:raw})
        if sha256(source)!=exp.SOURCE_SHA:raise ValueError('Original changed')
        final=out/'candidate.gguf';partial.rename(final)
        save(out/'report.json',{'recipe':recipe,'source_sha256':exp.SOURCE_SHA,'candidate_sha256':sha256(final),
          'candidate_path':str(final),'bytes':final.stat().st_size,'tensor':name,'geometry':stats,
          'alpha':alpha,'readout_protection':protect_info,'encoded_protected_leakage_ratio':leakage,'encoded_relative_change':change,'encoded_norm_ratio':float(np.linalg.norm(actual)/np.linalg.norm(w)),
          'rank_claim_applies_before_encoding_only':True,'byte_audit':audit,'training':False,
          'answers_in_weight_computation':False,'added_parameters':0,'release':'UNVERIFIED'})
        print('STRONG_CONVERT',recipe['label'],change,flush=True);return final
    finally:partial.unlink(missing_ok=True)

def evaluate(out,phase,models,cases,server):
    dest=out/phase;dest.mkdir();frozen={k:{'path':str(p),'sha256':sha256(p)} for k,p in models}
    save(dest/'protocol.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'models':frozen,'cfg':exp.CFG,
      'cases_sha256':sha256(out/('development.json' if phase=='development_run' else 'holdout.json'))})
    data={}
    for name,path in models:
        if (out/'CANCEL').exists():raise InterruptedError('User cancelled')
        data[name]=run_arm(name,path,frozen[name]['sha256'],server,dest,cases,exp.CFG)
        info=summarize(data[name]);save(dest/f'{name}.summary.json',info)
        print('SWEEP_ARM_DONE',phase,name,json.dumps(info),flush=True)
    return data

def main():
    project=Path(__file__).parent;out=project/'sweep'
    if out.exists():raise FileExistsError('Fresh study required')
    if not (project/'study/FINAL_DECISION.json').exists():raise RuntimeError('First experiment must be complete; no concurrent benchmark')
    proto=json.loads((project/'study/development/protocol.json').read_text(encoding='utf-8'))
    source=Path(proto['models']['original']['path']);server=Path(r'C:\Users\YOUR_USER\llama\runtime\llama-server.exe')
    if sha256(source)!=exp.SOURCE_SHA:raise ValueError('Original checkpoint hash mismatch')
    out.mkdir();exp.GATES={**exp.GATES,'holdout_multiple_comparisons':3}
    save(out/'study.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'recipes':RECIPES,'cfg':exp.CFG,
       'source_sha256':exp.SOURCE_SHA,'gates':exp.GATES,'n_development':47,'n_holdout':320,
       'followup_trigger':'at least 3 extra correct development items; no category/abort/cost regression gate breach',
       'selection':'most correct then least total reported tokens then label among eligible candidates',
       'geometry_uses_no_examples':True,'development_labels_select_recipe':True,
       'code_sha256':{p.name:sha256(p) for p in (Path(__file__),project/'geometry.py',project/'convert_geometry.py',project/'experiment.py',project/'screen.py',project/'stability.py',project/'anchor_protection.py')}})
    dev,test=prepare(project,out);cache={};models=[('original',source)]
    for recipe in RECIPES:models.append((recipe['label'],convert_one(source,out/'models'/recipe['label'],recipe,cache)))
    cache.clear();data=evaluate(out,'development_run',models,dev,server);base=summarize(data['original']);eligible=[];comps={}
    for label,rows in data.items():
        if label=='original':continue
        c=exp.compare(data['original'],rows,correction=3);comps[label]=c
        if c['candidate']>=c['baseline']+3 and c['valid_observation'] and c['no_extra_abort'] and c['token_ratio'] is not None and c['token_ratio']<=1.15 and min(c['category_delta_pp'].values())>=-2:eligible.append(label)
    if not eligible:
        save(out/'FINAL_DECISION.json',{'release':'RETAIN_ORIGINAL','reason':'NO_PROMISING_DEVELOPMENT_CANDIDATE',
           'development_comparisons':comps,'holdout_evaluated':False,'universal_gain_proven':False})
        print('SWEEP_FINAL NO_PROMISING_DEVELOPMENT_CANDIDATE',flush=True);return
    chosen=min(eligible,key=lambda k:(-summarize(data[k])['correct'],summarize(data[k])['reported_completion_tokens'],k))
    save(out/'selection.json',{'selected':chosen,'eligible':eligible,'development_comparisons':comps,
       'frozen_utc':datetime.now(timezone.utc).isoformat(),'holdout_sha256':sha256(out/'holdout.json')})
    data=evaluate(out,'holdout_run',[models[0],next(x for x in models if x[0]==chosen)],test,server)
    result=exp.compare(data['original'],data[chosen],correction=3)
    save(out/'FINAL_DECISION.json',{'selected':chosen,'release':result['release'],'result':result,
       'holdout_evaluated':True,'source_unchanged':sha256(source)==exp.SOURCE_SHA,
       'completed_utc':datetime.now(timezone.utc).isoformat(),'theoretical_model_limit_broken':False})
    print('SWEEP_FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
