"""Predeclared experiment: weights use no data; development selects; holdout judges."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,math,random,sys,time,urllib.request,hashlib
from pathlib import Path
from datetime import datetime,timezone
from convert_geometry import convert,recipes,save,GGUF,validate,sha256
from screen import run_arm
from stability import summarize
SOURCE_SHA='68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d'
DATA_REV='210d026faf9955653af8916fad021475a3f00453'
CFG={'max_tokens':2048,'seconds':120,'gpu_layers':99,'workers':4,'context':4096,'think':True}
GATES={'gain_pp_min':5.,'two_sided_p_max':.05,'cost_ratio_max':1.15,
       'observed_category_regression_max_pp':2.,'holdout_multiple_comparisons':2}

def download(url,path):
    with urllib.request.urlopen(url,timeout=60) as r:raw=r.read()
    path.write_bytes(raw)

def prepare(root,previous):
    import pyarrow.parquet as pq
    d=root/'data';d.mkdir();exclude=set()
    for p in previous:
        v=json.loads(p.read_text(encoding='utf-8'))
        exclude.update(x['id'] for x in v['cases'])
    rng=random.Random(2026090551);dev=[];test=[];source=[]
    for cat in ['ARC-Challenge','ARC-Easy']:
        for split,n,dest in [('validation',16,dev),('test',64,test)]:
            p=d/f'{cat}-{split}.parquet'
            url=f'https://huggingface.co/datasets/allenai/ai2_arc/resolve/{DATA_REV}/{cat}/{split}-00000-of-00001.parquet'
            download(url,p);rows=pq.read_table(p).to_pylist()
            rows=[r for r in rows if cat+':'+r['id'] not in exclude]
            rng.shuffle(rows)
            source.append({'url':url,'sha256':sha256(p),'selected':n})
            for r in rows[:n]:
                labels=r['choices']['label'];choices=r['choices']['text']
                expected=chr(65+labels.index(r['answerKey']))
                prompt='Answer the multiple-choice question. Write the answer as FINAL: followed by one option letter.\nQuestion: '+r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(choices))+'\nAnswer:'
                dest.append({'id':cat+':'+r['id'],'category':cat,'prompt':prompt,'check':'mcq','expected':expected})
    used=set()
    for split,n,dest in [('dev',8,dev),('holdout',16,test)]:
        for i in range(n):
            while True:
                a,b,c,d=[rng.randint(3,25) for _ in range(4)];tup=(a,b,c,d)
                if tup not in used:used.add(tup);break
            kind=i%4
            expressions=[(f'({a} * {b}) - {c}',a*b-c),(f'({a} + {b}) * {c}',(a+b)*c),
                (f'{a} * {b} + {c} * {d}',a*b+c*d),(f'({a} - {b}) * ({c} + {d})',(a-b)*(c+d))]
            text,ans=expressions[kind]
            dest.append({'id':f'mm04-{split}-arith-{i}','category':'arithmetic','check':'integer','expected':str(ans),
                'prompt':f'Compute {text}. Write FINAL: followed by the integer.'})
            if kind==0:code=f'a=[{a},{b}]; b=a; b.append({c}); print(sum(a))';ans=a+b+c
            elif kind==1:code=f'print(sum(i*i for i in range({i%4+3})))';ans=sum(j*j for j in range(i%4+3))
            elif kind==2:code=f'a=[{a},{b},{c}]; b=a[:]; b.append({d}); print(len(a)+sum(b))';ans=3+a+b+c+d
            else:code=f'print(sum(range({a},{a+2*b},2)))';ans=sum(range(a,a+2*b,2))
            dest.append({'id':f'mm04-{split}-code-{i}','category':'code','check':'integer','expected':str(ans),
                'prompt':f'What integer does this Python code print? {code}\nWrite FINAL: followed by the integer.'})
    assert len(dev)==48 and len(test)==160
    assert not ({x['id'] for x in dev}&{x['id'] for x in test})
    d=root/'data'
    save(d/'development.json',{'purpose':'development_recipe_selection','cases':dev})
    save(d/'holdout.json',{'purpose':'single_frozen_heldout_comparison','cases':test})
    save(d/'manifest.json',{'data_revision':DATA_REV,'sources':source,'excluded_previous_ids':len(exclude),
        'seed':2026090551,'development_sha256':sha256(d/'development.json'),'holdout_sha256':sha256(d/'holdout.json'),
        'no_pretraining_contamination_claim':True,'synthetic_templates_shared_across_splits':True})

def compare(a,b,correction=2):
    if [x['id'] for x in a]!=[x['id'] for x in b]:raise ValueError('Unpaired observations')
    n=len(a);w=sum(not x['correct'] and y['correct'] for x,y in zip(a,b));l=sum(x['correct'] and not y['correct'] for x,y in zip(a,b));d=w+l
    p=min(1.,2*sum(math.comb(d,k) for k in range(min(w,l)+1))/2**d) if d else 1.
    sb,sc=summarize(a),summarize(b)
    ratio=sc['reported_completion_tokens']/sb['reported_completion_tokens'] if sb['missing_usage_rows']==sc['missing_usage_rows']==0 else None
    delta=100*(sc['correct']-sb['correct'])/n
    cat={c:100*(sc['categories'][c]['correct']-v['correct'])/v['n'] for c,v in sb['categories'].items()}
    valid=not any(x['status'] in ('TRANSPORT_ERROR','USER_CANCELLED','WALL_TIME_CENSORED') for x in a+b)
    no_extra_abort=all(sc['status'].get(s,0)<=sb['status'].get(s,0) for s in ('TOKEN_LIMIT','LOOP_GUARD_ABORT'))
    passed=(valid and delta>=GATES['gain_pp_min'] and p*correction<GATES['two_sided_p_max']
        and ratio is not None and ratio<=GATES['cost_ratio_max']
        and min(cat.values())>=-GATES['observed_category_regression_max_pp'] and no_extra_abort)
    return {'n':n,'baseline':sb['correct'],'candidate':sc['correct'],'delta_pp':delta,'wins':w,'losses':l,
       'two_sided_exact_p':p,'p_adjusted':min(1.,p*correction),'token_ratio':ratio,'category_delta_pp':cat,
       'valid_observation':valid,'no_extra_abort':no_extra_abort,'gates':GATES,
       'release':'VERIFIED_SCOPED_BENCHMARK_GAIN' if passed else 'RETAIN_ORIGINAL',
       'universal_gain_proven':False,'theoretical_model_limit_broken':False}

def evaluate_set(root,split,models):
    out=root/split;out.mkdir()
    data=json.loads((root/'data'/f'{split}.json').read_text(encoding='utf-8'))['cases']
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'cfg':CFG,'gates':GATES,'n':len(data),
       'models':{k:{'path':str(p),'sha256':sha256(p)} for k,p in models},
       'suite_sha256':sha256(root/'data'/f'{split}.json'),'server_sha256':sha256(SERVER),
       'raw_final_only_scoring':True,'no_prompt_changes_per_arm':True}
    save(out/'protocol.json',plan);observed={}
    for label,path in models:
        if (root/'CANCEL').exists():raise InterruptedError('User cancelled experiment')
        observed[label]=run_arm(label,path,plan['models'][label]['sha256'],SERVER,out,data,CFG)
        brief=summarize(observed[label]);save(out/f'{label}.summary.json',brief)
        print('ARM_DONE',split,label,json.dumps(brief),flush=True)
    comps={k:compare(observed['original'],v) for k,v in observed.items() if k!='original'}
    save(out/'comparisons.json',comps);return observed,comps

def main():
    global SERVER
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--server',type=Path,required=True)
    p.add_argument('--previous-suite',type=Path,action='append',default=[]);a=p.parse_args()
    root=a.root.resolve();root.mkdir(exist_ok=True);SERVER=a.server.resolve(strict=True)
    if (root/'study.json').exists():raise FileExistsError('Study already started')
    if sha256(a.source)!=SOURCE_SHA:raise ValueError('Baseline hash mismatch')
    g=GGUF(a.source);_,blocks=validate(g);rs=recipes(blocks)
    save(root/'study.json',{'version':'0.4.0','source_sha256':SOURCE_SHA,'recipes':rs,'gates':GATES,'cfg':CFG,
       'weight_computation_uses_no_data':True,'development_labels_used_for_recipe_selection':True,
       'holdout_rule':'choose one recipe once by dev correct, then valid cost, then lexical label; do not tune after holdout',
       'n_development':48,'n_holdout':160,'frozen_utc':datetime.now(timezone.utc).isoformat(),
       'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ('geometry.py','convert_geometry.py','experiment.py','screen.py','stability.py')}})
    prepare(root,a.previous_suite)
    cache={};models=[('original',a.source.resolve())]
    for recipe in rs:
        q=convert(a.source,root/'models'/recipe['label'],recipe,cache,expected_sha=SOURCE_SHA);models.append((recipe['label'],q))
    cache.clear()
    obs,comps=evaluate_set(root,'development',models)
    def key(k):
        s=summarize(obs[k]);return (-s['correct'],s['missing_usage_rows'],s['reported_completion_tokens'],k)
    selected=min([k for k in obs if k!='original'],key=key)
    selection={'selected':selected,'rule':'dev correctness descending, missing cost then reported tokens ascending, label',
       'all_dev_results':comps,'holdout_sha256':sha256(root/'data/holdout.json'),'frozen_utc':datetime.now(timezone.utc).isoformat()}
    save(root/'selection.json',selection);print('SELECTED',selected,flush=True)
    finalists=[models[0],next(x for x in models if x[0]==selected)]
    if selected!='euclid_control':finalists.append(next(x for x in models if x[0]=='euclid_control'))
    obs,results=evaluate_set(root,'holdout',finalists)
    verdict=results[selected]
    save(root/'FINAL_DECISION.json',{'selected':selected,'result':verdict,'all_comparisons':results,
       'source_unchanged':sha256(a.source)==SOURCE_SHA,'model_paths':{k:str(p) for k,p in finalists},
       'completed_utc':datetime.now(timezone.utc).isoformat()})
    print('FINAL_DECISION',json.dumps(verdict),flush=True)

if __name__=='__main__':main()
