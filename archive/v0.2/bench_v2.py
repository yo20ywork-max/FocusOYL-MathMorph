"""Frozen paired generative evaluation; no tool use or learned judge."""
from __future__ import annotations
import sys,json,re,math,socket,subprocess,time,urllib.request,urllib.error,hashlib
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).parent/'src'))
from mathmorph.ggufio import sha256

def parse_answer(text,case):
    # Extraction never reads the expected answer.
    text=text.strip();mode=case['check']
    if mode=='mcq':
        hit=re.findall(r'(?im)\bFINAL\s*:\s*\**\s*([A-E])\b',text)
        if hit:return hit[-1]
        hit=re.fullmatch(r'\s*\**\(?([A-E])\)?\**[.。]?\s*',text)
        if hit:return hit.group(1)
        hit=re.findall(r'(?i)\b(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:)\s*\**\(?([A-E])\b',text)
        if hit and len(set(hit))==1:return hit[0].upper()
        return None
    if mode=='integer':
        hit=re.findall(r'(?im)\bFINAL\s*:\s*\**\s*([+-]?\d+)\b',text)
        if hit:return str(int(hit[-1]))
        if re.fullmatch(r'[+-]?\d+',text):return str(int(text))
        return None
    raise ValueError('Unsupported grader')

def paired(a,b):
    if not a or len(a)!=len(b):raise ValueError('Nonempty equal paired vectors required')
    n=len(a);w=sum(not x and y for x,y in zip(a,b));l=sum(x and not y for x,y in zip(a,b));d=w+l
    # Exact one-sided paired randomization/binomial test, conditional on discordance.
    p=sum(math.comb(d,k) for k in range(w,d+1))/(2**d) if d else 1.
    return {'n':n,'baseline':sum(a),'candidate':sum(b),'delta_pp':100*(sum(b)-sum(a))/n,
        'wins':w,'losses':l,'ties':n-d,'one_sided_exact_p':p,
        'p_assumption':'exchangeability of paired correctness under null; samples do not certify arbitrary tasks'}

def post(base,route,payload):
    req=urllib.request.Request(base+route,data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)

def run_model(server,model,cases,out,max_tokens=192,gpu_layers=99):
    out.mkdir(exist_ok=False);model_hash=sha256(model);alias='mathmorph-'+model_hash[:16]
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    cmd=[str(server),'-m',str(model),'--host','127.0.0.1','--port',str(port),'-ngl',str(gpu_layers),
         '-t','6','-c','4096','--parallel','1','--jinja','--alias',alias]
    (out/'launch.json').write_text(json.dumps({'command':cmd,'sha256':model_hash}),encoding='utf-8')
    results=[]
    with (out/'server.log').open('wb') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        try:
            end=time.monotonic()+180
            while True:
                if p.poll() is not None:raise RuntimeError('Server startup failed; see server.log')
                try:
                    with urllib.request.urlopen(base+'/health',timeout=2) as r:
                        if r.status==200:break
                except OSError:pass
                if time.monotonic()>end:raise TimeoutError('Server readiness timeout')
                time.sleep(.3)
            with urllib.request.urlopen(base+'/v1/models',timeout=5) as r:identity=json.load(r)
            if not any(x.get('id')==alias for x in identity['data']):raise RuntimeError('Wrong model identity')
            (out/'identity.json').write_text(json.dumps(identity),encoding='utf-8')
            probe={'model':alias,'messages':[{'role':'user','content':'Compute 13 plus 29. Answer briefly.'}],
                'max_tokens':1,'temperature':1.,'top_k':0,'top_p':1.,'min_p':0.,'seed':20260905,
                'logprobs':True,'top_logprobs':5,'chat_template_kwargs':{'enable_thinking':False},'cache_prompt':False}
            try:
                canary=post(base,'/v1/chat/completions',probe)
            except Exception as exc:canary={'probe_error':str(exc),'not_used_for_scoring':True}
            (out/'canary.json').write_text(json.dumps(canary,ensure_ascii=False),encoding='utf-8')
            for i,case in enumerate(cases):
                payload={'model':alias,'messages':[{'role':'user','content':case['prompt']}],
                    'temperature':0,'seed':20260905,'max_tokens':max_tokens,'stream':False,
                    'chat_template_kwargs':{'enable_thinking':False},'cache_prompt':False}
                start=time.monotonic();data=post(base,'/v1/chat/completions',payload)
                choice=data['choices'][0];m=choice['message'];text=m.get('content') or ''
                answer=parse_answer(text,case);finish=choice.get('finish_reason')
                correct=answer==case['expected'];strict=bool(re.fullmatch(r'FINAL:\s*'+re.escape(case['expected']),text.strip()))
                row={'id':case['id'],'category':case['category'],'output':text,'parsed':answer,
                    'answer_correct':correct,'completed_answer_correct':correct and finish=='stop',
                    'format_and_answer_correct':correct and strict and finish=='stop','finish_reason':finish,
                    'seconds':time.monotonic()-start,'usage':data.get('usage'),'model_sha256':model_hash,
                    'reasoning':m.get('reasoning_content') or m.get('reasoning')}
                results.append(row)
                with (out/'responses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                if (i+1)%25==0 or i+1==len(cases):
                    print(f'EVAL {model.name} {i+1}/{len(cases)} completed_correct={sum(x["completed_answer_correct"] for x in results)}',flush=True)
            if sha256(model)!=model_hash:raise RuntimeError('Model changed during evaluation')
            return results
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)

def summarize(root):
    root=Path(root);f=json.loads((root/'evaluation/protocol.json').read_text(encoding='utf-8'))
    labels=list(f['models']);obs={k:[json.loads(x) for x in (root/'evaluation'/k/'responses.jsonl').read_text(encoding='utf-8').splitlines()] for k in labels}
    n=f['n'];expected_ids=f['case_ids']
    if any(len(rows)!=n or [r['id'] for r in rows]!=expected_ids for rows in obs.values()):
        raise ValueError('Incomplete or mismatched evaluation')
    metrics=['completed_answer_correct','answer_correct','format_and_answer_correct'];comparisons={}
    # Family includes two signs x two model families x (overall + 4 categories).
    multiplicity=20
    for k in labels:
        if k=='original':continue
        per={}
        for cat in ['overall']+sorted(set(x['category'] for x in obs[k])):
            ids=[i for i,x in enumerate(obs[k]) if cat=='overall' or x['category']==cat]
            per[cat]={m:paired([obs['original'][i][m] for i in ids],[obs[k][i][m] for i in ids]) for m in metrics}
            p=per[cat][metrics[0]]['one_sided_exact_p'];per[cat]['primary_p_bonferroni']=min(1,p*multiplicity)
        comparisons[k]=per
    summary={'version':'0.2.0','n':n,'primary_metric':metrics[0],'comparisons':comparisons,
        'truncated':{k:sum(x['finish_reason']=='length' for x in v) for k,v in obs.items()},
        'identical_outputs':{k:sum(a['output']==b['output'] for a,b in zip(obs['original'],v)) for k,v in obs.items() if k!='original'},
        'scope':f['scope'],'universal_gain_proven':False,'model_limit_broken':False,
        'multiple_testing':{'planned_primary_tests':multiplicity,'method':'Bonferroni','alpha':.05},
        'status':'NO_VERIFIED_GENERAL_GAIN'}
    summary['observed_subset_gains']=[k for k,v in comparisons.items() if v['overall'][metrics[0]]['delta_pp']>0]
    summary['statistically_supported_subset_gains']=[k for k,v in comparisons.items() if v['overall']['primary_p_bonferroni']<.05]
    (root/'evaluation/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True);return summary

def evaluate(root,server,suite):
    root=Path(root);server=Path(server).resolve(strict=True);suite=Path(suite)
    r=json.loads((root/'report.json').read_text(encoding='utf-8'))
    if r['structural_status']!='PASS':raise ValueError('Unverified structure')
    d=json.loads(suite.read_text(encoding='utf-8'));cases=d['cases']
    folder=root/'evaluation';folder.mkdir(exist_ok=False)
    models={'original':{'path':r['input'],'sha256':r['source_sha256']},**r['outputs']}
    for x in models.values():
        if sha256(x['path'])!=x['sha256']:raise ValueError('Model hash mismatch')
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'models':models,'suite_sha256':sha256(suite),
        'server_sha256':sha256(server),'evaluator_sha256':sha256(__file__),'n':len(cases),'case_ids':[x['id'] for x in cases],
        'scope':d['scope'],'max_tokens':192,'gpu_layers':99,'temperature':0,'seed':20260905,
        'primary_metric':'completed_answer_correct','planned_tests_bonferroni':20,'answers_used_for_conversion':False}
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    try:
        for label,m in models.items():run_model(server,Path(m['path']),cases,folder/label)
        return summarize(root)
    except BaseException as exc:
        (folder/'FAILED.json').write_text(json.dumps({'status':'INVALID_INCOMPLETE','error':str(exc)}),encoding='utf-8')
        raise
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--server',type=Path,required=True);p.add_argument('--suite',type=Path,required=True)
    a=p.parse_args();evaluate(a.run,a.server,a.suite)
