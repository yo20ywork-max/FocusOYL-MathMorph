"""New frozen arithmetic confirmation after v2 discovery, without changing weights.

This is a scoped confirmation, not validation of universal model improvement.
Primary and composition examples are disjoint from the discovery expressions.
The explicit-step subset repeats 48 primary items with an identical added
instruction for both models, and is a correlated mechanism control.
"""
from __future__ import annotations
import argparse,json,random,re,sys
from pathlib import Path
from datetime import datetime,timezone
import bench_v2 as bench


def semantic_integer(text,case):
    if case['check']!='integer':raise ValueError('Integer-only confirmation')
    patterns=[r'(?im)\bFINAL\s*:\s*\**\s*([+-]?\d+)\b',
              r'(?i)\b(?:the\s+)?final\s+answer\s+is\s*[:=]?\s*\**\s*([+-]?\d+)\b',
              r'(?i)\b(?:the\s+)?final\s+(?:answer|result)\s*[:=]\s*\**\s*([+-]?\d+)\b']
    # Use the last explicitly marked answer, not any occurrence of the target.
    hits=[(m.start(),m.group(1)) for p in patterns for m in re.finditer(p,text)]
    if hits:return str(int(max(hits,key=lambda x:x[0])[1]))
    t=text.strip()
    if re.fullmatch(r'[+-]?\d+',t):return str(int(t))
    return None


def self_test():
    c={'check':'integer','expected':'999'}
    checks=[('FINAL: 85','85'),('Final: **-7**','-7'),('The final answer is 100.','100'),
            ('So, the final answer is 42.','42'),('Final result: 17','17'),
            ('100 is mentioned but FINAL: 99','99'),('FINAL: 7\nFinal answer is 8','8'),
            ('12 * 3 = 36',None),('17','17'),('The final answer is 11111.','11111')]
    for text,expected in checks:assert semantic_integer(text,c)==expected,(text,semantic_integer(text,c))
    for text,_ in checks:assert semantic_integer(text,c)==semantic_integer(text,{**c,'expected':'-2'})


def prepare(discovery_path:Path):
    old=json.loads(discovery_path.read_text(encoding='utf-8'))['cases']
    used=set()
    for x in old:
        m=re.search(r'Compute \((\d+) \* (\d+)\) - (\d+)',x['prompt'])
        if m:
            a,b,c=map(int,m.groups());used.add((min(a,b),max(a,b),c))
    rng=random.Random(2026090502);cases=[]
    while len(cases)<96:
        a,b,c=rng.randrange(3,26),rng.randrange(2,20),rng.randrange(1,10)
        key=(min(a,b),max(a,b),c)
        if key in used:continue
        used.add(key)
        cases.append({'id':f'confirm-mulsub-{len(cases)}','category':'primary-mulsub',
            'prompt':f'Compute ({a} * {b}) - {c}. Write FINAL: followed by the integer.',
            'check':'integer','expected':str(a*b-c)})
    unique=set()
    for i in range(48):
        while True:
            a,b,c,d=rng.randrange(3,19),rng.randrange(2,14),rng.randrange(1,10),rng.randrange(1,10)
            expr=f'({a} + {b}) * {c}' if i<24 else f'({a} * {b}) + ({c} * {d})'
            if expr not in unique:unique.add(expr);break
        ans=(a+b)*c if i<24 else a*b+c*d
        cases.append({'id':f'confirm-compose-{i}','category':'secondary-composition',
            'prompt':f'Compute {expr}. Write FINAL: followed by the integer.', 'check':'integer','expected':str(ans)})
    for i,x in enumerate(cases[:48]):
        cases.append({**x,'id':f'control-steps-{i}','category':'control-explicit-steps',
          'prompt':x['prompt']+' Work step by step before stating the final answer.'})
    return cases


def run(discovery,source,candidate,server,out):
    self_test();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cases=prepare(Path(discovery));suite={'cases':cases,'seed':2026090502,
       'scope':'96 fresh mul-sub items, 48 composition items, 48 correlated explicit-step controls',
       'selection':'MiniCPM sharpen selected after discovery; weights and original discovery scores unchanged',
       'semantic_grader':'fixed from discovery audit before this new set is evaluated',
       'not_a_general_capability_benchmark':True}
    (out/'suite.json').write_text(json.dumps(suite,ensure_ascii=False,indent=2),encoding='utf-8')
    models={'original':Path(source).resolve(strict=True),'sharpen':Path(candidate).resolve(strict=True)}
    proto={'frozen_utc':datetime.now(timezone.utc).isoformat(),'n':len(cases),
       'models':{k:{'path':str(v),'sha256':bench.sha256(v)} for k,v in models.items()},
       'suite_sha256':bench.sha256(out/'suite.json'),'confirm_code_sha256':bench.sha256(__file__),
       'runner_code_sha256':bench.sha256(bench.__file__),'server_sha256':bench.sha256(server),
       'max_tokens':192,'gpu_layers':99,'temperature':0,'seed':20260905,
       'primary':'primary-mulsub/completed_answer_correct','multiplicity':3,
       'secondary_and_control_not_independent_task_families':True,
       'weights_changed_after_discovery':False,'task_answers_used_for_conversion':False}
    (out/'protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf-8')
    bench.parse_answer=semantic_integer
    rows={k:bench.run_model(Path(server).resolve(strict=True),v,cases,out/k) for k,v in models.items()}
    stats={}
    for group in ['primary-mulsub','secondary-composition','control-explicit-steps']:
        selected={k:[x for x in v if x['category']==group] for k,v in rows.items()}
        s=bench.paired([x['completed_answer_correct'] for x in selected['original']],
                       [x['completed_answer_correct'] for x in selected['sharpen']])
        s['p_bonferroni_3']=min(1,s['one_sided_exact_p']*3)
        s['completion_tokens']={k:sum(x['usage']['completion_tokens'] for x in v) for k,v in selected.items()}
        s['truncated']={k:sum(x['finish_reason']=='length' for x in v) for k,v in selected.items()}
        stats[group]=s
    for k,v in models.items():assert bench.sha256(v)==proto['models'][k]['sha256']
    summary={'kind':'scoped_arithmetic_confirmation','groups':stats,'model_limit_broken':False,
       'universal_gain_proven':False,'general_upgrade_release':'REJECTED_DUE_TO_DISCOVERY_REGRESSIONS'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');
    for name in ['discovery','source','candidate','server','out']:p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.self_test:self_test();print('10 parser cases and target-independence checks passed')
    else:
        if not all(getattr(a,k) for k in ['discovery','source','candidate','server','out']):p.error('All paths required')
        run(a.discovery,a.source,a.candidate,a.server,a.out)
