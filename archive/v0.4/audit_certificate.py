"""Audit preserved results plus the label-blind pre-test exclusion addendum."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
from experiment import compare
from stability import parse_final,summarize

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(project):
    project=Path(project);root=project/'study'
    final=json.loads((root/'FINAL_DECISION.json').read_text(encoding='utf-8'))
    path=root/'data/holdout.json';suite=json.loads(path.read_text(encoding='utf-8'))['cases']
    protocol=json.loads((root/'holdout/protocol.json').read_text(encoding='utf-8'))
    add=json.loads((project/'HOLDOUT_AUDIT_ADDENDUM.json').read_text(encoding='utf-8'))
    assert protocol['suite_sha256']==sha(path)
    assert len(suite)==len({x['id'] for x in suite})==160
    assert datetime.fromisoformat(add['declared_local_time'])<datetime.fromisoformat(protocol['created_utc'])
    excluded=set(add['excluded_ids']);assert len(suite)-len(excluded)==add['nonoverlap_certificate_cases']
    ids=[x['id'] for x in suite];expected={x['id']:x for x in suite}
    arms={};raw_hashes={}
    for name in protocol['models']:
        p=root/'holdout'/name/'responses.jsonl'
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==160 and {x['id'] for x in rows}==set(ids)
        for x in rows:
            c=expected[x['id']];assert x['expected']==c['expected']
            pred=parse_final(x['output'],c['check']);assert pred==x['parsed']
            assert x['correct']==(pred==c['expected'] and x['status']=='NATURAL_STOP')
        by_id={x['id']:x for x in rows};arms[name]=[by_id[i] for i in ids if i not in excluded]
        raw_hashes[name]=sha(p)
    comparisons={k:compare(arms['original'],v) for k,v in arms.items() if k!='original'}
    selected=final['selected'];qualified=all(x['release']=='VERIFIED_SCOPED_BENCHMARK_GAIN' for x in (final['result'],comparisons[selected]))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'selected':selected,
      'original_protocol_n':160,'nonoverlap_n':len(arms['original']),
      'pretest_addendum_sha256':sha(project/'HOLDOUT_AUDIT_ADDENDUM.json'),'exclusions':sorted(excluded),
      'raw_data_and_scores_verified':True,'raw_file_hashes':raw_hashes,
      'summaries':{k:summarize(v) for k,v in arms.items()},'comparisons':comparisons,
      'release':'VERIFIED_SCOPED_BENCHMARK_GAIN' if qualified else 'RETAIN_ORIGINAL',
      'scope':'Fixed checkpoint and finite greedy-think budget; not universal ability or unseen task-family generalization.',
      'universal_gain_proven':False,'theoretical_model_limit_broken':False}
    (project/'AUDITED_FINAL_DECISION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);audit(p.parse_args().project)
