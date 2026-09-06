"""Recompute archived scores without inference or network access."""
from pathlib import Path
import json,math,collections
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def paired(a,b):
 if len(a)!=len(b):raise ValueError('Unequal paired sample counts')
 wins=sum(not x[1] and y[1] for x,y in zip(a,b))
 losses=sum(x[1] and not y[1] for x,y in zip(a,b));n=wins+losses
 p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
 return {'n':len(a),'baseline_correct':sum(x[1] for x in a),'candidate_correct':sum(x[1] for x in b),
         'wins':wins,'losses':losses,'two_sided_p':p,'two_comparison_p':min(1.,2*p)}
def main():
 first=load('evidence/v0.4/SCORE_EVIDENCE.json');phase=first['phases']['holdout']
 excluded=set(first['addendum']['excluded_ids'])
 indexes=[i for i,c in enumerate(phase['cases']) if c[0] not in excluded]
 arms={k:[v['rows'][i] for i in indexes] for k,v in phase['arms'].items()}
 assert len(indexes)==156
 out={'v04_nonoverlap':{k:paired(arms['original'],v) for k,v in arms.items() if k!='original'}}
 sweep=load('evidence/v0.4/SWEEP_SCORE_EVIDENCE.json')['phases']['development_run']
 out['v04_second_development']={k:{'n':len(v['rows']),'correct':sum(x[1] for x in v['rows']),
  'statuses':dict(collections.Counter(x[2] for x in v['rows']))} for k,v in sweep['arms'].items()}
 samples=load('evidence/localbench/latest_incomplete/sample_metrics.json')
 out['incomplete_standard_task_snapshot']={}
 for arm in ['original','prism']:
  rows=samples[arm+'/gsm8k_cot_zeroshot']['rows'];metrics={}
  for name in ['strict-match','flexible-extract']:
   selected=[r for r in rows if r['filter']==name]
   assert len(selected)==1319
   metrics[name]={'n':len(selected),'correct':sum(x['exact_match'] for x in selected),
                  'score':sum(x['exact_match'] for x in selected)/len(selected)}
  out['incomplete_standard_task_snapshot'][arm]=metrics
 out['note']='Read-only recalculation, not new inference or a completed combined evaluation.'
 print(json.dumps(out,ensure_ascii=False,indent=2))
 return out
if __name__=='__main__':main()
