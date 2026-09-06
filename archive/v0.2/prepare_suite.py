"""Pinned external evaluation only. No dataset enters the converter."""
from pathlib import Path
import json,hashlib,random,urllib.request,io
PIN='210d026faf9955653af8916fad021475a3f00453'

def prepare(out):
    import pyarrow.parquet as pq
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    cases=[];sources=[]
    for config in ['ARC-Challenge','ARC-Easy']:
        url=f'https://huggingface.co/datasets/allenai/ai2_arc/resolve/{PIN}/{config}/test-00000-of-00001.parquet'
        raw=urllib.request.urlopen(url,timeout=45).read()
        (out/(config+'.parquet')).write_bytes(raw)
        rows=pq.read_table(io.BytesIO(raw)).to_pylist()
        ids=random.Random(20260905).sample(range(len(rows)),100)
        sources.append({'url':url,'revision':PIN,'sha256':hashlib.sha256(raw).hexdigest(),'indices':ids,'total':len(rows),'license':'CC-BY-SA-4.0'})
        for i in ids:
            x=rows[i];options=list(zip(x['choices']['label'],x['choices']['text']))
            answer_index=[str(a) for a,b in options].index(str(x['answerKey']))
            labels=list('ABCDE')[:len(options)]
            prompt='Answer the multiple-choice question. Write the answer as FINAL: followed by one option letter.\nQuestion: '+x['question']+'\n'
            prompt+='\n'.join(f'{label}. {text}' for label,(_,text) in zip(labels,options))
            prompt+='\nAnswer:'
            cases.append({'id':config+':'+x['id'],'category':config,'prompt':prompt,'check':'mcq','expected':labels[answer_index],
                'options':labels,'answer_text':options[answer_index][1]})
    rng=random.Random(20260906)
    for i in range(32):
        a,b,c=rng.randrange(3,26),rng.randrange(2,20),rng.randrange(1,10)
        q=f'Compute ({a} * {b}) - {c}. Write FINAL: followed by the integer.'
        cases.append({'id':f'new-arith-{i}','category':'procedural-arithmetic','prompt':q,'check':'integer','expected':str(a*b-c)})
    for i in range(32):
        vals=rng.sample(range(1,40),4);op=i%4
        code=[f'print(sum({vals}))',f'print(max({vals})-min({vals}))',f'print(len(set({vals+vals[:2]})))',f'print(sorted({vals})[1])'][op]
        answer=[sum(vals),max(vals)-min(vals),len(set(vals+vals[:2])),sorted(vals)[1]][op]
        cases.append({'id':f'new-code-{i}','category':'procedural-code','prompt':f'What integer does this Python code print? {code}\nWrite FINAL: followed by the integer.','check':'integer','expected':str(answer)})
    obj={'name':'ARC-fixed-200+procedural64-v2','cases':cases,'sources':sources,
         'scope':'200 fixed public test items plus 64 fresh parameterized tasks; not full benchmarks, not guaranteed unseen in pretraining',
         'primary_metric':'completed_answer_correct','secondary_metric':'format_and_answer_correct','evaluation_only':True}
    (out/'suite.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUITE',len(cases),hashlib.sha256((out/'suite.json').read_bytes()).hexdigest(),flush=True)
    return obj
if __name__=='__main__':
    import sys;prepare(sys.argv[1])
