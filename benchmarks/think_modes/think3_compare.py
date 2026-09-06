"""Three-arm GGUF thinking comparison. No weight writes, training or answer-based tuning."""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, math, re, socket, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

EXPECTED = {
    'original': '68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d',
    'sharpen': '5a9084ab6ce3ac20126cd6edbcbe11ffb42b220a292d1e6d266cf377eb064f72',
}
ARMS = [('original_think_on', 'original', True), ('sharpen_think_on', 'sharpen', True),
        ('sharpen_think_off', 'sharpen', False)]
LIMIT = 2048
WORKERS = 4
SEED = 20260905


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def post(base, route, data=None, timeout=240):
    req = Request(base+route, data=json.dumps(data, ensure_ascii=False).encode() if data is not None else None,
                  headers={'Content-Type':'application/json'})
    with urlopen(req, timeout=timeout) as r: return json.load(r)


def parse(text, kind):
    # Expected answers are intentionally absent from this function.
    if kind == 'mcq':
        hits = re.findall(r'(?im)\bFINAL\s*:\s*\**\s*([A-E])\b', text)
        if hits: return hits[-1].upper()
        hit = re.fullmatch(r'\s*\**\(?([A-E])\)?\**[.。]?\s*', text)
        if hit: return hit.group(1).upper()
        hits = re.findall(r'(?i)\b(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:)\s*\**\(?([A-E])\b', text)
        return hits[0].upper() if hits and len(set(x.upper() for x in hits)) == 1 else None
    if kind != 'integer': raise ValueError(kind)
    pats = [r'(?im)\bFINAL\s*:\s*\**\s*([+-]?\d+)\b',
            r'(?i)\b(?:the\s+)?final\s+answer\s+is\s*[:=]?\s*\**\s*([+-]?\d+)\b',
            r'(?i)\b(?:the\s+)?final\s+(?:answer|result)\s*[:=]\s*\**\s*([+-]?\d+)\b']
    hits = [(m.start(), m.group(1)) for p in pats for m in re.finditer(p, text)]
    if hits: return str(int(max(hits, key=lambda x:x[0])[1]))
    return str(int(text.strip())) if re.fullmatch(r'[+-]?\d+', text.strip()) else None


def paired(a, b):
    w=sum(not x and y for x,y in zip(a,b)); l=sum(x and not y for x,y in zip(a,b)); d=w+l
    p=min(1., 2*sum(math.comb(d,k) for k in range(min(w,l)+1))/(2**d)) if d else 1.
    return {'n':len(a),'first_correct':sum(a),'second_correct':sum(b),
            'second_wins':w,'second_losses':l,'delta_pp':100*(sum(b)-sum(a))/len(a),
            'two_sided_exact_p':p,'p_adjusted_3_pairs':min(1.,p*3)}


def selftest():
    for text,kind,ans in [('FINAL: A','mcq','A'),('The correct answer is (B).','mcq','B'),
                          ('Let us consider A and D.','mcq',None),('FINAL: -87','integer','-87'),
                          ('The final answer is 42.','integer','42'),('3 * 4 = 12','integer',None),
                          ('FINAL: 2\nFinal result: 7','integer','7'),('87','integer','87')]:
        assert parse(text,kind)==ans,(text,parse(text,kind))
    assert paired([False]*10,[True]*10)['two_sided_exact_p']==2/1024


def run_arm(label, model, think, cases, server, root):
    out=root/label; out.mkdir(exist_ok=False)
    alias=label+'-'+sha(model)[:12]
    with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    cmd=[str(server),'-m',str(model),'--host','127.0.0.1','--port',str(port),'-ngl','99',
         '-t','2','-tb','2','-c',str(WORKERS*4096),'--parallel',str(WORKERS),'--jinja',
         '--reasoning','on' if think else 'off','--reasoning-format','deepseek',
         '--reasoning-budget','-1','--alias',alias]
    save(out/'launch.json',{'command':cmd,'model_sha256':sha(model)})
    log=(out/'server.log').open('wb')
    proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    rows=[]
    try:
        deadline=time.monotonic()+180
        while True:
            if proc.poll() is not None: raise RuntimeError('Server startup failed: '+str(out/'server.log'))
            try: post(base,'/health',timeout=2); break
            except OSError: pass
            if time.monotonic()>deadline: raise TimeoutError('server readiness')
            time.sleep(.3)
        identity=post(base,'/v1/models'); save(out/'identity.json',identity)
        if not any(x.get('id')==alias for x in identity['data']): raise RuntimeError('Wrong model')
        props=post(base,'/props'); save(out/'props.json',props)
        if props['default_generation_settings']['n_ctx']<4096: raise RuntimeError('Context too small')
        probe={'messages':[{'role':'user','content':'Compute (13 * 7) - 4. Write FINAL: followed by the integer.'}],
               'chat_template_kwargs':{'enable_thinking':think}}
        template=post(base,'/apply-template',probe); save(out/'template.json',template)
        tail='<|im_start|>assistant\n<think>\n' if think else '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        if not template.get('prompt','').endswith(tail): raise RuntimeError('Thinking template switch did not apply')
        common={'model':alias,'temperature':0.,'top_k':0,'top_p':1.,'min_p':0.,'repeat_penalty':1.,
                'seed':SEED,'max_tokens':LIMIT,'stream':False,'cache_prompt':False,
                'chat_template_kwargs':{'enable_thinking':think}}
        canary=post(base,'/v1/chat/completions',{**common,'messages':probe['messages']})
        save(out/'mode_probe.json',canary)
        message=canary['choices'][0]['message']
        reasoning=message.get('reasoning_content') or message.get('reasoning') or ''
        if bool(reasoning)!=think: raise RuntimeError('Probe reasoning field disagrees with mode')
        print('MODE_VERIFIED',label,'reasoning_chars',len(reasoning),flush=True)
        def one(case):
            before=time.monotonic()
            response=post(base,'/v1/chat/completions',{**common,'messages':[{'role':'user','content':case['prompt']}]})
            seconds=time.monotonic()-before
            c=response['choices'][0]; m=c['message']; text=m.get('content') or ''
            reasoning=m.get('reasoning_content') or m.get('reasoning') or ''
            pred=parse(text,case['check']); finish=c.get('finish_reason'); correct=pred==case['expected']
            # Only final content is graded. Reasoning is never fed to the scorer.
            r={'id':case['id'],'category':case['category'],'suite':case['suite'],'parsed':pred,
               'answer_correct':correct,'completed_correct':bool(correct and finish=='stop'),
               'strict_correct':bool(correct and finish=='stop' and re.fullmatch(r'FINAL:\s*'+re.escape(case['expected']),text.strip())),
               'finish_reason':finish,'seconds':seconds,'usage':response.get('usage',{}),
               'timings':response.get('timings',{}),'reasoning_chars':len(reasoning),
               'final_chars':len(text),'reasoning_retokenized':None,
               'output':text,'reasoning':reasoning,'raw':response}
            if reasoning:
                r['reasoning_retokenized']=len(post(base,'/tokenize',{'content':reasoning,'add_special':False,'parse_special':True})['tokens'])
            return r
        start=time.monotonic()
        with (out/'responses.jsonl').open('x',encoding='utf-8') as f, cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i,r in enumerate(pool.map(one,cases)):
                rows.append(r); f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush()
                if (i+1)%24==0 or i+1==len(cases):
                    print('EVAL',label,f'{i+1}/{len(cases)}','correct',sum(x['completed_correct'] for x in rows),
                          'truncated',sum(x['finish_reason']=='length' for x in rows),flush=True)
        elapsed=time.monotonic()-start
        save(out/'runtime.json',{'evaluation_wall_seconds':elapsed,'probe_and_load_excluded':True})
        if sha(model)!=EXPECTED['original' if label.startswith('original') else 'sharpen']: raise RuntimeError('Weight file changed')
        return rows
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=15)
        log.close()


def summarize(root):
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    cases=json.loads((root/'suite.json').read_text(encoding='utf-8'))['cases']
    data={label:[json.loads(x) for x in (root/label/'responses.jsonl').read_text(encoding='utf-8').splitlines()] for label,_,_ in ARMS}
    ids=[x['id'] for x in cases]
    for rows in data.values():
        assert [x['id'] for x in rows]==ids,'Missing or out-of-order observations'
    summary={'protocol':protocol,'n_per_arm':len(cases),'groups':{},'pairwise':{},'cost':{},'source_unchanged':True,
             'scope':'Paired rerun of previously used tasks; not new generalization evidence; 48 controls repeat earlier expressions.'}
    groups={'mixed-264':[i for i,x in enumerate(cases) if x['suite']=='discovery']}
    groups.update({cat:[i for i,x in enumerate(cases) if x['category']==cat] for cat in sorted(set(x['category'] for x in cases))})
    for group,indices in groups.items():
        summary['groups'][group]={}
        for label,_,_ in ARMS:
            v=[data[label][i] for i in indices]
            summary['groups'][group][label]={'n':len(v),'correct':sum(x['completed_correct'] for x in v),
                'correct_even_if_truncated':sum(x['answer_correct'] for x in v),'strict_correct':sum(x['strict_correct'] for x in v),
                'truncated':sum(x['finish_reason']=='length' for x in v),'unparsed':sum(x['parsed'] is None for x in v),
                'completion_tokens':sum(x['usage']['completion_tokens'] for x in v),
                'reasoning_nonempty':sum(x['reasoning_chars']>0 for x in v),'sum_seconds':sum(x['seconds'] for x in v)}
        summary['pairwise'][group]={}
        for a,b in [(ARMS[0][0],ARMS[1][0]),(ARMS[0][0],ARMS[2][0]),(ARMS[1][0],ARMS[2][0])]:
            summary['pairwise'][group][a+' vs '+b]=paired([data[a][i]['completed_correct'] for i in indices],[data[b][i]['completed_correct'] for i in indices])
    for label,_,_ in ARMS:
        v=data[label]; timings=json.loads((root/label/'runtime.json').read_text())
        n=len(v); tokens=sum(x['usage']['completion_tokens'] for x in v); sec=sum(x['seconds'] for x in v)
        summary['cost'][label]={'n':n,'total_completion_tokens':tokens,'mean_completion_tokens':tokens/n,
           'mean_response_seconds_4_concurrent':sec/n,'evaluation_wall_seconds':timings['evaluation_wall_seconds'],
           'reasoning_nonempty':sum(x['reasoning_chars']>0 for x in v),'truncated':sum(x['finish_reason']=='length' for x in v),
           'retokenized_reasoning_tokens_approx':sum(x['reasoning_retokenized'] or 0 for x in v)}
    save(root/'summary.json',summary)
    compact={'protocol':protocol,'cases':cases,'summary':summary,'arms':{},'raw_file_hashes':{},'mode_probes':{}}
    for label,_,_ in ARMS:
        fields=['id','category','suite','parsed','answer_correct','completed_correct','strict_correct','finish_reason',
                'seconds','usage','timings','reasoning_chars','final_chars','reasoning_retokenized']
        compact['arms'][label]=[{k:r[k] for k in fields} for r in data[label]]
        compact['raw_file_hashes'][label]=sha(root/label/'responses.jsonl')
        probe=json.loads((root/label/'mode_probe.json').read_text(encoding='utf-8')); msg=probe['choices'][0]['message']
        compact['mode_probes'][label]={'template':json.loads((root/label/'template.json').read_text()),
             'final_content':msg.get('content'),'reasoning_chars':len(msg.get('reasoning_content') or msg.get('reasoning') or ''),
             'usage':probe['usage'],'raw_sha256':sha(root/label/'mode_probe.json')}
    save(root/'COMPACT_EVIDENCE.json',compact)
    print('COMPLETE',json.dumps(summary['groups'],ensure_ascii=False),flush=True)
    print('COST',json.dumps(summary['cost']),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--previous',type=Path); parser.add_argument('--root',type=Path)
    parser.add_argument('--server',type=Path); parser.add_argument('--source',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();selftest()
    if args.self_test: print('9 tests passed');return
    root=args.root;root.mkdir(exist_ok=True)
    if (root/'protocol.json').exists(): raise FileExistsError('Run exists; refusing to overwrite')
    candidates={'original':args.source.resolve(strict=True),'sharpen':(args.previous/'runs/minicpm5-npsr/sharpen.gguf').resolve(strict=True)}
    for name,p in candidates.items():
        if sha(p)!=EXPECTED[name]: raise RuntimeError('Model hash mismatch: '+name)
    cases=[];source_suites={}
    for name,rel in [('discovery','evaluation-data/suite.json'),('confirmation','confirmation/suite.json')]:
        path=args.previous/rel;source_suites[name]={'path':str(path),'sha256':sha(path)}
        cases += [{**x,'suite':name} for x in json.loads(path.read_text(encoding='utf-8'))['cases']]
    assert len(cases)==456 and len(set(x['id'] for x in cases))==456
    save(root/'suite.json',{'cases':cases,'source_suites':source_suites})
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'n':len(cases),'arms':ARMS,
         'models':{k:{'path':str(p),'sha256':sha(p)} for k,p in candidates.items()},
         'server':str(args.server),'server_sha256':sha(args.server),'code_sha256':sha(__file__),
         'suite_sha256':sha(root/'suite.json'),'source_suites':source_suites,'max_tokens_including_reasoning':LIMIT,
         'temperature':0,'top_k':0,'top_p':1.,'min_p':0.,'seed':SEED,'parallel_requests':WORKERS,
         'context_per_request':4096,'gpu_layers':99,'threads':2,'primary':'mixed-264/completed_correct',
         'thinking_control':'server --reasoning on/off plus per-request enable_thinking; template and reasoning-field probes',
         'reasoning_budget':-1,'no_forced_early_thinking_end':True,'cache_prompt':False,
         'scoring':'final content only, previous MCQ and confirmation integer extraction; no gold used in parser',
         'official_sampling_preset':False,'latency_caveat':'4 concurrent requests; another user HF evaluation was already running; not an isolated speed benchmark',
         'no_training_or_weight_modification':True,'rerun_of_previous_questions':True}
    save(root/'protocol.json',protocol)
    try:
        for label,name,think in ARMS: run_arm(label,candidates[name],think,cases,args.server,root)
        for name,p in candidates.items():
            if sha(p)!=EXPECTED[name]:raise RuntimeError('Source hash changed')
        summarize(root)
    except BaseException as exc:
        save(root/'FAILED.json',{'error':repr(exc),'time':datetime.now(timezone.utc).isoformat()});raise

if __name__=='__main__':main()
