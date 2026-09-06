"""Short paired regression runner. Loop aborts are failures, not corrected answers.

Stops only its OWN server, never changes weights/templates, saves traces before
cleanup, and checks CANCEL before scheduling any more requests.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, os, secrets, socket, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import requests
from guarded import save, sha256
from stability import LoopMonitor, lexical, parse_final, summarize, assess


def one(base, headers, alias, case, index, folder, cancel, limit, seconds, think):
    start=time.monotonic(); reason=''; final=''; finish=None; usage={}; status='TRANSPORT_ERROR'; error=None
    monitor=LoopMonitor(); last_save=start; chunks=0
    request={'model':alias,'messages':[{'role':'user','content':case['prompt']}], 'max_tokens':limit,
        'temperature':0.,'top_p':1.,'top_k':0,'min_p':0.,'repeat_penalty':1.,'seed':20260905,
        'cache_prompt':False,'stream':True,'stream_options':{'include_usage':True},
        'chat_template_kwargs':{'enable_thinking':think}}
    try:
        if cancel.exists(): status='USER_CANCELLED'
        else:
            with requests.post(base+'/v1/chat/completions',json=request,headers=headers,
                               stream=True,timeout=(10,30)) as response:
                response.raise_for_status()
                for raw in response.iter_lines(chunk_size=512):
                    if cancel.exists(): status='USER_CANCELLED'; break
                    if time.monotonic()-start > seconds: status='WALL_TIME_CENSORED'; break
                    if not raw.startswith(b'data:'): continue
                    payload=raw[5:].strip()
                    if payload==b'[DONE]': break
                    part=json.loads(payload)
                    if part.get('error'): raise RuntimeError(str(part['error']))
                    if part.get('usage'): usage=part['usage']
                    for choice in part.get('choices',[]):
                        delta=choice.get('delta',{})
                        reason+=delta.get('reasoning_content') or delta.get('reasoning') or ''
                        final+=delta.get('content') or ''
                        finish=choice.get('finish_reason') or finish
                    chunks+=1
                    if monitor.update(reason,final): status='LOOP_GUARD_ABORT'; break
                    if time.monotonic()-last_save >= 5:
                        save(folder/f'{index:04d}.partial.json', {'id':case['id'],'reasoning':reason,'output':final,
                            'status':'RUNNING','seconds':time.monotonic()-start})
                        last_save=time.monotonic()
                if status not in ('LOOP_GUARD_ABORT','USER_CANCELLED','WALL_TIME_CENSORED'):
                    status='NATURAL_STOP' if finish=='stop' else ('TOKEN_LIMIT' if finish=='length' else 'TRANSPORT_ERROR')
    except Exception as exc:
        error=repr(exc)
        status='USER_CANCELLED' if cancel.exists() else 'TRANSPORT_ERROR'
    parsed=parse_final(final,case['check'])
    row={'id':case['id'],'category':case['category'],'status':status,'finish_reason':finish,
        'parsed':parsed,'expected':case['expected'],'correct':status=='NATURAL_STOP' and parsed==case['expected'],
        'reasoning':reason,'output':final,'reasoning_chars':len(reason),'final_chars':len(final),
        'lexical':lexical(reason),'completion_tokens':usage.get('completion_tokens'),
        'stream_chunks_not_tokens':chunks,'usage':usage,'seconds':time.monotonic()-start,'error':error}
    save(folder/f'{index:04d}.json',row)
    (folder/f'{index:04d}.partial.json').unlink(missing_ok=True)
    return row


def run_arm(label, model, digest, server, out, cases, cfg):
    root=out/label;root.mkdir();(root/'items').mkdir()
    key=secrets.token_hex(24);headers={'Authorization':'Bearer '+key}
    with socket.socket() as s: s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}';alias=label+'-'+digest[:12]
    cmd=[str(server),'-m',str(model),'--host','127.0.0.1','--port',str(port),
         '-ngl',str(cfg['gpu_layers']),'-t','2','-tb','2','-c',str(cfg['workers']*cfg['context']),
         '--parallel',str(cfg['workers']),'--no-kv-unified','--no-context-shift','--cache-ram','0',
         '--jinja','--reasoning','on' if cfg['think'] else 'off','--reasoning-format','deepseek',
         '--reasoning-budget','-1','--alias',alias,'--api-key',key]
    save(root/'launch.json',{'command':cmd[:-1]+['<ephemeral-redacted>'],'model_sha256':digest})
    rows=[]; proc=None; start=time.monotonic(); log=(root/'server.log').open('wb')
    try:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        until=time.monotonic()+120
        while True:
            if proc.poll() is not None: raise RuntimeError('Server exited during load: '+str(root/'server.log'))
            try:
                h=requests.get(base+'/health',headers=headers,timeout=2)
                if h.ok: break
            except requests.RequestException: pass
            if time.monotonic()>until: raise TimeoutError('Server load')
            if (out/'CANCEL').exists(): raise InterruptedError('Cancelled during load')
            time.sleep(.25)
        identity=requests.get(base+'/v1/models',headers=headers,timeout=5).json()
        save(root/'identity.json',identity)
        if not any(x.get('id')==alias for x in identity['data']): raise RuntimeError('Wrong served model')
        props=requests.get(base+'/props',headers=headers,timeout=5).json();save(root/'props.json',props)
        params=props['default_generation_settings']['params']
        if props['default_generation_settings']['n_ctx'] < cfg['context']:
            raise RuntimeError('Context mismatch')
        template=requests.post(base+'/apply-template',headers=headers,json={'messages':[{'role':'user','content':'Mode check.'}],
            'chat_template_kwargs':{'enable_thinking':cfg['think']}},timeout=5)
        template.raise_for_status();t=template.json();save(root/'template.json',t)
        if cfg['think'] and not t.get('prompt','').rstrip().endswith('<think>'):
            raise RuntimeError('Think template verification failed; no benchmark run')
        # Bound the queue to the active workers instead of pre-submitting all tasks.
        iterator=iter(enumerate(cases)); pending={}
        with cf.ThreadPoolExecutor(max_workers=cfg['workers']) as pool, (root/'responses.jsonl').open('x',encoding='utf-8') as f:
            def submit_next():
                if (out/'CANCEL').exists(): return
                try: i,c=next(iterator)
                except StopIteration: return
                fut=pool.submit(one,base,headers,alias,c,i,root/'items',out/'CANCEL',cfg['max_tokens'],cfg['seconds'],cfg['think'])
                pending[fut]=i
            for _ in range(cfg['workers']):submit_next()
            while pending:
                done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    pending.pop(future);r=future.result();rows.append(r)
                    f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                    print('SCREEN',label,len(rows),'/',len(cases),r['id'],r['status'],r['correct'],flush=True)
                    submit_next()
        order={c['id']:i for i,c in enumerate(cases)};rows.sort(key=lambda r:order[r['id']])
        if cfg['think'] and rows and not any(r['reasoning_chars'] for r in rows):
            raise RuntimeError('Think mode emitted no reasoning')
        if sha256(model)!=digest: raise RuntimeError('Model modified externally')
        return rows
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        log.close()
        save(root/'cleanup.json',{'owned_pid':proc.pid if proc else None,'owned_server_exited':proc is None or proc.poll() is not None,
            'ephemeral_key_not_persisted':True,'disk_prompt_cache_enabled':False,
            'in_process_kv_released_with_server':True,'raw_traces_preserved':True,'wall_seconds':time.monotonic()-start})


def run(models, server, suite_path, out, cfg):
    out=Path(out).resolve();server=Path(server).resolve(strict=True)
    if out.exists(): raise FileExistsError('NEW screen directory required')
    if not isinstance(cfg['max_tokens'],int) or not 1<=cfg['max_tokens']<cfg['context']-1024:
        raise ValueError('Finite regression token budget with prompt reserve required')
    if cfg['workers'] not in (1,2) or not 1<=cfg['seconds']<=300: raise ValueError('Bounded screen resources required')
    suite=json.loads(Path(suite_path).read_text(encoding='utf-8'));cases=suite['cases']
    if not 1<=len(cases)<=100 or len({c['id'] for c in cases})!=len(cases): raise ValueError('Invalid screen suite')
    if not models or models[0][0]!='original': raise ValueError('First arm must be original')
    if len(set(k for k,_ in models))!=len(models): raise ValueError('Duplicate model labels')
    for label,_ in models:
        if not label.isidentifier(): raise ValueError('Unsafe arm label')
    frozen={k:{'path':str(Path(p).resolve(strict=True)),'sha256':sha256(p)} for k,p in models}
    out.mkdir(parents=True)
    save(out/'protocol.json', {'version':'0.3.0','created_utc':datetime.now(timezone.utc).isoformat(),
        'models':frozen,'suite_sha256':sha256(suite_path),'server_sha256':sha256(server),'config':cfg,
        'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ['screen.py','stability.py']},
        'purpose':suite['purpose'],'weight_tuning':False,'no_context_shift':True,
        'loop_detection':'3 successive >=65% repeated 12-word windows, checked each 800 chars; no final content',
        'no_manual_reasoning_end':True,'score_reasoning':False,'loop_abort_is_not_success':True})
    save(out/'suite.json',suite);data={}
    try:
        for label,item in frozen.items():
            if (out/'CANCEL').exists():break
            data[label]=run_arm(label,Path(item['path']),item['sha256'],server,out,cases,cfg)
        summary={'arms':{k:summarize(v) for k,v in data.items()},'decisions':{},'complete':len(data)==len(models) and all(len(x)==len(cases) for x in data.values())}
        if 'original' in data:
            summary['decisions']={k:assess(data['original'],v,suite['purpose']) for k,v in data.items() if k!='original'}
        save(out/'summary.json',summary)
        print('SUMMARY',json.dumps(summary),flush=True)
        return summary
    except BaseException as exc:
        save(out/'FAILED.json',{'error':repr(exc),'partial_data_retained':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',action='append',required=True,help='label=path; first original')
    p.add_argument('--server',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,default=3072);p.add_argument('--seconds',type=int,default=120)
    p.add_argument('--gpu-layers',type=int,default=99);p.add_argument('--workers',type=int,choices=[1,2],default=2)
    a=p.parse_args();run([x.split('=',1) for x in a.model],a.server,a.suite,a.out,
        {'max_tokens':a.max_tokens,'seconds':a.seconds,'gpu_layers':a.gpu_layers,'workers':a.workers,'context':8192,'think':True})
