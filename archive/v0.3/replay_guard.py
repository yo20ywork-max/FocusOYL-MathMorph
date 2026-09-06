"""Replay preserved text prefixes, without starting a model or changing gold labels."""
from pathlib import Path
import json, statistics
from guarded import save, sha256
from stability import LoopMonitor
root=Path(__file__).parent
source=Path(r'C:\Users\YOUR_USER\Documents\FocusOYL-UnlimitedThink-20260905-B\pause-audit-20260905-184617\snapshot\sharpen_think_on\responses.jsonl')
rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
assert len(rows)==159
results=[]
for row in rows:
    monitor=LoopMonitor();reason=row['reasoning'];first=None
    for end in range(800,len(reason)+1,800):
        if monitor.update(reason[:end]):first=end;break
    results.append({'id':row['id'],'old_status':row['status'],'reasoning_chars':len(reason),'guard_triggered':first is not None,'first_trigger_chars':first})
s={}
for status in ['NATURAL_STOP','OBSERVATION_CENSORED']:
    v=[r for r in results if r['old_status']==status]
    hits=[r['first_trigger_chars'] for r in v if r['guard_triggered']]
    s[status]={'n':len(v),'guard_triggered':len(hits),'median_first_trigger_chars':statistics.median(hits) if hits else None}
data={'source_sha256':sha256(source),'detector_sha256':sha256(root/'stability.py'),'summary':s,'rows':results,'is_offline_replay':True,'not_measured_time_or_token_savings':True,'same_diagnosis_data_not_independent_validation':True}
save(root/'guard-replay.json',data)
print(json.dumps(s))
