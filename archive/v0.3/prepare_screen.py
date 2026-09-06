"""Freeze a 24-case engineering regression suite; not a general benchmark."""
import argparse, json
from pathlib import Path
from guarded import save, sha256
OLD_IDS = ['ARC-Challenge:Mercury_LBS10933', 'ARC-Easy:Mercury_7056315',
           'ARC-Challenge:Mercury_400750', 'ARC-Easy:Mercury_7264040',
           'ARC-Easy:Mercury_406546', 'ARC-Easy:Mercury_7110968',
           'ARC-Easy:TIMSS_2007_8_pg130', 'ARC-Easy:Mercury_SC_415335']

def build(previous, out):
    if out.exists(): raise FileExistsError('Do not overwrite a frozen suite')
    old={x['id']:x for x in json.loads(previous.read_text(encoding='utf-8'))['cases']}
    cases=[{**old[k], 'category':'historical-loop-regression'} for k in OLD_IDS]
    for i,(a,b,c) in enumerate([(13,17,5),(29,8,7),(16,23,9),(27,12,11)]):
        cases.append({'id':f'v03-arith-{i}', 'category':'arithmetic', 'check':'integer',
            'prompt':f'Compute ({a} * {b}) - {c}. Write FINAL: followed by the integer.', 'expected':str(a*b-c)})
    codes=[('print(sum(i*i for i in range(5)))',30),('print(len(set([2,2,4,7,7,9])))',4),
           ('a=[1,3]; b=a; b.append(5); print(sum(a))',9),('print(sum(range(2,10,2)))',20)]
    for i,(code,expected) in enumerate(codes):
        cases.append({'id':f'v03-code-{i}','category':'code','check':'integer','expected':str(expected),
            'prompt':f'What integer does this Python code print? {code}\nWrite FINAL: followed by the integer.'})
    qs=[('All A are B. All B are C. Must all A be C?',['Yes','No'],'A'),
        ('Some A are B. Does that prove all A are B?',['Yes','No'],'B'),
        ('Nora arrives before Sam. Lee arrives after Sam. Who arrives last?',['Nora','Sam','Lee'],'C'),
        ('Every red token is round. A token is round. Must it be red?',['Yes','No'],'B'),
        ('Which structure carries most of the genetic material in a typical human cell?',['Cell membrane','Nucleus','Cell wall','Vacuole'],'B'),
        ('What change makes liquid water become water vapor?',['Freezing','Melting','Evaporation','Solidification'],'C'),
        ('Which is an electrical insulator under ordinary dry conditions?',['Copper','Aluminum','Rubber','Silver'],'C'),
        ('Which quantity is the sum of protons and neutrons in an atom?',['Mass number','Atomic number','Charge','Electron count'],'A')]
    for i,(q,options,ans) in enumerate(qs):
        cases.append({'id':f'v03-mcq-{i}','category':'logic' if i<4 else 'science','check':'mcq','expected':ans,
            'prompt':q+'\n'+'\n'.join(f'{chr(65+j)}. {s}' for j,s in enumerate(options))+'\nWrite FINAL: followed by one letter.'})
    assert len(cases)==24
    save(out,{'purpose':'regression_screen_not_independent_capability_benchmark', 'old_suite_sha256':sha256(previous),
        'historical_items':8, 'fresh_handmade_items':16, 'case_selection_uses_previous_failures':True,
        'not_training_data':True,'cases':cases})
    print('FROZEN_SUITE',sha256(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.previous,a.out)
