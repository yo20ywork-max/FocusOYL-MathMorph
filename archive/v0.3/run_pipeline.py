"""Convert -> paired screen -> record decision. Never replaces or promotes the input."""
from pathlib import Path
import argparse
from guarded import Policy, convert, save
from screen import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--server',required=True,type=Path);p.add_argument('--suite',required=True,type=Path)
    p.add_argument('--expected-source-sha256');p.add_argument('--quantizer',type=Path)
    p.add_argument('--max-tokens',type=int,default=3072);p.add_argument('--seconds',type=int,default=120)
    p.add_argument('--gpu-layers',type=int,default=99)
    a=p.parse_args();root=a.out.resolve()
    if root.exists():raise FileExistsError('NEW pipeline directory required')
    source=a.source.resolve(strict=True);server=a.server.resolve(strict=True);suite=a.suite.resolve(strict=True)
    if a.max_tokens<1 or a.max_tokens>=8192-1024 or not 1<=a.seconds<=300:raise ValueError('Invalid screen resource budget')
    root.mkdir(parents=True)
    report=convert(source,root/'conversion',Policy(),a.expected_source_sha256,a.quantizer)
    results=run([('original',source),('candidate',Path(report['candidate']['path']))],server,suite,root/'screen',
        {'max_tokens':a.max_tokens,'seconds':a.seconds,'workers':2,'gpu_layers':a.gpu_layers,'context':8192,'think':True})
    save(root/'DECISION.json',{'candidate_sha256':report['candidate']['sha256'],
        'source_sha256':report['source_sha256'],'decision':results['decisions'].get('candidate',{'release':'RETAIN_ORIGINAL','screen':'INCOMPLETE'}),
        'model_replaced':False,'automatic_capability_claim':False})

if __name__=='__main__':main()
