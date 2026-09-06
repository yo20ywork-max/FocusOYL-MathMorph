"""Frozen research converter for the exact tested MiniCPM5 F16 checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
from convert_geometry import sha256
from experiment import SOURCE_SHA
from projection_sweep import RECIPES,convert_one

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recipe',choices=[r['label'] for r in RECIPES],default='euclid_last512_protected')
    p.add_argument('--allow-strong-research-edit',action='store_true')
    a=p.parse_args();source=a.source.resolve(strict=True);out=a.out.resolve()
    recipe=next(r for r in RECIPES if r['label']==a.recipe)
    if out.exists():p.error('Output directory must not already exist')
    if recipe.get('alpha')==1 and not a.allow_strong_research_edit:
        p.error('Full projections require --allow-strong-research-edit')
    if sha256(source)!=SOURCE_SHA:p.error('This recipe requires the tested baseline SHA-256')
    result=convert_one(source,out,recipe,{})
    print('Research candidate:',result)
    print('NOT AUTOMATICALLY PROMOTED. Read the independent evaluation first.')

if __name__=='__main__':main()
