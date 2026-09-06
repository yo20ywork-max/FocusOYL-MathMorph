from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description="Training-free GGUF mathematical reconstruction. Capability gains are experimental.")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("inspect"); p.add_argument("model",type=Path)
    p=sub.add_parser("convert")
    p.add_argument("model",type=Path); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--layers",help="Comma-separated block indices; default: one block at 2/3 depth")
    p.add_argument("--rank",type=int,default=64); p.add_argument("--strength",type=float,default=0.05)
    p.add_argument("--max-change",type=float,default=0.02); p.add_argument("--quadrature",type=int,default=32)
    p.add_argument("--seed",type=int,default=20260905); p.add_argument("--threads",type=int,default=4)
    p.add_argument("--quantizer",type=Path); p.add_argument("--mode",choices=["gpep","unweighted","roundtrip"],default="gpep")
    p.add_argument("--no-control",action="store_true"); p.add_argument("--max-workspace-gib",type=float,default=4.0)
    p=sub.add_parser("evaluate")
    p.add_argument("run",type=Path); p.add_argument("--server",type=Path,required=True)
    p.add_argument("--suite",type=Path,required=True); p.add_argument("--threads",type=int,default=6)
    p.add_argument("--max-tokens",type=int,default=96); p.add_argument("--context",type=int,default=2048)
    args=parser.parse_args()
    threads=getattr(args,"threads",4)
    if not 1<=threads<=256:
        parser.error("threads must be in [1,256]")
    # Set before importing NumPy. Conversion does not reserve a GPU.
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[key]=str(threads)
    try:
        if args.command=="inspect":
            from .pipeline import inspect_model
            result=inspect_model(args.model)
        elif args.command=="convert":
            from .pipeline import build
            from .mathcore import Config
            cfg=Config(rank=args.rank,strength=args.strength,max_change=args.max_change,quadrature=args.quadrature,seed=args.seed)
            layers=list(map(int,args.layers.split(','))) if args.layers else None
            result=build(args.model,args.out,cfg,layers,args.quantizer,args.mode,threads,not args.no_control,args.max_workspace_gib)
        else:
            from .evaluate import evaluate_run
            result=evaluate_run(args.run,args.server,args.suite,args.threads,args.max_tokens,args.context)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,ArithmeticError,RuntimeError,MemoryError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
