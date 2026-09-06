"""Immutable-input GGUF surgery with same-type quantization and byte audit."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import numpy as np
from .ggufio import GGUF, GGUFError, TYPES, encode_native, decode, sha256, assert_only_patches
from .mathcore import Config, transform

# Shape alone cannot establish FFN semantics. Start with explicit architecture adapters.
ARCHITECTURES = {"llama", "qwen2", "qwen3"}


def inspect_model(path: Path) -> dict:
    g=GGUF(path)
    arch=g.metadata.get("general.architecture")
    blocks=[]
    for name,t in g.tensors.items():
        m=re.fullmatch(r"blk\.(\d+)\.ffn_down\.weight",name)
        if m and len(t.shape)==2:
            i=int(m.group(1))
            required=[f"blk.{i}.ffn_{x}.weight" for x in ("down","up","gate")]
            if all(k in g.tensors for k in required):
                blocks.append(i)
    return {"path":str(g.path),"size_bytes":g.size,"architecture":arch,
            "model_name":g.metadata.get("general.name"),"gguf_version":g.version,
            "tensor_count":len(g.tensors),"dense_gated_blocks":sorted(blocks),
            "architecture_adapter_available":arch in ARCHITECTURES,
            "mutation_scope":"down projection only; gate/up/norm read-only",
            "quant_types":sorted({t.type_name for t in g.tensors.values()})}


def encode_same_type(g: GGUF, name: str, array: np.ndarray, quantizer: Path | None,
                     scratch: Path, tag: str, threads: int=4) -> bytes:
    t=g.tensors[name]
    if t.qtype in (0,1,30):
        return encode_native(array,t.qtype)
    if quantizer is None:
        raise GGUFError(f"{t.type_name} writes require --quantizer path/to/llama-quantize")
    qexe=Path(quantizer).resolve(strict=True)
    inp=scratch/f"{tag}.f32.gguf"; out=scratch/f"{tag}.encoded.gguf"
    g.write_single_f32(inp,name,array)
    # Enable quantization with Q8_0, then explicitly override the only tensor.
    # F32 disables quantization before overrides on llama.cpp b10672.
    cmd=[str(qexe),"--pure","--tensor-type",name+"="+t.type_name.lower(),str(inp),str(out),"Q8_0",str(threads)]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600,check=False)
    (scratch/f"{tag}.quantize.log").write_bytes(proc.stdout)
    if proc.returncode:
        raise GGUFError("llama-quantize failed: "+proc.stdout.decode("utf-8",errors="replace")[-3000:])
    result=GGUF(out)
    rt=result.tensors.get(name)
    if rt is None or rt.qtype!=t.qtype or rt.shape!=t.shape or rt.nbytes!=t.nbytes:
        raise GGUFError("Quantizer did not preserve the requested tensor type and shape")
    return result.raw(name)


def build(input_path: Path, out_dir: Path, config: Config, layers: list[int] | None=None,
          quantizer: Path | None=None, mode="gpep", threads=4,
          make_control=True, max_workspace_gib=4.0) -> dict:
    started=time.monotonic(); config.validate()
    source=Path(input_path).resolve(strict=True)
    root=Path(out_dir).resolve()
    if root.exists():
        raise FileExistsError("Output directory must not already exist; originals are never overwritten")
    if mode not in ("gpep","unweighted","roundtrip"):
        raise ValueError("Unknown mode")
    info=inspect_model(source)
    if not info["architecture_adapter_available"]:
        raise GGUFError(f"Unsupported architecture adapter: {info['architecture']}")
    blocks=info["dense_gated_blocks"]
    if not blocks:
        raise GGUFError("No supported dense gated FFN found")
    if layers is None:
        layers=[blocks[min(len(blocks)-1,(2*len(blocks))//3)]]
    if len(set(layers))!=len(layers) or not set(layers)<=set(blocks):
        raise ValueError("Requested layers unavailable or repeated")
    g=GGUF(source)
    for i in layers:
        biases=[f"blk.{i}.ffn_{kind}.bias" for kind in ("down","up","gate")]
        if any(name in g.tensors for name in biases):
            raise GGUFError("Biased FFNs are not supported by the Gaussian path model")
        for kind in ("down","up","gate"):
            t=g.tensors[f"blk.{i}.ffn_{kind}.weight"]
            # Conservative bound includes float64 moment and linear-algebra temporaries.
            estimate=np.prod(t.shape,dtype=np.int64)*48
            if estimate>max_workspace_gib*(1024**3):
                raise MemoryError("Estimated workspace exceeds configured limit")
    root.parent.mkdir(parents=True,exist_ok=True)
    needed=g.size*(3 if make_control else 2)+256*1024*1024
    if shutil.disk_usage(root.parent).free<needed:
        raise OSError("Insufficient free disk for safe output and rollback")
    source_stat=source.stat()
    before=sha256(source)
    root.mkdir(exist_ok=False)
    report={"schema":1,"tool":"FocusOYL MathMorph","version":"0.1.0",
        "started_utc":datetime.now(timezone.utc).isoformat(),"input":info,"source_sha256":before,
        "config":asdict(config),"layers":layers,"operator":mode,
        "training":False,"external_data":False,"teacher":False,"parameters_added":0,
        "capability_status":"UNTESTED","release_status":"CANDIDATE_UNVERIFIED",
        "numpy":np.__version__,"python":platform.python_version(),"platform":platform.platform(),
        "threads":threads,"patches":[],"quantizer":str(quantizer) if quantizer else None,
        "quantizer_sha256":sha256(quantizer) if quantizer else None}
    (root/"plan.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    patches={}; controls={}
    try:
        scratch=root/"scratch"; scratch.mkdir()
        for i in layers:
            name=f"blk.{i}.ffn_down.weight"
            print(f"MATH layer={i} tensor={name}",flush=True)
            down=g.matrix(name); up=g.matrix(f"blk.{i}.ffn_up.weight"); gate=g.matrix(f"blk.{i}.ffn_gate.weight")
            nk=f"blk.{i}.ffn_norm.weight"
            norm=g.matrix(nk) if nk in g.tensors else None
            cand,stats=transform(down,up,gate,norm,config,mode)
            original_raw=g.raw(name)
            control=encode_same_type(g,name,down,quantizer,scratch,f"layer{i}.control",threads)
            patch=encode_same_type(g,name,cand,quantizer,scratch,f"layer{i}.candidate",threads)
            td=g.tensors[name]
            if len(patch)!=td.nbytes or len(control)!=td.nbytes:
                raise GGUFError("Encoded byte length changed")
            decoded=decode(patch,td.qtype,td.shape)
            decoded_control=decode(control,td.qtype,td.shape)
            dn=max(float(np.linalg.norm(down.astype(np.float64))),1e-30)
            real=decoded-down; roundtrip=decoded_control-down; effect=decoded-decoded_control
            target=cand-down
            realized=float(np.linalg.norm(real.astype(np.float64))/dn)
            rt=float(np.linalg.norm(roundtrip.astype(np.float64))/dn)
            if not np.isfinite(decoded).all() or not np.isfinite(decoded_control).all():
                raise GGUFError("Encoded non-finite weights")
            # A large quantization perturbation is not accepted as a mathematical gain.
            if realized>config.max_change*1.1+1e-6 and mode!="roundtrip":
                raise GGUFError(f"Post-encoding perturbation {realized:.6f} exceeds strict safety bound")
            denom=max(float(np.linalg.norm(effect.astype(np.float64))*np.linalg.norm(target.astype(np.float64))),1e-30)
            stats.update({"name":name,"shape":list(td.shape),"type":td.type_name,
                "original_tensor_sha256":hashlib.sha256(original_raw).hexdigest(),
                "candidate_tensor_sha256":hashlib.sha256(patch).hexdigest(),
                "realized_relative_change":realized,"roundtrip_relative_change":rt,
                "effect_relative_to_control":float(np.linalg.norm(effect.astype(np.float64))/dn),
                "encoding_direction_cosine":float(np.sum(effect.astype(np.float64)*target)/denom),
                "changed_weight_fraction":float(np.count_nonzero(real)/real.size),
                "candidate_differs_from_control":patch!=control})
            report["patches"].append(stats)
            patches[name]=patch; controls[name]=control
            del down,up,gate,cand,decoded,decoded_control,real,roundtrip,effect,target
        if source.stat().st_size!=source_stat.st_size or sha256(source)!=before:
            raise GGUFError("Source changed during computation; aborting")
        outputs=[("candidate.gguf",patches)]
        if make_control:
            outputs.append(("roundtrip-control.gguf",controls))
        for filename,patchset in outputs:
            partial=root/(filename+".partial")
            shutil.copyfile(source,partial)
            with partial.open("r+b") as f:
                for name,data in patchset.items():
                    f.seek(g.tensors[name].offset); f.write(data)
                f.flush(); os.fsync(f.fileno())
            audit=assert_only_patches(source,partial,patchset)
            final=root/filename
            partial.rename(final)
            report[filename]={"path":str(final),"sha256":sha256(final),**audit}
        report["source_unchanged"]=sha256(source)==before
        if not report["source_unchanged"]:
            raise GGUFError("Source changed during output creation")
        if not any(p["candidate_differs_from_control"] for p in report["patches"]):
            report["release_status"]="NO_EFFECT_AFTER_ENCODING"
        report["elapsed_seconds"]=time.monotonic()-started
        report["structural_status"]="PASS"
        (root/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        # Scratch files are generated intermediates, never user originals.
        shutil.rmtree(scratch)
        return report
    except BaseException as exc:
        report.update({"structural_status":"FAILED","release_status":"REJECTED","error":str(exc),"elapsed_seconds":time.monotonic()-started})
        (root/"FAILED.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        # Leave diagnostics and partial files for inspection; never present a partial as final.
        for filename in ("candidate.gguf","roundtrip-control.gguf"):
            p=root/filename
            if p.exists():
                p.rename(root/(filename+".rejected"))
        raise
