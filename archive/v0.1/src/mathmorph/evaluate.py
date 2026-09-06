"""Equal-budget local inference checks. Evaluation data never enters conversion."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request
from .ggufio import sha256


def _json_equal(actual, expected) -> bool:
    # JSON numbers compare numerically, but booleans are not numbers.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_json_equal(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_json_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def grade(text: str, case: dict) -> bool:
    mode=case.get("check","exact")
    if mode=="exact":
        return text.strip()==str(case["expected"]).strip()
    if mode=="one_of":
        return text.strip() in case["expected"]
    if mode=="json":
        try:
            return _json_equal(json.loads(text),case["expected"])
        except (ValueError,TypeError):
            return False
    raise ValueError(f"Unknown grading mode: {mode}")


def summarize_pair(base: list[bool],candidate: list[bool],alpha: float=0.05) -> dict:
    if not base or len(base)!=len(candidate):
        raise ValueError("Nonempty paired observations required")
    n=len(base); differences=[int(c)-int(b) for b,c in zip(base,candidate)]
    delta=sum(differences)/n
    # Hoeffding for independent observations in [-1,1]. These assumptions
    # need representative independent tasks; handcrafted smoke tests do not meet
    # a claim of population-wide generalization. No degenerate bootstrap interval.
    radius=math.sqrt(2*math.log(2/alpha)/n)
    return {"n":n,"baseline_pass":sum(base),"candidate_pass":sum(candidate),
        "baseline_rate":sum(base)/n,"candidate_rate":sum(candidate)/n,"delta":delta,
        "wins":sum(x==1 for x in differences),"losses":sum(x==-1 for x in differences),
        "ties":sum(x==0 for x in differences),
        "conservative_interval":[max(-1,delta-radius),min(1,delta+radius)],
        "interval_method":"Hoeffding; assumes independent representative tasks",
        "alpha":alpha}


def _post(base: str, route: str, payload: dict, timeout=120):
    req=urllib.request.Request(base+route,data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
                               headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.load(r)


def run_model(server: Path, model: Path, cases: list[dict], out: Path,
              threads=6, max_tokens=96, context=2048, seed=20260905) -> list[dict]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
    base=f"http://127.0.0.1:{port}"
    cmd=[str(server.resolve(strict=True)),"-m",str(model.resolve(strict=True)),
         "--host","127.0.0.1","--port",str(port),"-ngl","0","-t",str(threads),
         "-c",str(context),"--parallel","1","--jinja"]
    results=[]
    with (out/"server.log").open("wb") as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            end=time.monotonic()+150
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("llama-server exited before readiness; see server.log")
                try:
                    with urllib.request.urlopen(base+"/health",timeout=2) as r:
                        if r.status==200:
                            break
                except (OSError,urllib.error.URLError):
                    pass
                if time.monotonic()>end:
                    raise TimeoutError("llama-server readiness timeout")
                time.sleep(0.4)
            for i,case in enumerate(cases):
                started=time.monotonic()
                payload={"model":"local","messages":[{"role":"user","content":case["prompt"]}],
                    "temperature":0,"seed":seed,"max_tokens":max_tokens,"stream":False,
                    "chat_template_kwargs":{"enable_thinking":False},"cache_prompt":False}
                data=_post(base,"/v1/chat/completions",payload)
                choice=data["choices"][0]; message=choice["message"]
                text=message.get("content") or ""
                row={"id":case["id"],"category":case["category"],"prompt":case["prompt"],
                     "expected":case["expected"],"check":case.get("check","exact"),"output":text,
                     "pass":grade(text,case),"finish_reason":choice.get("finish_reason"),
                     "usage":data.get("usage"),"seconds":time.monotonic()-started,
                     "reasoning_nonempty":bool(message.get("reasoning_content") or message.get("reasoning"))}
                results.append(row)
                with (out/"responses.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(f"EVAL {model.name} {i+1}/{len(cases)} {case['id']} pass={row['pass']}",flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=10)
    return results


def evaluate_run(run_dir: Path, server: Path, suite_path: Path, threads=6,max_tokens=96,context=2048) -> dict:
    run_dir=run_dir.resolve(strict=True)
    manifest=json.loads((run_dir/"report.json").read_text(encoding="utf-8"))
    if manifest.get("structural_status")!="PASS":
        raise ValueError("Structural verification must pass first")
    suite=json.loads(suite_path.read_text(encoding="utf-8")); cases=suite["cases"]
    if not cases or len({c["id"] for c in cases})!=len(cases):
        raise ValueError("Suite must have nonempty unique case IDs")
    for case in cases:
        grade("",case)  # Validate graders before starting a model.
    out=run_dir/"evaluation"
    out.mkdir(exist_ok=False)
    models={"original":Path(manifest["input"]["path"]),"candidate":run_dir/"candidate.gguf"}
    if (run_dir/"roundtrip-control.gguf").exists():
        models["roundtrip"]=run_dir/"roundtrip-control.gguf"
    expected={"original":manifest["source_sha256"],"candidate":manifest["candidate.gguf"]["sha256"]}
    if "roundtrip" in models:
        expected["roundtrip"]=manifest["roundtrip-control.gguf"]["sha256"]
    actual={k:sha256(p) for k,p in models.items()}
    if actual!=expected:
        raise ValueError("Model changed since conversion")
    frozen={"frozen_utc":datetime.now(timezone.utc).isoformat(),"suite_sha256":sha256(suite_path),
        "suite_name":suite.get("name"),"purpose":suite.get("purpose","unspecified"),
        "model_hashes":actual,"server_sha256":sha256(server),
        "inference":{"temperature":0,"seed":20260905,"max_tokens":max_tokens,"context":context,
                     "threads":threads,"gpu_layers":0,"enable_thinking":False,"cache_prompt":False}}
    (out/"frozen_protocol.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding="utf-8")
    observations={}; completed_hashes={}
    try:
        for label,model in models.items():
            # A byte-identical roundtrip cannot change model weights; record reuse explicitly.
            alias=next((k for k,h in completed_hashes.items() if h==actual[label]),None)
            if alias:
                observations[label]=observations[alias]
                continue
            folder=out/label; folder.mkdir()
            observations[label]=run_model(server,model,cases,folder,threads,max_tokens,context)
            completed_hashes[label]=actual[label]
        comparisons={}
        for ref in ("original","roundtrip"):
            if ref not in observations:
                continue
            base=observations[ref]; cand=observations["candidate"]
            cats=sorted({x["category"] for x in base})
            overall=summarize_pair([x["pass"] for x in base],[x["pass"] for x in cand],0.05/(len(cats)+1))
            by={cat:summarize_pair([x["pass"] for x in base if x["category"]==cat],
                                  [x["pass"] for x in cand if x["category"]==cat],0.05/(len(cats)+1)) for cat in cats}
            comparisons[ref]={"overall":overall,"categories":by}
        # An explicit scope statement is mandatory. A bundled smoke suite never
        # authorizes a broad capability or release claim.
        summary={**frozen,"comparisons":comparisons,"status":"NO_VERIFIED_GENERAL_GAIN",
                 "scope":"Observed results on the provided suite only; not a broad benchmark or proof of generalization",
                 "source_unchanged":sha256(models["original"])==expected["original"],
                 "byte_identical_control_reused":actual.get("roundtrip")==actual["original"],
                 "response_count":len(cases),"test_data_used_for_conversion":False}
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        return summary
    except BaseException as exc:
        (out/"FAILED.json").write_text(json.dumps({"error":str(exc),"status":"INVALID_INCOMPLETE_EVALUATION"},indent=2),encoding="utf-8")
        raise
