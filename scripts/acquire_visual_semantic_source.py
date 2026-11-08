#!/usr/bin/env python3
"""Acquire the complete audited visual-semantic release, without loading it.

Use the project volume for new large files and reuse matching shared snapshots.
No tensor deserialization, package installation, forward pass or GPU use occurs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from ilm.image_edit.baseline import ROOT,save_json,sha

REPO="deepgenteam/DeepGen-1.0-diffusers"
REVISION="85c2ed257b9406d8531965c68e140dc7fdb81382"
CODE_SHA="41e0f263d9ad55940426a4b2092688a74711395499c12a71ee69dc378ec9d3a9"
EVIDENCE=ROOT/"references/evidence/visual_conditioning_package_headers.json"
CACHE=ROOT/"artifacts/external/visual_semantic_source/hub"


def check_file(path,row):
    if not path.is_file() or path.stat().st_size!=row["size"]:return False
    if row.get("lfs"):return sha(path)==row["lfs"]["sha256"]
    body=path.read_bytes()
    return hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()==row["blobId"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    paths=("scripts/acquire_visual_semantic_source.py",str(EVIDENCE.relative_to(ROOT)))
    if subprocess.check_output(["git","status","--porcelain","--",*paths],cwd=ROOT,text=True):
        raise ValueError("commit exact acquisition source and audited manifest first")
    implementation={name:sha(ROOT/name) for name in paths}
    lock=(ROOT/"artifacts/visual_semantic_acquisition.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();verified=[]
    status=lambda phase,**extra:save_json(output/"status.json",dict(phase=phase,
        pid=None if phase in ("failed","complete_verified") else os.getpid(),
        verified_files=len(verified),implementation_sha256=implementation,weights_loaded=False,
        gpu_used=False,**extra))
    status("starting")
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",timeout=30) as response:metadata=json.load(response)
        if metadata["sha"]!=REVISION or metadata.get("gated"):raise ValueError("pinned public model source required")
        audited=json.loads(EVIDENCE.read_text());expected={r["file"]:r for r in audited["files"]}
        rows=[r for r in metadata["siblings"] if r["rfilename"]!=".gitattributes"]
        weights={r["rfilename"]:r for r in rows if r["rfilename"].endswith(".safetensors")}
        if weights.keys()!=expected.keys():raise ValueError("complete published weight allowlist differs")
        for name,row in weights.items():
            if row["size"]!=expected[name]["file_bytes"] or row["lfs"]["sha256"]!=expected[name]["expected_full_file_sha256"]:
                raise ValueError("published weight metadata differs from audited source")
        save_json(output/"hub_metadata.json",metadata)
        from huggingface_hub import constants,hf_hub_download
        shared=Path(constants.HF_HUB_CACHE)/("models--"+REPO.replace("/","--"))/"snapshots"/REVISION
        snapshot=CACHE/("models--"+REPO.replace("/","--"))/"snapshots"/REVISION
        snapshot.mkdir(parents=True,exist_ok=True)
        # Existing verified large files are shared by reference, never copied.
        reused=[]
        for row in rows:
            name=row["rfilename"];source=shared/name;target=snapshot/name
            if source.is_file() and not target.exists() and check_file(source,row):
                target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source.resolve());reused.append(name)
        missing=sum(r["size"] for r in rows if not check_file(snapshot/r["rfilename"],r))
        if shutil.disk_usage(CACHE).free<100*1024**3+missing:
            raise RuntimeError("acquisition must leave 100 GiB free on the project volume")
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",HF_HUB_DISABLE_XET="1",HF_XET_HIGH_PERFORMANCE="0")
        for row in rows:
            name=row["rfilename"];target=snapshot/name;existed=check_file(target,row)
            status("verifying_cache" if existed else "downloading",current_file=name,missing_payload_bytes_at_start=missing)
            if not existed:
                target=Path(hf_hub_download(REPO,name,revision=REVISION,cache_dir=CACHE,token=False))
            if target.resolve()!= (snapshot/name).resolve() or not check_file(target,row):
                raise ValueError("complete acquired source checksum differs: "+name)
            if name=="deepgen_pipeline.py" and sha(target)!=CODE_SHA:
                raise ValueError("audited executable source differs")
            verified.append(dict(file=name,bytes=row["size"],sha256=sha(target),
                reused_before_download=existed,shared_cache_link=name in reused))
            save_json(output/"verified_files.json",verified)
            print(f"Verified {len(verified)}/{len(rows)}: {name}",flush=True)
        if any(sha(ROOT/p)!=h for p,h in implementation.items()):raise ValueError("acquisition implementation changed")
        result=dict(repo=REPO,revision=REVISION,model_snapshot=str(snapshot),cache=str(CACHE),
            files=verified,complete_weight_files=len(weights),published_vlm_included=True,
            base_qwen_weights_substituted=False,whole_weight_file_checksums_verified=True,
            reused_shared_cache_files=reused,implementation_sha256=implementation,
            total_source_bytes=sum(r["bytes"] for r in verified),elapsed_seconds=time.perf_counter()-started,
            free_project_bytes_after=shutil.disk_usage(CACHE).free,
            package_installation_performed=False,weights_loaded=False,gpu_used=False,
            numerical_equivalence_established=False,language_quality_established=False,general_ilm_goal_complete=False)
        save_json(output/"report.json",result);status("complete_verified")
        print(json.dumps({k:result[k] for k in ("complete_weight_files","total_source_bytes","elapsed_seconds","whole_weight_file_checksums_verified","weights_loaded")},indent=2))
    except BaseException as error:
        status("failed",reason=str(error));raise


if __name__=="__main__":main()
