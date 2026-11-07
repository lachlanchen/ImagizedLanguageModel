#!/usr/bin/env python3
"""Acquire/verify pinned native source assets or create an isolated runtime."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCK = ROOT / "references/native_unified_assets.lock.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def acquire(out):
    import requests
    from huggingface_hub import hf_hub_download
    lock = json.loads(LOCK.read_text()); started = time.monotonic()
    response = requests.get(f'https://api.github.com/repos/{lock["source_repo"]}/git/trees/{lock["source_commit"]}?recursive=1',timeout=30)
    response.raise_for_status(); tree = response.json()
    if tree.get("truncated"):
        raise ValueError("incomplete source tree")
    chosen = [x for x in tree["tree"] if x["path"] == "LICENSE" or x["path"].startswith("show-o2/")
        and x["path"].endswith((".py", ".yaml", "README.md", "build_env.sh"))]

    def fetch(item):
        path = out/"repo"/item["path"];path.parent.mkdir(parents=True,exist_ok=True)
        raw = path.read_bytes() if path.exists() else None
        git_digest = lambda data: hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if raw is None or git_digest(raw) != item["sha"]:
            r = requests.get(f'https://raw.githubusercontent.com/{lock["source_repo"]}/{lock["source_commit"]}/{item["path"]}',timeout=30)
            r.raise_for_status();raw = r.content
        if git_digest(raw) != item["sha"]:
            raise ValueError("native source Git blob mismatch")
        path.write_bytes(raw)
        return dict(path=item["path"],git_blob_sha1=item["sha"],sha256=digest(path),bytes=len(raw))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        source = list(pool.map(fetch,chosen))
    save(out/"source_manifest.json",dict(repo=lock["source_repo"],commit=lock["source_commit"],files=source))
    records = []
    for spec in lock["files"]:
        print("Acquire",spec["repo"],spec["file"],flush=True)
        path = Path(hf_hub_download(spec["repo"],filename=spec["file"],revision=spec["revision"],cache_dir=out/"hub",token=False))
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("source asset SHA mismatch: "+spec["file"])
        records.append(dict(repo=spec["repo"],revision=spec["revision"],file=spec["file"],
            bytes=spec["bytes"],sha256=spec["sha256"],path=str(path.resolve()),
            source_lfs_sha256=None if spec["git_blob_sha1"] else spec["sha256"]))
        save(out/"acquisition_progress.json",dict(files=records,elapsed_seconds=time.monotonic()-started))
    save(out/"acquisition.json",dict(files=records,elapsed_seconds=time.monotonic()-started,complete=True))


def verify(out):
    lock = json.loads(LOCK.read_text()); receipt = json.loads((out/"acquisition.json").read_text())
    records = {(x["repo"],x["file"]):x for x in receipt["files"]}
    if not receipt["complete"] or len(records) != len(lock["files"]):
        raise ValueError("complete expected asset set required")
    for spec in lock["files"]:
        record = records[spec["repo"],spec["file"]]
        for key in ("revision","bytes","sha256"):
            if record[key] != spec[key]:raise ValueError("locked asset differs: "+spec["file"])
        path = Path(record["path"])
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("asset bytes changed: "+spec["file"])
    source = json.loads((out/"source_manifest.json").read_text())
    if source["repo"] != lock["source_repo"] or source["commit"] != lock["source_commit"]:
        raise ValueError("native code identity differs")
    for item in source["files"]:
        raw = (out/"repo"/item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        if blob != item["git_blob_sha1"] or digest(out/"repo"/item["path"]) != item["sha256"]:
            raise ValueError("source code changed")
    result = dict(assets=len(records),source_files=len(source["files"]),all_hashes_verified=True,
        lock_sha256=digest(LOCK),model_calls=0,optimizer_updates=0,language_quality_qualified=False)
    save(out/"verification.json",result);print(json.dumps(result,indent=2))


def runtime(out, shared_python):
    environment = out/"runtime"
    info = json.loads(subprocess.check_output([str(shared_python),"-c",
        "import sysconfig,json,sys; print(json.dumps(dict(site=sysconfig.get_path('purelib'),version=list(sys.version_info[:2]))))"],text=True))
    if not (environment/"bin/python").exists():
        subprocess.run([str(shared_python),"-m","venv",str(environment)],check=True)
    site = next((environment/"lib").glob("python*/site-packages"))
    (site/"ilm_shared_image_packages.pth").write_text(info["site"]+"\n")
    packages = ["transformers==4.47.0","diffusers==0.31.0","huggingface-hub==0.36.0",
        "tokenizers==0.21.0","einops==0.8.0","torchdiffeq==0.2.5","scipy==1.15.3"]
    with (out/"runtime_setup.log").open("a") as log:
        subprocess.run([str(environment/"bin/python"),"-m","pip","install","--no-deps",*packages],stdout=log,stderr=subprocess.STDOUT,check=True)
    probe = subprocess.check_output([str(environment/"bin/python"),"-c",
        "import torch,transformers,diffusers,einops,scipy; print(torch.__version__,transformers.__version__,diffusers.__version__,einops.__version__,scipy.__version__)"],text=True).strip()
    save(out/"runtime.json",dict(pinned_packages=packages,shared_site=info["site"],shared_python=str(shared_python),
        actual_versions=probe,weights_copied=False,shared_environment_modified=False))
    print(probe)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("acquire","verify","runtime"),required=True)
    p.add_argument("--output",type=Path,default=ROOT/"artifacts/external/native_unified_source")
    p.add_argument("--shared-python",type=Path,default=ROOT.parent/"LocalLLM/.local/image-generation/venv/bin/python")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # Preserve a venv interpreter path: resolving its symlink would silently
    # select the base interpreter and the wrong shared site-packages.
    if args.stage=="runtime":runtime(out,args.shared_python.expanduser().absolute())
    elif args.stage=="acquire":acquire(out);verify(out)
    else:verify(out)


if __name__=="__main__":main()
