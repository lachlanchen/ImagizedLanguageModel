"""Verify a portable release outside the repository using real PNG and typed input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

LAUNCH = """
import runpy, site, sys
from pathlib import Path
root = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(root))
sys.path.append(site.getusersitepackages())
def offline(event, args):
    if event in ('socket.connect','socket.sendto','socket.sendmsg','socket.getaddrinfo','socket.gethostbyname'):
        raise RuntimeError('Network access in standalone inference')
sys.addaudithook(offline)
runpy.run_path(str(root/'read.py'), run_name='__main__')
for name,module in sys.modules.items():
    if name=='ilm' or name.startswith('ilm.'):
        assert root in Path(module.__file__).resolve().parents
assert not any(k.startswith('scripts.') for k in sys.modules)
"""
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(archive):
    env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with TemporaryDirectory() as tmp:
        root=Path(tmp)/"bundle";root.mkdir()
        with ZipFile(archive) as z:
            for name in z.namelist():
                if not (root/name).resolve().is_relative_to(root):
                    raise ValueError("Invalid bundle path")
            z.extractall(root)
        verified={}
        hist=Path(__file__).resolve().parents[1]/"artifacts/visual_role_writer_v1/historical"
        for i,p in enumerate(("秦有铜镜，楚没有铜镜。","楚有铜镜，秦没有铜镜。")):
            for key,inputs,expected in [
                ("bronze",["--passage-image",root/f"examples/bronze{i}_premise.png","--question-image",root/"examples/bronze0_query.png"],hist/f"bronze{i}_answer.png"),
                ("typed",["--passage-text",p,"--question-text","请把秦的情况写成一句话，把姓名放在“的是”后面。"],hist/f"print{i}_answer.png"),
            ]:
                dest=Path(tmp)/(key+str(i))
                subprocess.run([sys.executable,"-I","-c",LAUNCH,root,*inputs,"--output",dest],cwd=tmp,env=env,
                               stdout=subprocess.DEVNULL,check=True,timeout=60)
                if digest(dest/"answer.png")!=digest(expected):raise ValueError("Output differs from research model")
                verified[dest.name]=dict(png=digest(expected),**json.loads((dest/"receipt.json").read_text()))
        return dict(bundle_sha256=digest(archive),bytes=archive.stat().st_size,standalone_outputs=verified,
                    network_blocked=True,training_imports=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("--receipt",type=Path)
    args=p.parse_args();s=json.dumps(verify(args.archive.resolve()),indent=2)
    if args.receipt:
        with args.receipt.open("x") as f:f.write(s+"\n")
    print(s)
