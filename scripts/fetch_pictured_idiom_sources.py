"""Fetch the pinned public training and validation tables used for pictured cloze."""
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from scripts.prepare_pictured_idioms import BIND,RAW
from scripts.diagnose_visual_context_gradients import require,sha

BASE='https://huggingface.co/datasets/clue/clue/resolve/28178267a609dd08bdc703dd6c931dfc2c2f4431/chid/'

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    for name,expected in BIND.items():
        path=RAW/(name+'.parquet')
        if path.exists():
            require(sha(path)==expected,'Existing source differs');continue
        tmp=None
        try:
            with NamedTemporaryFile(dir=RAW,delete=False) as dest:
                tmp=Path(dest.name);h=hashlib.sha256()
                with urlopen(BASE+name+'-00000-of-00001.parquet',timeout=60) as response:
                    while block:=response.read(1<<20):
                        dest.write(block);h.update(block)
            require(h.hexdigest()==expected,'Downloaded source differs')
            try:os.link(tmp,path)
            except FileExistsError:require(sha(path)==expected,'Concurrent source differs')
        finally:
            if tmp is not None:tmp.unlink(missing_ok=True)
        print('Verified',name)

if __name__=='__main__':
    main()
