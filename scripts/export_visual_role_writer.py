"""Export the measured participant writer, shape memory and image-only inference program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]
FIT=ROOT/"artifacts/visual_role_writer_v1/matched"


def export(output):
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fit=json.loads((FIT/"fit.json").read_text())
    reg=json.loads((ROOT/"references/evidence/visual_role_writer_20261119/registration.json").read_text())
    memory="artifacts/grounded_glyph_writer_v1/memory.npy"
    hashes={"final.safetensors":fit["sha256"],"memory.npy":reg[memory]}
    expected=json.loads((ROOT/"references/evidence/visual_role_writer_20261119/result.json").read_text())
    if fit!=expected["fits"]["matched"]:raise ValueError("Unmeasured checkpoint")
    example=ROOT/"references/evidence/visual_role_writer_20261119"
    with TemporaryDirectory(dir=output.parent) as tmp:
        path=Path(tmp)/"writer.zip"
        with ZipFile(path,"x",compression=ZIP_DEFLATED) as z:
            for name,src in (("final.safetensors",FIT/"final.safetensors"),("memory.npy",ROOT/memory)):
                if hashlib.sha256(src.read_bytes()).hexdigest()!=hashes[name]:raise ValueError(f"Changed {src}")
                z.write(src,name)
            for name in ("composed_answer_writer","grounded_glyph_decoder","scoped_question","portable_writer"):
                src=Path("ilm/visual_lm")/(name+".py")
                z.write(ROOT/src,str(src))
            for name in ("ilm/__init__.py","ilm/visual_lm/__init__.py"):
                z.writestr(name,"")
            z.writestr("read.py","from pathlib import Path\nfrom ilm.visual_lm.portable_writer import main\nmain(Path(__file__).resolve().parent)\n")
            z.writestr("writer.json",json.dumps(dict(sha256=hashes,limit=64,parameters=fit["parameters"]),indent=2)+"\n")
            z.writestr("requirements.txt","torch\nnumpy\nPillow\nsafetensors\nfonttools\n")
            for src in sorted(example.glob("bronze*.png")):z.write(src,"examples/"+src.name)
            z.write(ROOT/"docs/visual-role-writer.md","README.md")
        os.link(path,output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    export(p.parse_args().output)
