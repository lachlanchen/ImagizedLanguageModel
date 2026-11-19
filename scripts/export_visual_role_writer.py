"""Export the measured participant writer, shape memory and image-only inference program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]


def export(output, inverse=False, interval=False):
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    inverse = inverse or interval
    study = "inverse_role_writer" if inverse else "visual_role_writer"
    fit_dir = ROOT/"artifacts"/(study+"_v1")/"matched"
    example = ROOT/"references/evidence"/(study+"_20261119")
    fit=json.loads((fit_dir/"fit.json").read_text())
    reg=json.loads((example/"registration.json").read_text())
    memory="artifacts/grounded_glyph_writer_v1/memory.npy"
    hashes={"final.safetensors":fit["sha256"],"memory.npy":reg[memory]}
    expected=json.loads((example/"result.json").read_text())
    if fit!=expected["fits"]["matched"]:raise ValueError("Unmeasured checkpoint")
    with TemporaryDirectory(dir=output.parent) as tmp:
        path=Path(tmp)/"writer.zip"
        with ZipFile(path,"x",compression=ZIP_DEFLATED) as z:
            for name,src in (("final.safetensors",fit_dir/"final.safetensors"),("memory.npy",ROOT/memory)):
                if hashlib.sha256(src.read_bytes()).hexdigest()!=hashes[name]:raise ValueError(f"Changed {src}")
                z.write(src,name)
            for name in ("composed_answer_writer","grounded_glyph_decoder","scoped_question","portable_writer"):
                src=Path("ilm/visual_lm")/(name+".py")
                z.write(ROOT/src,str(src))
            for name in ("ilm/__init__.py","ilm/visual_lm/__init__.py"):
                z.writestr(name,"")
            if inverse:
                from scripts.prepare_pictured_entry_answers import Rasterizer
                import numpy as np
                raster = Rasterizer()
                marks = np.frombuffer(b"".join(raster.tile(c) for c in "。！？!?"),np.uint8).reshape(-1,128)
                boundary_file = Path(tmp)/"sentence_ends.npy"
                np.save(boundary_file,marks)
                hashes[boundary_file.name] = hashlib.sha256(boundary_file.read_bytes()).hexdigest()
                z.write(boundary_file,boundary_file.name)
                z.write(ROOT/"ilm/visual_lm/pictured_sentence.py","ilm/visual_lm/pictured_sentence.py")
            parameters = fit["parameters"]
            guide = study.replace("_","-")
            if interval:
                guide = "shared-sentence-writer"
                evidence = ROOT/"references/evidence/shared_sentence_writer_20261119"
                run = ROOT/"artifacts/shared_sentence_writer_v1/calibrated"
                initial = json.loads((evidence/"result.json").read_text())["fits"]["matched"]
                measured = json.loads((evidence/"calibrated.json").read_text())["fit"]
                if measured["source"] != initial["weights"]:
                    raise ValueError("Changed source-run initialization")
                weights = run/"interval.safetensors"
                hashes[weights.name] = hashlib.sha256(weights.read_bytes()).hexdigest()
                if hashes[weights.name] != measured["weights"] or measured["base"] != hashes["final.safetensors"]:
                    raise ValueError("Unmeasured source-run model")
                parameters += initial["new_parameters"]
                z.write(weights,weights.name)
                z.write(ROOT/"ilm/visual_lm/interval_writer.py","ilm/visual_lm/interval_writer.py")
                for src in sorted(evidence.glob("pair*.png")):
                    z.write(src,"examples/"+src.name)
            selection = ", sentence_selection=True" if inverse else ""
            z.writestr("read.py",f"from pathlib import Path\nfrom ilm.visual_lm.portable_writer import main\nmain(Path(__file__).resolve().parent{selection})\n")
            z.writestr("writer.json",json.dumps(dict(sha256=hashes,limit=64,parameters=parameters),indent=2)+"\n")
            z.writestr("requirements.txt","torch\nnumpy\nPillow\nsafetensors\nfonttools\n")
            for src in sorted(example.glob("bronze*.png")):z.write(src,"examples/"+src.name)
            z.write(ROOT/"docs"/(guide+".md"),"README.md")
        os.link(path,output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inverse",action="store_true",help="Export the inverse participant checkpoint")
    p.add_argument("--interval",action="store_true",help="Include the learned complete source-run action")
    args=p.parse_args()
    export(args.output,inverse=args.inverse,interval=args.interval)
