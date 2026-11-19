"""Package the selected decoder and verify every inherited reader weight location."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from shutil import copyfile

from scripts.cache_ordered_page_features import PUBLIC,WORK,ROOT,READER,sha
from scripts.compare_ordered_requests import OUT


def export():
    result=json.loads((PUBLIC/"variation.json").read_text())["arms"]["varied"]
    fit=json.loads((PUBLIC/"fit.json").read_text())
    dest=ROOT/"models/ordered_page_writer"
    model={"writer.safetensors":(OUT/"varied.safetensors",result["weights"]),
           "writing_memory.npy":(WORK/"writing_memory.npy",fit["memory_sha256"])}
    for n,(src,h) in model.items():
        assert sha(src)==h
        copyfile(src,dest/n)
    reader=json.loads((READER/"fit.json").read_text())["weights"]
    assert all(sha(READER/n)==h for n,h in reader.items())
    manifest={"reader":reader,"writer":{n:h for n,(_,h) in model.items()},
              "decoder_training_receipt":str(PUBLIC.relative_to(ROOT)/"variation.json")}
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    bundle=WORK/"ordered-writing-addon.zip"
    with ZipFile(bundle,"x",compression=ZIP_DEFLATED,compresslevel=4) as archive:
        for f in dest.iterdir():
            if f.is_file():
                archive.write(f,"ordered_writer/"+f.name)
        archive.write(ROOT/"scripts/respond_ordered_page.py","respond_ordered.py")
        for name in ("ordered_page_features","ordered_page_writer"):
            rel="ilm/visual_lm/"+name+".py"
            archive.write(ROOT/rel,rel)
        for name in ("0_言_2","0_言_3"):
            label="forward" if name.endswith("2") else "reverse"
            for orig in ("source","question","target"):
                archive.write(WORK/"evaluation"/name/(orig+".png"),f"ordered_examples/{label}_{orig}.png")
    print(bundle,sha(bundle))


if __name__=="__main__":
    export()
