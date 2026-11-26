"""Archive the original stopped answer images, per-case readings and source coordinates."""

from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from ilm.visual_lm.portable_writer import read_image
from scripts.evaluate_role_grounding import PUB, ROOT, RUN, WRITERS, read, sha


def main():
    panel = read(RUN/"unseen-books.json")
    evals = read(PUB/"evaluation.json")
    gen = read(RUN/"generated.json")
    if sha(RUN/"generated.json") != evals["files"]["generated.json"]:
        raise ValueError("Original generation differs")
    dest = ROOT/"publication/ilm-nature/evidence/role_translation_outputs.zip"
    temp = dest.with_suffix(".tmp")
    manifest = []
    with ZipFile(temp,"w") as archive:
        def store(name, data):
            info = ZipInfo(name,(2026,11,26,0,0,0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info,data)
        for i in range(384):
            r = gen["rows"][i]
            src = RUN/"pixels"/f"{i:04d}"
            pair = panel[r["id"]]
            entry = {k:pair[k] for k in ("doc","chunk","paragraph","unit")}
            entry.update(request=i, target_language=("en","zh")[r["t"]], source_png=sha(src/"input.png"), answers={})
            for m in WRITERS:
                out = gen["answers"][i][m]
                path = src/(m+".png")
                pixels = read_image(path,len(out["cells"])).tobytes()
                raw = path.read_bytes()
                if sha256(pixels).hexdigest()!=out["packed"] or sha256(raw).hexdigest()!=out["png"]:
                    raise ValueError("Changed original answer")
                name = f"{i:04d}/{m}.png"
                store(name, raw)
                entry["answers"][m] = dict(path=name,png=out["png"],cells=len(out["cells"]),stop=out["stop"],
                                            exact=out["exact"],source_only=evals["source_only"]["records"][i][m],
                                            reference_informed=evals["reference_informed"]["records"][i][m])
            manifest.append(entry)
        store("manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2).encode())
    with ZipFile(temp) as z:
        if z.testzip() is not None or len(z.namelist()) != 1537:
            raise ValueError("Incomplete archive")
    temp.replace(dest)
    print(sha(dest),dest.stat().st_size)


if __name__=="__main__":
    main()
