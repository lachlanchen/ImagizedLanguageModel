"""Build a standalone image responder, with its contextual weights, glyph writer and provenance."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED,ZipFile

from scripts.fit_font_invariant_reader import PUBLIC,ROOT,WORK,sha,write_json
from scripts.train_interval_writer import WORK as ORIGINAL, PUBLIC as ORIGINAL_PUBLIC

MODULES=("visual_page_responder","font_invariant_reader","continuous_page_reader","scoped_question",
         "interval_writer","grounded_glyph_decoder","composed_answer_writer","portable_writer")
BUNDLE=WORK/"visual-page-responder.zip"


def build():
    if BUNDLE.exists():raise FileExistsError(BUNDLE)
    previous=ORIGINAL/"shared-sentence-writer.zip"
    assert sha(previous)==json.loads((ORIGINAL_PUBLIC/"portable.json").read_text())["bundle_sha256"]
    receipt=json.loads((PUBLIC/"fit.json").read_text())
    with TemporaryDirectory(dir=WORK) as tmp:
        tmp=Path(tmp)
        files={}
        for n,digest in receipt["weights"].items():
            assert sha(WORK/n)==digest
            files["reader/"+n]=WORK/n
        with ZipFile(previous) as z:
            for n in ("final.safetensors","interval.safetensors","memory.npy"):
                files["writer/"+n]=Path(z.extract(n,tmp))
        weights={n:sha(p) for n,p in files.items()}
        manifests=dict(weights=weights,context_source="ckiplab/bert-base-chinese-qa",
                       context_revision="77b858271f8926e3768410e3f8694454bf4c9314",
                       original_context_weight_license="GPL-3.0",visual_fit=sha(WORK/"fit.json"))
        write_json(tmp/"manifest.json",manifests)
        files["manifest.json"]=tmp/"manifest.json"
        for n in MODULES:files["ilm/visual_lm/"+n+".py"]=ROOT/"ilm/visual_lm"/(n+".py")
        files["respond.py"]=ROOT/"scripts/respond_visual_page.py"
        doc=(ROOT/"docs/visual-page-responder.md").read_text()
        (tmp/"README.md").write_text(doc.replace("../references/continuous_interface_result.md",
            "https://github.com/lachlanchen/ImagizedLanguageModel/blob/main/references/continuous_interface_result.md"))
        files["README.md"]=tmp/"README.md"
        for p in (ROOT/"references/evidence/yan_page_responses_20261119").glob("*.png"):
            files["examples/yan_"+p.name]=p
        files["COPYING-CKIP"]=Path("/usr/share/common-licenses/GPL-3")
        for p in sorted((WORK/"responder").glob("*.png")):
            files["examples/"+p.name]=p
        for n in ("responder","result","fit"):
            files["evidence/"+n+".json"]=PUBLIC/(n+".json")
        staged=tmp/"bundle.zip"
        with ZipFile(staged,"w",ZIP_DEFLATED,compresslevel=1) as out:
            for n,p in files.items():out.write(p,n)
            out.writestr("ilm/__init__.py","")
            out.writestr("ilm/visual_lm/__init__.py","")
            out.writestr("requirements.txt","torch>=2.6\ntransformers>=4.50,<5\nsafetensors\nnumpy\nPillow\nfonttools\n")
        staged.rename(BUNDLE)
    return BUNDLE


if __name__=="__main__":
    print(build())
