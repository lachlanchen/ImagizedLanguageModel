"""Fit a continuous visual input map to a pinned contextual Chinese QA reader."""

import argparse
import fcntl
from functools import lru_cache
import os
from pathlib import Path
from time import monotonic

from fontTools.ttLib import TTFont
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
from transformers import BertForQuestionAnswering, BertTokenizerFast

from ilm.visual_lm.continuous_page_reader import visual_features
from scripts.prepare_pictured_entry_answers import sha, write_json

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/"artifacts/continuous_page_reader_v1"
PUBLIC=ROOT/"references/evidence/continuous_page_reader_20261119"
MODEL="ckiplab/bert-base-chinese-qa"
REV="77b858271f8926e3768410e3f8694454bf4c9314"
WEIGHTS="b97fee98374da90086cf6dc7e49ffa6c35057e5900bec2bbb7125ea81a807088"
FONTDIR=Path("/usr/share/fonts/opentype/noto")
FONTS=tuple(FONTDIR/f for f in ("NotoSansCJK-Regular.ttc","NotoSerifCJK-Regular.ttc","NotoSansCJK-Bold.ttc"))
WIDTH=24576
RIDGE=.0003


class Renderer:
    def __init__(self, path):
        self.font=ImageFont.truetype(str(path),26,index=2)
        with TTFont(path,fontNumber=2) as f:
            self.coverage=set(f.getBestCmap())

    @lru_cache(maxsize=32768)
    def cell(self,c):
        if not c.isspace() and ord(c) not in self.coverage:
            raise ValueError(f"No glyph for U+{ord(c):04X}")
        im=Image.new("L",(32,32),255)
        if not c.isspace():
            draw=ImageDraw.Draw(im)
            draw.text(((32-draw.textlength(c,font=self.font))/2,26),c,anchor="ls",font=self.font,fill=0)
        return np.packbits(np.asarray(im)<128).tobytes()

    def __call__(self,s):
        return b"".join(self.cell(c) for c in s)


def snapshot(fetch=False):
    path=Path(snapshot_download(MODEL,revision=REV,local_files_only=not fetch,
        allow_patterns=["config.json","vocab.txt","pytorch_model.bin","README.md"]))
    assert sha(path/"pytorch_model.bin")==WEIGHTS
    return path


def binding():
    files=[Path(__file__),ROOT/"scripts/evaluate_continuous_page_reader.py",ROOT/"scripts/pictured_participant_scenes.py",
           ROOT/"scripts/explicit_participant_cases.py",ROOT/"ilm/visual_lm/continuous_page_reader.py",
           ROOT/"references/continuous_page_reader_protocol.md", *FONTS]
    source=snapshot()
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha(p)
            for p in files+[source/"vocab.txt",source/"config.json",source/"pytorch_model.bin"]}


@torch.inference_mode()
def fit():
    WORK.mkdir()
    source=snapshot()
    tokenizer=BertTokenizerFast.from_pretrained(source,local_files_only=True)
    teacher=BertForQuestionAnswering.from_pretrained(source,local_files_only=True).eval()
    word_vectors=teacher.bert.embeddings.word_embeddings.weight.detach()
    renderers=[Renderer(p) for p in FONTS[:2]]
    cells={}
    for text,idx in sorted(tokenizer.get_vocab().items(), key=lambda row: row[1]):
        if len(text)!=1 or text.isspace() or tokenizer(text,add_special_tokens=False).input_ids != [idx]:
            continue
        for render in renderers:
            if ord(text) in render.coverage:
                raw=render.cell(text)
                if any(raw):
                    cells.setdefault(raw,set()).add(idx)
    targets=torch.stack([word_vectors[sorted(ids)].mean(0) for ids in cells.values()])
    assert len(cells)<WIDTH
    config=teacher.config.to_dict()
    config.update(vocab_size=0,pad_token_id=None)
    write_json(WORK/"config.json",config)
    context={k:v.contiguous() for k,v in teacher.state_dict().items() if k!="bert.embeddings.word_embeddings.weight"}
    save_file(context,str(WORK/"context.safetensors"))
    markers=word_vectors[[tokenizer.cls_token_id,tokenizer.sep_token_id]].clone()
    del teacher, word_vectors, context, tokenizer
    torch.cuda.reset_peak_memory_stats()
    start=monotonic()
    bits=np.unpackbits(np.frombuffer(b"".join(cells),np.uint8).reshape(-1,128),axis=1)
    x=visual_features(torch.from_numpy(bits).cuda())
    torch.manual_seed(17039)
    projection=torch.randn(x.shape[1],WIDTH,device=x.device)
    hidden=F.relu(x@projection)*(2/WIDTH)**.5
    target=targets.cuda()
    gram=hidden@hidden.T
    gram.diagonal().add_(RIDGE)
    chol=torch.linalg.cholesky(gram)
    regression=hidden.T @ torch.cholesky_solve(target,chol)
    fitted=hidden @ regression
    quality=F.cosine_similarity(target,fitted)
    assert torch.isfinite(quality).all()
    peak=torch.cuda.max_memory_allocated()
    elapsed=monotonic()-start
    if peak>18*1024**3 or elapsed>600:
        raise RuntimeError("Ridge fit exceeded the registered resources")
    save_file(dict(projection=projection.cpu(),regression=regression.cpu(),markers=markers.contiguous()),
              str(WORK/"vision.safetensors"))
    report=dict(model=MODEL,revision=REV,source_weights=WEIGHTS,unique_training_images=len(cells),
        teacher_indices=len(set().union(*cells.values())),ambiguous_images=sum(len(v)>1 for v in cells.values()),
        hidden_width=WIDTH,ridge=RIDGE,seconds=elapsed,peak_allocated_bytes=peak,
        cosine_mean=quality.mean().item(),cosine_p05=torch.quantile(quality,.05).item(),
        context_sha256=sha(WORK/"context.safetensors"),vision_sha256=sha(WORK/"vision.safetensors"),
        config_sha256=sha(WORK/"config.json"),trained_coefficients=regression.numel(),
        fixed_projection_coefficients=projection.numel(),registration=sha(PUBLIC/"registration.json"))
    write_json(WORK/"fit.json",report);write_json(PUBLIC/"fit.json",report)
    print(report,flush=True)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    parser.add_argument("--fetch",action="store_true")
    args=parser.parse_args()
    if args.fetch:
        snapshot(True)
    reg=PUBLIC/"registration.json"
    if args.register:
        PUBLIC.mkdir(exist_ok=True,parents=True)
        if reg.exists(): raise FileExistsError(reg)
        write_json(reg,binding());return
    assert binding()==json.loads(reg.read_text()) and os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        fit()
        from scripts.evaluate_continuous_page_reader import evaluate
        evaluate()


if __name__=="__main__":
    main()
