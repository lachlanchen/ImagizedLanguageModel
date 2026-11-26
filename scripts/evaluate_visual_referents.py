"""Read two naturally annotated participants and use their ink to write a Chinese event."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from PIL import ImageFont
import torch

from scripts.prepare_visual_referents import FONT, MODEL, PUB, ROOT, WORK, bitmap, dump, read, sha
from scripts.read_visual_referents import ReferentReader
from scripts.locate_pictured_roles import PicturedLocator
from scripts.read_natural_roles import strip_png, load_strip
from scripts.train_field_to_glyph import gpu_owner, configure_cuda_cli
from scripts.write_ordered_translation import restore, stopped
from scripts.train_paired_ink import BASE, material
from scripts.train_ordered_translation import pack
from scripts.prepare_region_writer import Ink

ARCHIVE = ROOT/"publication/ilm-nature/evidence/visual_referents.zip"
BOOK = ROOT/"artifacts/translation_paraphrases_v1/voice-outputs.json"
GENUINE = [
    "有人欺骗我们。","天使们羡慕这份爱。","天使们给少女起名。","ML等于DE。",
    "教授接替他们。","这些行为违背信任。","你一直面对这些处分。","其他人必须表明立场。",
]


def event_panel():
    for item in read(BOOK):
        if item["voice"]!="active" or item["negative"]:
            continue
        a,v,b = re.fullmatch(r"(.+?) (helped|visited|thanked|called|invited|followed|saw|found|hugged|praised) (.+)\.",item["source"]).groups()
        def mid(phrase):
            return phrase[:1].lower()+phrase[1:] if phrase.startswith("The ") else phrase
        for form,source in (
            ("plain",f"{a} {v} {b}."),
            ("subject",f"It was {mid(a)} who {v} {mid(b)}."),
            ("object",f"It was {mid(b)} whom {mid(a)} {v}."),
            ("reported",f"The witness said it was {mid(a)} who {v} {mid(b)}."),
        ):
            def word(w):
                return re.search(rf"\b{w}\b",source).span()
            yield dict(text=source,verb=word(v),target=[word(a.split()[-1]),word(b.split()[-1])],reference=item["reference"],
                       form=form,source_id=item["source"],reverse=f"{b} {v} {a}.".casefold())


def picture(packed):
    ink = np.unpackbits(packed,axis=1).reshape(-1,32,32).astype(np.uint8)*255
    return strip_png(ink) if len(ink) else strip_png(np.zeros((1,32,32),np.uint8))


@torch.inference_mode()
def main():
    configure_cuda_cli()
    if (PUB/"outcomes.json").exists():
        assert sha(ARCHIVE)==read(PUB/"outcomes.json")["archive"]
        return
    torch.set_num_threads(2)
    models = {"parent":PicturedLocator(),"local":ReferentReader("local"),"target":ReferentReader()}
    palette = np.load(WORK/"palette.npy")
    codes = read(WORK/"codes.json")
    corpus = read(PUB/"corpus.json")
    font = ImageFont.truetype(str(FONT),24)
    def render(text):
        return np.stack([palette[codes[c]] if c in codes else bitmap(c,font) for c in text])
    natural = []
    writing = list(event_panel())
    for split in ("test","external"):
        assert sha(WORK/f"{split}.json")==corpus["splits"][split]
        for r in read(WORK/f"{split}.json"):
            natural.append(dict(**r,split=split))
    genuine = [r for r in natural if r["split"]=="external" and r["linked"] and r["text"].isascii() and len(r["text"])<=96]
    for r,ref in zip(genuine,GENUINE,strict=True):
        writing.append(dict(**r,form="external",reference=ref,source_id=r["sentence"]))
    stats = defaultdict(Counter)
    details, answers = [], []
    # A fixed print-face conversion maps individual source bitmaps into the writer font.
    writer_palette = np.load(BASE/"palette.npy")
    inventory = Ink(writer_palette)
    ascii_ink = {}
    for char in sorted({c for r in writing for c in r["text"]} | {" ","."}):
        key = render(char).tobytes()
        idx, = inventory.encode(char)
        if idx>=len(writer_palette) or (key in ascii_ink and idx!=ascii_ink[key]):
            raise ValueError(f"Unsupported or ambiguous interface glyph {char!r}")
        ascii_ink[key] = idx
    blank,period = inventory.encode(" .")
    modes = ("full","local","target","annotated")
    model_inputs = {mode:[] for mode in modes}
    cached = {}
    def predict(ink,verb):
        key = sha256(ink.tobytes()).digest(),tuple(verb)
        if key not in cached:
            cached[key] = {arm:reader.locate(ink,verb) for arm,reader in models.items()}
        return cached[key]
    temporary = ARCHIVE.with_suffix(".partial")
    with ZipFile(temporary,"w",ZIP_DEFLATED) as z:
        for i,r in enumerate(natural):
            ink = render(r["text"])
            preds = predict(ink,r["verb"])
            gold = tuple(map(tuple,r["target"]))
            group = r["split"]+("/linked" if r["linked"] else "/direct")
            hits = {k:p==gold for k,p in preds.items()}
            for name in (group, group+"/"+r["corpus"]+"/"+r["voice"]):
                stats[name].update(n=1,**hits,gain=hits["target"] and not hits["local"],
                                   loss=hits["local"] and not hits["target"])
            details.append(dict(id=r["sentence"],group=group,verb=r["verb"],gold=gold,spans=preds))
        for i,r in enumerate(writing):
            ink = render(r["text"])
            prefix = f"{i:04d}/"
            z.writestr(prefix+"source.png",strip_png(ink))
            preds = predict(ink,r["verb"])
            spans = dict(local=preds["local"],target=preds["target"],annotated=r["target"])
            inputs = [ascii_ink[g.tobytes()] for g in ink]
            model_inputs["full"].append(inputs)
            for mode,(actor,undergoer) in spans.items():
                src = inputs[slice(*actor)]+[blank]+inputs[slice(*r["verb"])]+[blank]+inputs[slice(*undergoer)]+[period]
                model_inputs[mode].append(src)
                z.writestr(prefix+mode+"-event.png",picture(writer_palette[src]))
            answers.append(dict(source=r["text"],source_id=r["source_id"],reverse=r.get("reverse"),form=r["form"],gold=r["target"],
                                verb=r["verb"],spans=spans,reference=r["reference"],out={}))
        names = read(BASE/"readback.json")
        with gpu_owner():
            writer = restore("book-rehearsal","cuda")
            features, inv, _ = material("cuda")
            for mode in modes:
                for start in range(0,len(writing),48):
                    result = writer.generate(features,pack(model_inputs[mode][start:start+48],"cuda"),inv[1],1).cpu()
                    for i,seq in enumerate(result,start):
                        ids, stop = stopped(seq)
                        text = "".join(names[str(g)] for g in ids)
                        ref = answers[i]["reference"]
                        # Canonical identity covers every pixel including the final punctuation.
                        ref_ids = inventory.encode(ref)
                        exact = stop and ids==ref_ids
                        blob = picture(writer_palette[ids])
                        z.writestr(f"{i:04d}/{mode}.png",blob)
                        answers[i]["out"][mode] = dict(text=text,stop=stop,exact=exact,cells=len(ids),
                                                        png=sha256(blob).hexdigest())
        reverse = {r["source_id"].casefold():r["reference"] for r in answers if r["form"]=="plain"}
        writing_stats = defaultdict(lambda:defaultdict(Counter))
        for r in answers:
            opposite = None
            if r["form"]!="external":
                opposite = reverse.get(r["reverse"])
            for mode,o in r["out"].items():
                writing_stats[r["form"]][mode].update(n=1,exact=o["exact"],stopped=o["stop"],
                        reversible=opposite is not None,reversed=opposite is not None and o["stop"] and o["text"]==opposite)
        z.writestr("reading.json",json.dumps(details))
        z.writestr("writing.json",json.dumps(answers,ensure_ascii=False))
    temporary.replace(ARCHIVE)
    dump(PUB/"outcomes.json",dict(archive=sha(ARCHIVE),reading=stats,writing=writing_stats,
        program=sha(Path(__file__)),corpus=sha(PUB/"corpus.json"),challenge_source=sha(BOOK),
        writer=sha(ROOT/"models/book_breadth/rehearsal/weights.safetensors"),natural_n=len(details),writing_n=len(answers),
        fits={arm:sha(MODEL/arm/"weights.safetensors") for arm in ("local","target")}))
    print({k:dict(v) for k,v in stats.items() if k.count("/")==1})
    print({f:{k:v["exact"] for k,v in m.items()} for f,m in writing_stats.items()})


if __name__=="__main__":
    main()
