"""Check reading/writing evidence, direct inference and physical representation collisions."""
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
import torch
from scripts.prepare_visual_referents import PUB, WORK, ROOT, read, dump, sha
from scripts.evaluate_visual_referents import ARCHIVE, event_panel
from scripts.read_visual_referents import load_strip, render_sentence
from scripts.write_visual_referent_event import EventWriter
from scripts.prepare_natural_pictured_roles import RUN as OLD


def main():
    torch.set_num_threads(2)
    result = read(PUB/"outcomes.json")
    assert sha(ARCHIVE)==result["archive"]
    cases = list(event_panel())
    common = read(WORK/"codes.json")
    palette = np.load(WORK/"palette.npy")
    lookup = {c:sha256(palette[i].tobytes()).digest() for c,i in common.items()}
    def cache_hash(text):
        return sha256(b"".join(lookup[c] for c in text)).hexdigest()
    old_hash = {cache_hash(r["text"]) for r in read(OLD/"train.json") if all(c in lookup for c in r["text"])}
    groups,seen = {},set()
    for split in ("external","test","dev","train"):
        keys = {cache_hash(r["text"]) for r in read(WORK/f"{split}.json")}
        groups[split] = dict(unique=len(keys),previous=len(keys & seen),in_parent_train=len(keys & old_hash))
        assert not keys & seen
        if split!="train":
            assert not keys & old_hash
        seen.update(keys)
    with ZipFile(ARCHIVE) as z:
        writing = json.loads(z.read("writing.json"))
        rows = json.loads(z.read("reading.json"))
        for group,scores in result["reading"].items():
            if group.count("/")!=1:
                continue
            these = [r for r in rows if r["group"]==group]
            assert len(these)==scores["n"]
            assert all(sum(r["spans"][arm]==r["gold"] for r in these)==scores[arm] for arm in ("local","target","parent"))
        for i,r in enumerate(writing):
            for mode,answers in r["out"].items():
                blob = z.read(f"{i:04d}/{mode}.png")
                assert sha256(blob).hexdigest()==answers["png"]
        writer = EventWriter()
        replays = {}
        for i in (1,2,3,len(writing)-8,len(writing)-1):
            r = writing[i]
            src = load_strip(z.read(f"{i:04d}/source.png"),len(r["source"]))
            ink,stop,spans = writer.write(src,r["verb"])
            rec = r["out"]["target"]
            assert len(ink)//128 == rec["cells"] and stop==rec["stop"]
            orig = load_strip(z.read(f"{i:04d}/target.png"),rec["cells"])
            assert ink==np.packbits(orig>0,axis=None).tobytes()
            assert list(map(list,spans))==r["spans"]["target"]
            replays[str(i)] = sha256(ink).hexdigest()
        paired = {}
        for f in ("plain","subject","object","reported"):
            subset = [r for r in writing if r["form"]==f]
            index = {r["source_id"].lower():r for r in subset}
            paired[f] = {m:sum(r["out"][m]["exact"] and index[r["reverse"]]["out"][m]["exact"]
                               for r in subset)//2 for m in ("full","local","target","annotated")}
        # Equal-verb contrasts: identical subject, lexical verb and object at the annotated
        # locations. Explicit do-support keeps the verb pixels identical across polarities.
        base_forms = dict(helped="help",visited="visit",thanked="thank",called="call",invited="invite",
                          followed="follow",saw="see",found="find",hugged="hug",praised="praise")
        n = equal = learned = 0
        sep = render_sentence(" ")[0].tobytes()
        def frame(ink,spans):
            return sep.join(ink[slice(*span)].tobytes() for span in spans)
        for r in cases:
            if r["form"]!="plain":
                continue
            a,verb,b = [r["text"][slice(*span)] for span in (r["target"][0],r["verb"],r["target"][1])]
            outputs = []
            for aux in ("did","did not"):
                predicate = base_forms[verb]
                text = f"{a} {aux} {predicate} {b}."
                ink = render_sentence(text)
                start = len(a)+len(aux)+2
                v = (start,start+len(predicate))
                assert text[slice(*v)]==predicate
                gold = ((0,len(a)),v,(len(text)-len(b)-1,len(text)-1))
                first,last = writer.reader.locate(ink,v)
                outputs.append((frame(ink,gold),frame(ink,(first,v,last))))
            equal += outputs[0][0]==outputs[1][0]
            learned += outputs[0][1]==outputs[1][1]
            n += 1
        unique = len({r["source_id"] for r in cases})
        assert n==equal==unique
        external = writing[-8:]
        assert all(r["form"]=="external" for r in external)
        natural_gold = sum(r["spans"]["target"]==r["gold"] for r in external)
    dump(PUB/"audit.json",dict(splits=groups,replays=replays,pairs_both_exact=paired,pairs_per_form=118,
        checked_output_hashes=4*len(writing),positive_negative_pairs=n,annotated_frame_collisions=equal,
        learned_frame_collisions=learned,external_correct_roles=natural_gold,archive=sha(ARCHIVE),
        proof=sha(ROOT/"formal/span_list/EventBottleneck.lean"),program=sha(Path(__file__))))
    print(n,equal,learned,paired,natural_gold)


if __name__=="__main__":
    main()
