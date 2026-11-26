"""Score complete true and crossed translations with an offline teacher and optical student."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic

from huggingface_hub import snapshot_download
import numpy as np
import torch
from torch.nn import functional as F

from scripts.evaluate_visual_prefix import cohort
from scripts.prepare_teacher_translation import dump
from scripts.sample_visual_library import ROOT, RECEIPT, library, read, sha
from scripts.train_paired_ink import conditional
from scripts.train_visual_prefix import gpu_owner, material, pack
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore

RUN = ROOT / "artifacts/library_teacher_v1"
PUB = ROOT / "references/evidence/library_teacher_v1"
TEACHERS = (("zh-en", "cf109095479db38d6df799875e34039d4938aaa6"),
            ("en-zh", "408d9bc410a388e1d9aef112a2daba955b945255"))


def proposals():
    groups = defaultdict(list)
    for i,r in enumerate(library("train")):
        groups[r["work"]].append((i,r))
    rng = np.random.default_rng(603121)
    out = []
    for title, members in sorted(groups.items()):
        order = sorted(members,key=lambda a:sha256(("teacher-bind/1/" + repr(a[1]["text"])).encode()).digest())
        remaining = list(range(len(order)))
        grams = [set(zip(r["text"][0].lower(),r["text"][0].lower()[1:])) for _,r in order]
        for _ in range(min(64,len(order)//2)):
            i = remaining.pop(0)
            sample = rng.choice(remaining, min(len(remaining),128), replace=False)
            j = max(sample, key=lambda k:len(grams[i]&grams[k])/max(1,len(grams[i]|grams[k])))
            remaining.remove(j)
            a,b = order[i],order[j]
            if any(a[1]["cells"][t]==b[1]["cells"][t] for t in (0,1)):
                raise ValueError("Repeated natural endpoint")
            out.append(dict(kind="natural",doc=title,id=[a[0],b[0]],
                            text=[a[1]["text"],b[1]["text"]],cells=[a[1]["cells"],b[1]["cells"]]))
    roles = cohort()[144:1088]
    for i in range(0,len(roles),8):
        for kind, flip in (("polarity",2),("exchange",4)):
            for j in range(8):
                k = j ^ flip
                if j >= k:
                    continue
                pair = [roles[i+j],roles[i+k]]
                out.append(dict(kind=kind,doc="controlled",id=[i//8,j,k],
                                text=[[r["source"],r["ref"]] for r in pair],
                                cells=[[r["input"],r["target"]] for r in pair]))
    return out


def candidates(rows, t, view):
    return [(r[view][s][1-t], r[view][y][t]) for r in rows for s,y in ((0,0),(1,1),(0,1),(1,0))]


@torch.inference_mode()
def teacher_scores(examples, t):
    from transformers import MarianMTModel, MarianTokenizer
    name, revision = TEACHERS[t]
    folder = Path(snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=revision,local_files_only=True))
    tokenizer = MarianTokenizer.from_pretrained(folder,local_files_only=True)
    net = MarianMTModel.from_pretrained(folder,local_files_only=True).eval().to("cuda")
    # Score each distinct conditional sentence only once.
    unique = list(dict.fromkeys(examples))
    values = {}
    for j in range(0,len(unique),24):
        data = unique[j:j+24]
        x = tokenizer([a for a,b in data],return_tensors="pt",padding=True).to("cuda")
        labels = tokenizer(text_target=[b for a,b in data],return_tensors="pt",padding=True).input_ids.to("cuda")
        labels = labels.masked_fill(labels==net.config.pad_token_id,-100)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits = net(**x,labels=labels,use_cache=False).logits
        loss = F.cross_entropy(logits.float().transpose(1,2),labels,reduction="none",ignore_index=-100)
        values.update(zip(data,(-loss.double().sum(1)).cpu().tolist(),strict=True))
    info = dict(model="Helsinki-NLP/opus-mt-"+name, revision=revision,
                parameters=sum(p.numel() for p in net.parameters()), unique_sequences=len(unique),
                files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    return np.array([values[e] for e in examples]).reshape(-1,4), info


@torch.inference_mode()
def optical_scores(examples, t, model, optical):
    features, inv, maps = optical
    values = []
    for j in range(0,len(examples),48):
        batch = examples[j:j+48]
        x = pack([a for a,b in batch],"cuda")
        y = pack([b+[1] for a,b in batch],"cuda")
        values.extend(conditional(model,features,x,y,inv[t],maps[t],t).cpu().tolist())
    return np.asarray(values).reshape(-1,4)


def summarize(rows, teacher, student):
    groups = {}
    for group in ("natural","polarity","exchange"):
        indices = [i for i,r in enumerate(rows) if r["kind"]==group]
        def margin(arr):
            return arr[indices,:2].sum(1)-arr[indices,2:].sum(1)
        a,b = margin(teacher),margin(student)
        doc = defaultdict(list)
        for k,i in enumerate(indices):
            doc[str(rows[i]["id"][0]) if group!="natural" else rows[i]["doc"]].append(k)
        groups[group] = dict(n=len(indices),teacher_correct=int(sum(a>1e-5)),student_correct=int(sum(b>1e-5)),
                             teacher_ties=int(sum(abs(a)<=1e-5)),student_ties=int(sum(abs(b)<=1e-5)),
                             teacher_only=int(sum((a>1e-5)&(b<=1e-5))),student_only=int(sum((b>1e-5)&(a<=1e-5))),
                             all_comparisons_correct={label:sum(bool((v[ix]>1e-5).all()) for ix in doc.values())
                                                     for label,v in (("teacher",a),("student",b))},
                             groups=len(doc),median_teacher_margin=float(np.median(a)),median_student_margin=float(np.median(b)))
    return groups


def main():
    configure_cuda_cli()
    rows = proposals()
    RUN.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    selection = RUN/"pairs.json"
    dump(selection,rows)
    scores, metadata = {}, {}
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 8*2**30:
            raise RuntimeError("Insufficient free GPU memory")
        net, optical = restore("library-broad","cuda"), material("cuda")
        for t,lang in enumerate(("en","zh")):
            torch.cuda.reset_peak_memory_stats()
            start = monotonic()
            a,teacher = teacher_scores(candidates(rows,t,"text"),t)
            b = optical_scores(candidates(rows,t,"cells"),t,net,optical)
            scores[lang+"_teacher"],scores[lang+"_writer"] = a,b
            metadata[lang]=dict(teacher=teacher,result=summarize(rows,a,b),seconds=monotonic()-start,
                                peak_bytes=torch.cuda.max_memory_allocated())
            print(lang,metadata[lang]["result"],flush=True)
    file = RUN/"scores.npz"
    np.savez_compressed(file,**scores)
    evidence=dict(program=sha(Path(__file__)),protocol=sha(ROOT/"references/bilingual_teacher_assignment_protocol.md"),
                  library=sha(RECEIPT),writer=sha(ROOT/"models/library_diversity/broad/weights.safetensors"),
                  pairs_sha256=sha(selection),scores_sha256=sha(file),directions=metadata,
                  qualified=all(r["teacher_correct"]>=.9*r["n"] for d in metadata.values() for r in d["result"].values()))
    dump(PUB/"assignments.json",evidence)


if __name__=="__main__":
    main()
