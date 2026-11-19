"""Transfer a causal Chinese language model to pictured prompts and answer glyphs."""
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, BertTokenizer

from ilm.visual_lm.causal_glyph_writer import CausalGlyphWriter
from scripts.fit_font_invariant_reader import Face, FONTS, ROOT, WORK as METRIC, sha
from scripts.ordered_image_corpus import OPEN
from scripts.audit_pictured_entry_answers import sequences

WORK = ROOT / "artifacts/causal_glyph_writer_v1"
PUBLIC = ROOT / "references/evidence/causal_glyph_writer_20261119"
REV = "c2c0249d8a2731f269414cc3b22dff021f8e07a3"
DIGEST = "5478b950f53650c82f7f7e91010787c870813a76adeabe23c8571b01af1379b4"
SEED = 93652
BATCH = 16


def record(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def in_cells(stream):
    return [stream[k:k+128] for k in range(0, len(stream), 128)]


def examples(split):
    folder = OPEN / split
    receipt = json.loads((OPEN / "preparation.json").read_text())[split]
    for name, digest in receipt["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Changed {split}/{name}")
    bank = np.load(folder / "pixels.npy")
    seq = [sequences(folder, key, bank) for key in ("passages", "questions", "answers")]
    return list(zip(*seq, strict=True))


@torch.no_grad()
def prepare():
    dest = WORK / "base"
    dest.mkdir(parents=True)
    folder = Path(snapshot_download("uer/gpt2-chinese-cluecorpussmall", revision=REV, local_files_only=True))
    if sha(folder / "pytorch_model.bin") != DIGEST:
        raise ValueError("Unexpected upstream weights")
    teacher = GPT2LMHeadModel.from_pretrained(folder, local_files_only=True).eval()
    vocab = BertTokenizer.from_pretrained(folder, local_files_only=True).get_vocab()
    table = teacher.transformer.wte.weight.detach().cuda()
    if not torch.equal(teacher.lm_head.weight, teacher.transformer.wte.weight):
        raise ValueError("Expected tied embeddings")
    markers = table[[vocab["[CLS]"], vocab["[SEP]"]]]
    render = Face((FONTS[0], 2))
    native_ids, names = {}, {}
    for c, idx in sorted(vocab.items(), key=lambda item: item[1]):
        if len(c) == 1 and ord(c) in render.coverage and not c.isspace():
            bits = render.cell(c)
            if any(bits) and bits not in native_ids:
                native_ids[bits] = idx
                names[bits] = c
    book = list(native_ids)
    native_count = len(book)
    training_cells = np.load(OPEN / "train/pixels.npy")
    book.extend(bytes(b) for b in training_cells if bytes(b) not in native_ids)
    if len(set(book)) != len(book):
        raise ValueError("Duplicate physical proposals")
    packed = torch.from_numpy(np.frombuffer(b"".join(book), np.uint8).reshape(-1,128).copy()).cuda()
    projection = load_file(str(METRIC / "vision.safetensors"))["projection"].cuda()
    width = projection.shape[1]
    config = teacher.config
    config.to_json_file(dest / "config.json")
    writer = CausalGlyphWriter(config, projection, torch.empty(0), markers, packed).cuda().eval()
    del teacher.transformer.wte
    writer.body.load_state_dict(teacher.transformer.state_dict())
    writer.metric.load_state_dict(load_file(str(METRIC / "metric.safetensors")))
    features = []
    for batch in packed[:native_count].split(256):
        bits = ((batch[..., None] >> torch.arange(7,-1,-1,device="cuda")) & 1).reshape(-1,32,32)
        features.append(F.relu(writer.metric(bits) @ projection) * (2 / width)**.5)
    h = torch.cat(features)
    gold = table[list(native_ids.values())]
    gram = h @ h.T
    gram.diagonal().add_(0.0001)
    writer.regression = (h.T @ torch.cholesky_solve(gold, torch.linalg.cholesky(gram))).contiguous()
    fitted = writer.encode(packed[:1])
    assert fitted.shape == (1, gold.shape[1])
    fixed = writer.alphabet()
    cosine = F.cosine_similarity(fixed[1:1+native_count], gold,dim=1)
    centered = F.cosine_similarity(fixed[1:1+native_count]-gold.mean(0), gold-gold.mean(0), dim=1)
    native_out = fixed.clone()
    native_out[1:1+native_count] = gold
    torch.save(native_out.cpu(), WORK / "restricted_native.pt")
    # Human-readable legend is for audits only. No inference code opens it.
    for c in " \n\t０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        names.setdefault(render.cell(c), c)
    record(WORK / "audit_legend.json", [names.get(b, "�") for b in book])
    save_file({k:getattr(writer,k).detach().cpu().contiguous() for k in ("projection","regression","packed","markers")}, str(dest/"interface.safetensors"))
    shutil.copy2(METRIC/"metric.safetensors", dest/"metric.safetensors")
    save_file({k:v.detach().cpu().contiguous() for k,v in writer.body.state_dict().items()},str(dest/"body.safetensors"))
    return dict(upstream_revision=REV,upstream_sha256=DIGEST,native_proposals=native_count,total_proposals=len(book),
                native_mean_cosine=float(cosine.mean()), centered_cosine=float(centered.mean()),
                body_parameters=sum(p.numel() for p in writer.body.parameters()),
                regression=writer.regression.numel(),metric_parameters=sum(p.numel() for p in writer.metric.parameters()),
                fixed_projection=projection.numel(),markers=markers.numel())


def dataset(writer, records, native=False):
    """Cache only identical physical inputs. Unwritable references remain in evaluation."""
    bank = [bytes(v) for v in writer.packed.cpu().numpy()]
    subset = set(bank)
    extra = sorted({b for p,q,y in records for s in (p,q,y) for b in in_cells(s) if b not in subset})
    read_ids = {b:i+1 for i,b in enumerate(bank+extra)}
    with torch.no_grad():
        output = torch.load(WORK/"restricted_native.pt",weights_only=True).cuda() if native else writer.alphabet()
        read = [output]
        if extra:
            v = torch.from_numpy(np.frombuffer(b"".join(extra),np.uint8).reshape(-1,128).copy()).cuda()
            read.extend(writer.encode(t) for t in v.split(256))
        read.append(writer.markers[:1])
        read = torch.cat(read)
    start = len(read)-1
    enc = lambda s:[read_ids[b] for b in in_cells(s)]
    prompts = [[start,*enc(p),0,*enc(q),0] for p,q,_ in records]
    answers = [enc(y)+[0] for _,_,y in records]
    return read, output, prompts, answers


def derange(prompts):
    order = sorted(range(len(prompts)), key=lambda i:len(prompts[i]))
    # Local cyclic blocks keep prefill lengths close while replacing the entire pair.
    index = list(range(len(order)))
    groups = np.array_split(order,max(1,len(order)//32))
    for group in groups:
        for i,j in zip(group,np.roll(group,1)):
            index[int(i)] = int(j)
    assert all(prompts[a] != prompts[b] for a,b in enumerate(index))
    return index


def loss_inputs(ids, prompts, answers, source_ids, read):
    x, positions, target = [], [], []
    for i in ids:
        prompt = prompts[source_ids[i]]
        answer = answers[i]
        x.append(prompt + answer[:-1])
        positions.extend((len(x)-1, j) for j in range(len(prompt)-1,len(x[-1])))
        target.extend(answer)
    pad = max(map(len,x))
    device = read.device
    rows = torch.tensor([s+[0]*(pad-len(s)) for s in x],device=device)
    pos = torch.tensor(positions,device=device).T
    return read[rows], pos, torch.tensor(target,device=device)


def arm(name, records):
    start_time = monotonic()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    writer = CausalGlyphWriter.restore(WORK/"base","cuda")
    writer.requires_grad_(False)
    writer.body.h[-4:].requires_grad_(True)
    writer.body.ln_f.requires_grad_(True)
    trainables = [p for p in writer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainables, lr=.00008, fused=True)
    read,output,prompts,answers = dataset(writer,records)
    indices = derange(prompts) if name == "shuffled" else list(range(len(records)))
    trace = []
    count = 0
    for epoch in range(2):
        loss_sum = 0.
        n = 0
        for ids in np.array_split(rng.permutation(len(records)), int(np.ceil(len(records)/BATCH))):
            x,pos,targets = loss_inputs(ids,prompts,answers,indices,read)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                states = writer.body(inputs_embeds=x,use_cache=False).last_hidden_state
                logits = states[pos[0],pos[1]] @ output.T
                loss = F.cross_entropy(logits.float(), targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
            opt.step()
            loss_sum += float(loss.detach())*len(targets)
            n += len(targets)
            count += 1
            if count % 50 == 0:
                print(name,count,round(loss_sum/n,3),round(monotonic()-start_time,1),flush=True)
            if monotonic()-start_time > 900 or torch.cuda.max_memory_allocated()>19*1024**3:
                raise RuntimeError("Fit exceeded registered ceiling")
        trace.append(loss_sum/n)
    dest = WORK/name
    dest.mkdir()
    for f in ("config.json","metric.safetensors","interface.safetensors"):
        os.link(WORK/"base"/f,dest/f)
    save_file({k:v.detach().cpu().contiguous() for k,v in writer.body.state_dict().items()},str(dest/"body.safetensors"))
    return dict(records=len(records),epochs=2,updates=count,trained_parameters=sum(p.numel() for p in trainables),
                answer_positions=n*2,epoch_nll=trace,seconds=monotonic()-start_time,peak_bytes=torch.cuda.max_memory_allocated(),
                files={p.name:sha(p) for p in dest.iterdir()})


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("stage",choices=["transfer","matched","shuffled","all"])
    stage = arg.parse_args().stage
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Use assigned GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        if stage in ("transfer","all"):
            t = monotonic()
            result = prepare()
            record(PUBLIC/"transfer.json", result | dict(seconds=monotonic()-t))
            print(result,flush=True)
        for name in ("matched","shuffled"):
            if stage in (name,"all"):
                if (WORK/name).exists():
                    raise FileExistsError(WORK/name)
                torch.cuda.reset_peak_memory_stats()
                result = arm(name,examples("train"))
                record(PUBLIC/(name+"_fit.json"),result)
                print(name,result,flush=True)
