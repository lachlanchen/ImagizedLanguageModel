"""Generate two physical training drafts and measure reverse source reconstruction."""
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel, MarianTokenizer

from ilm.visual_lm.pictured_path_writer import COUNTS
from ilm.visual_lm.sentence_reconstruction import START, restore
from scripts.reserve_emitted_history import BASE, ROOT, RUN, PUB, dump, read, sha
from scripts.reserve_target_glyph_book_panel import library
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner, material, pack

PROTOCOL = ROOT / "references/sentence_reconstruction_protocol.md"
DRAFTS = RUN / "drafts.json"
RECEIPT = PUB / "drafts.json"


@torch.inference_mode()
def draw(net, features, bank, ids, source, direction, feedback, keys):
    """Batched deployment decoder. Table rows are optical readings of actual motor ink."""
    context, valid = net.writer.encode(bank, source)
    field = net.readers[direction](features, source)[0][:,1:source.shape[1]+1]
    memory = net.add_field(context, field)
    count = len(source)
    x = net.writer.begin[direction].expand(count,1,-1)
    finished = torch.zeros(count, device=x.device, dtype=torch.bool)
    lengths = torch.zeros_like(finished, dtype=torch.long)
    steps, past = [], None
    for pos in range(97):
        h, past = net.writer.decode(x, memory, valid, past, pos)
        scores = h[:,0] @ keys.T
        mode = scores[:count//2].argmax(-1)
        top, choices = (scores[count//2:] / (h.size(-1)**0.5 * .85)).topk(4, dim=-1)
        draw_ix = torch.multinomial(top.softmax(-1), 1)
        choice = torch.cat((mode, choices.gather(1,draw_ix).flatten()))
        finished |= choice.eq(0)
        if pos == 96 or finished.all():
            break
        lengths += ~finished
        steps.append(ids[choice])
        x = feedback[choice][:,None]
    grid = torch.stack(steps, 1).cpu().tolist() if steps else [[] for _ in range(count)]
    return [(line[:n],stop) for line,n,stop in zip(grid,lengths.tolist(),finished.tolist(),strict=True)]


@torch.inference_mode()
def score_reverse(lang, samples, originals):
    rec = read(COUNTS / (lang+"-fit.json"))
    folder = snapshot_download(rec["model"], revision=rec["revision"], local_files_only=True)
    tok = MarianTokenizer.from_pretrained(folder, local_files_only=True)
    teacher = MarianMTModel.from_pretrained(folder,local_files_only=True,attn_implementation="eager").to("cuda").eval()
    result = []
    for i in range(0,len(samples),24):
        inp = tok(samples[i:i+24], padding=True, return_tensors="pt").to("cuda")
        labels = tok(text_target=originals[i:i+24],padding=True, return_tensors="pt").input_ids.cuda()
        labels = labels.masked_fill(labels.eq(tok.pad_token_id),-100)
        logits = teacher(**inp, labels=labels, use_cache=False).logits
        from ilm.visual_lm.sentence_reconstruction import mean_log_score
        result.extend(mean_log_score(logits,labels).tolist())
    return result, rec["revision"]


@torch.inference_mode()
def collect():
    start = monotonic()
    torch.manual_seed(9128)
    rows = library("train")
    chosen = sorted(np.random.default_rng(61307).choice(len(rows),4096,replace=False).tolist())
    palette = np.load(BASE/"palette.npy")
    names = read(BASE/"readback.json")
    features, inv, _ = material("cuda")
    net = restore("parent","cuda")
    bank = net.writer.optical(features)
    drafts, stats = [], []
    for t in (0,1):
        codes = getattr(net,f"codes_{t}")
        pixels = torch.cat([net.motor(c).gt(0) for c in codes.split(256)])
        packed = np.packbits(pixels.cpu().numpy(),axis=1)
        if not np.array_equal(packed,palette[inv[t][1:].cpu().numpy()]):
            raise ValueError("A deployed motor output does not match its physical training tile")
        feedback = net.writer.optical(torch.nn.functional.avg_pool2d(pixels.float().reshape(-1,1,32,32),2).flatten(1))
        feedback = torch.nn.functional.pad(feedback,(0,0,1,0))
        keys = torch.cat((net.writer.end[None],net.writer.target(codes)))
        records = []
        for i in range(0,len(chosen),64):
            indices = chosen[i:i+64]
            x = pack([rows[j]["cells"][1-t] for j in indices],"cuda")
            outputs = draw(net,features,bank,inv[t],x.repeat(2,1),t,feedback,keys)
            for j,a,b in zip(indices,outputs[:len(x)],outputs[len(x):],strict=True):
                records.append(dict(i=j,a=a[0],b=b[0],stops=[a[1],b[1]]))
        samples, originals = [], []
        for r in records:
            pair = rows[r["i"]]["text"]
            samples.extend(["".join(names[str(k)] for k in r[v]) for v in ("a","b")]+[pair[t]])
            originals.extend([pair[1-t]]*3)
        reverse = ("zh","en")[t]
        values, revision = score_reverse(reverse,samples,originals)
        margins = []
        for r,s in zip(records,np.reshape(values,(-1,3)).tolist(),strict=True):
            r["scores"] = s
            valid = all(r["stops"]) and r["a"]!=r["b"]
            margins.append(s[0]-s[1] if valid else 0.)
        stats.append(dict(drafts=len(records)*2,eligible=sum(x!=0 for x in margins),
                          reference_above_both=sum(r["scores"][2]>max(r["scores"][:2]) for r in records),
                          revision=revision,reverse_target=reverse))
        drafts.append(records)
        print(t,stats[-1],flush=True)
    peak = torch.cuda.max_memory_allocated()
    if peak>12*2**30 or monotonic()-start>3600:
        raise RuntimeError("Collection resource ceiling")
    dump(DRAFTS,drafts)
    dump(RECEIPT,dict(sha256=sha(DRAFTS),program=sha(Path(__file__)),protocol=sha(PROTOCOL),
                      parent=read(START/"guided/fit.json")["weights"],panel=sha(PUB/"panel.json"),
                      stats=stats,seconds=monotonic()-start,peak=peak))


if __name__=="__main__":
    assert sha(RUN/"panel.json")==read(PUB/"panel.json")["sha256"]
    if RECEIPT.exists():
        assert sha(DRAFTS)==read(RECEIPT)["sha256"]
    else:
        configure_cuda_cli()
        with gpu_owner():
            torch.set_num_threads(2)
            collect()
