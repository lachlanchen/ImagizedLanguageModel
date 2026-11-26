"""Check native, replicated and mass-corrected OPUS-MT states on the same complete passages."""

from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from time import monotonic

from huggingface_hub import snapshot_download
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ilm.visual_lm.glyph_measure import positions, expand_states, weighted_cross_attention
from scripts.score_bilingual_assignments import ROOT, PUB, RUN, TEACHERS, library, read, sha, dump
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def exact_assignments(tok, texts, encoded):
    out = []
    for s,ids,mask in zip(texts,encoded.input_ids,encoded.attention_mask,strict=True):
        pieces = tok.spm_source.encode_as_immutable_proto(s).pieces
        teacher = tok.convert_tokens_to_ids([p.piece for p in pieces]) + [tok.eos_token_id]
        if ids[mask.bool()].tolist() != teacher:
            raise ValueError("Teacher pieces differ from the actual encoded source")
        out.append(positions([p.end for p in pieces],len(s)))
    return out


def until_stop(seq, eos):
    values = seq[1:].tolist()
    return values[:values.index(eos)+1] if eos in values else values


@torch.inference_mode()
def direction(t, pairs):
    name, version = TEACHERS[t]
    folder = snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=version,local_files_only=True)
    tok = MarianTokenizer.from_pretrained(folder,local_files_only=True)
    net = MarianMTModel.from_pretrained(folder,local_files_only=True,attn_implementation="eager").eval().to("cuda")
    # A batch of size one is unnecessary. Keep all conditions on identical examples and batch shapes.
    rows, stats = [], {a:Counter() for a in ("copies","corrected")}
    for j in range(0,len(pairs),12):
        batch = pairs[j:j+12]
        src = [r["text"][1-t] for r in batch]
        x = tok(src,padding=True,return_tensors="pt").to("cuda")
        labels = tok(text_target=[r["text"][t] for r in batch],padding=True,return_tensors="pt").input_ids.to("cuda")
        mask = labels.ne(net.config.pad_token_id)
        hidden = net.model.encoder(**x).last_hidden_state
        mapped = exact_assignments(tok,src,x)
        expanded, valid, mass = expand_states(hidden,mapped)
        sequences, scores, modes = {}, {}, {}
        for arm in ("native","copies","corrected"):
            state = hidden if arm=="native" else expanded
            attn = x.attention_mask if arm=="native" else valid
            context = weighted_cross_attention(net,mass) if arm=="corrected" else nullcontext()
            with context:
                encoded = BaseModelOutput(last_hidden_state=state)
                logits = net(encoder_outputs=encoded,attention_mask=attn,labels=labels,use_cache=False).logits
                scores[arm] = logits.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
                modes[arm] = logits.argmax(-1)
                del logits
                written = net.generate(encoder_outputs=encoded,attention_mask=attn,num_beams=1,do_sample=False,
                                       max_new_tokens=128,forced_eos_token_id=None)
                sequences[arm] = [until_stop(s,net.config.eos_token_id) for s in written.cpu()]
        for arm in stats:
            diff = (scores[arm]-scores["native"]).abs()[mask]
            q = stats[arm]
            q["positions"] += int(mask.sum())
            q["absolute_sum"] += diff.double().sum().item()
            q["maximum"] = max(q["maximum"],diff.max().item())
            q["mode_changes"] += int(((modes[arm]!=modes["native"])&mask).sum())
        for i,r in enumerate(batch):
            seq = {m:v[i] for m,v in sequences.items()}
            for arm in stats:
                stats[arm]["equal_free"] += seq[arm]==seq["native"]
                stats[arm]["stopped"] += seq[arm][-1]==net.config.eos_token_id
            rec = dict(work=r["work"], pieces=int(x.attention_mask[i].sum()), cells=len(mapped[i]), sequences=seq)
            if seq["corrected"]!=seq["native"]:
                first = next(k for k,(a,b) in enumerate(zip(seq["corrected"],seq["native"])) if a!=b)
                prefix = torch.tensor([[net.config.decoder_start_token_id]+seq["native"][:first]],device=hidden.device)
                diagnostic = net(encoder_outputs=BaseModelOutput(last_hidden_state=hidden[i:i+1]),
                                 attention_mask=x.attention_mask[i:i+1],decoder_input_ids=prefix,use_cache=False)
                vals = diagnostic.logits[0,-1].topk(2).values
                rec.update(first_difference=first,native_top_two_margin=(vals[0]-vals[1]).item())
            rows.append(rec)
    for q in stats.values():
        q["mean_absolute"] = q["absolute_sum"]/q["positions"]
    return dict(model=name,revision=version,requests=len(rows),unrepresentable=0,
                native_stops=sum(r["sequences"]["native"][-1]==net.config.eos_token_id for r in rows),
                conditions=stats), rows


def main():
    configure_cuda_cli()
    pairs = library("dev")
    results, raw = {}, {}
    with gpu_owner():
        torch.set_float32_matmul_precision("highest")
        for t,lang in enumerate(("en","zh")):
            start = monotonic()
            results[lang],raw[lang] = direction(t,pairs)
            results[lang]["seconds"] = monotonic()-start
            print(lang, results[lang],flush=True)
    file = RUN/"glyph-measure-outputs.json"
    dump(file,raw)
    binding = {p:sha(ROOT/p) for p in ["scripts/check_glyph_measure.py","ilm/visual_lm/glyph_measure.py",
               "formal/span_list/GlyphAttentionMeasure.lean","references/glyph_attention_measure_protocol.md",
               "references/evidence/visual_library_v1.json"]}
    dump(PUB/"glyph-measure.json",dict(binding=binding,raw=sha(file),results=results))


if __name__=="__main__":
    main()
