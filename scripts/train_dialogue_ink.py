"""Train the word-pointer and word-spelling visual dialogue writer on one GPU."""

import argparse
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.dialogue_ink import DialogueInk, WordCodeDialogueInk
from scripts.pretrain_dialogue_spelling import pretrain, alignment_loss
from scripts.answer_optical_passage_qa import load as parent, RELEASE as READER, PARENT
from scripts.dialogue_ink_data import Corpus, read_data
from scripts.prepare_dialogue_ink import PUB, dump, sha
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

SEED = 710491


def targets(net, bank, idx):
    x, variants = bank.inputs(idx)
    mem, ev = net.memory(*x)
    ans = F.pad(bank.ans[idx].long(), (0, 1))
    lens = bank.al[idx]
    t = torch.arange(ans.size(1), device=ans.device)
    prev = bank.bank.features[F.pad(ans[:, :-1], (1, 0)), :50]
    codes, ptr, latent, _ = net.step(mem, prev, mem[-1][None])
    ptr = ptr.masked_fill(variants[:, None] == 0, -1e4).flatten(2)
    matches = (variants.flatten(1)[:, None, :] == ans[:, :, None]) & (
        ans[:, :, None] != 0
    )
    copy = matches.any(-1)
    word = t[None] < lens[:, None]
    active = t[None] <= lens[:, None]
    channels = torch.where(word, torch.where(copy, 1, 2), 0)
    return dict(
        channel=codes,
        target=channels,
        active=active,
        copy=copy,
        word=word,
        ptr=ptr,
        matches=matches,
        ans=ans,
        latent=latent,
        evidence=ev,
    )


def dialogue_loss(net, train, idx, labels):
    d = targets(net, train, idx)
    channel = F.cross_entropy(d["channel"][d["active"]], d["target"][d["active"]])
    copy = d["copy"]
    copylog = d["ptr"][copy].log_softmax(-1).masked_fill(~d["matches"][copy], -1e4)
    copyloss = -copylog.logsumexp(-1).sum() / copy.sum().clamp(min=1)
    spell = d["word"] & ~copy
    letters = labels[d["ans"][spell]].long()
    spellloss = channel.new_zeros(())
    if len(letters):
        width = min(25, int((letters != 0).sum(1).max()) + 1)
        letters = letters[:, :width]
        spellloss = net.spell_loss(d["latent"][spell], letters, net.letter_inputs())
    ev = train.ev[idx]
    valid = ev[:, 0] >= 0
    aux = channel.new_zeros(())
    if valid.any():
        aux = sum(
            F.cross_entropy(d["evidence"][valid, :, j], ev[valid, j]) for j in (0, 1)
        )
    loss = channel + copyloss + spellloss + 0.20 * aux
    return loss, (channel, copyloss, spellloss, aux), d, spell, letters


def fit(output, word_code=False):
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    old, optical = parent("cuda")
    data = read_data()
    start = monotonic()
    meter = snapshot()
    train = Corpus(data, "train", old, optical)
    palette, labels = train.glyph_targets()
    net = (WordCodeDialogueInk if word_code else DialogueInk)(palette).cuda()
    net.reader.load_state_dict(
        {k: v for k, v in old.state_dict().items() if not k.startswith("trim.")}
    )
    del old
    torch.cuda.synchronize()
    prep = dict(seconds=monotonic() - start, energy=consumed(meter, snapshot()))
    inverse = pretrain(net, train, palette, SEED) if word_code else None
    opt = torch.optim.AdamW(net.parameters(), lr=0.00075)
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    trace = []
    seen = np.zeros(len(train.rows), bool)
    for step in range(8000):
        idx = rng.integers(len(seen), size=32)
        seen[idx] = True
        loss, parts, d, spell, letters = dialogue_loss(net, train, idx, labels)
        channel, copyloss, spellloss, aux = parts
        if word_code:
            idx_word = d["ans"][d["word"]]
            code = net.observed_code(train.bank.features[idx_word])
            loss = loss + alignment_loss(
                net.predicted_code(d["latent"][d["word"]]), code, idx_word
            )
            if len(letters):
                observed = net.observed_code(train.bank.features[d["ans"][spell]])
                loss = loss + 0.35 * net.supplied_loss(
                    observed, letters, net.letter_inputs()
                )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        if (step + 1) % 250 == 0:
            trace.append(
                [float(z.detach()) for z in (channel, copyloss, spellloss, aux)]
            )
            print(step + 1, trace[-1], round(monotonic() - start, 1), flush=True)
            if (
                monotonic() - start > 3000
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Fit exceeded registered resources")
    torch.cuda.synchronize()
    cost = dict(
        seconds=monotonic() - start,
        energy=consumed(meter, snapshot()),
        peak_bytes=torch.cuda.max_memory_allocated(),
    )
    output.mkdir(parents=True)
    f = output / "writer.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(f)
    )
    meta = dict(
        architecture="word_code_dialogue_ink_v1" if word_code else "dialogue_ink_v1",
        sha256=sha(f),
        input_reader=sha(READER / "reader.safetensors"),
        optical_encoder=sha(PARENT),
        split=sha(PUB / "split.json"),
        seed=SEED,
        updates=8000,
        presentations=256000,
        distinct_sampled=int(seen.sum()),
        trainable=sum(v.numel() for v in net.parameters()),
        frozen=sum(v.numel() for v in optical.parameters()),
        glyph_templates=len(palette) - 1,
        preprocessing=prep,
        fit=cost,
        loss_trace=trace,
    )
    if inverse is not None:
        meta["word_inverse"] = inverse
    dump(output / "weights.json", meta)
    print(meta, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--word-code", action="store_true")
    args = p.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        fit(args.output, args.word_code)
