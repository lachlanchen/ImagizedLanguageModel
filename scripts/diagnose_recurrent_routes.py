"""Ablate the complete-span route after fitting and locate where paired answers fail."""

import gzip, json
import numpy as np
import torch
from scripts.prepare_recurrent_evidence import WORK, PUBLIC, dump
from scripts.train_recurrent_evidence import Data, Corpus
from scripts.train_paired_evidence import load_model, gpu_owner
from scripts.write_recurrent_evidence import load, generate
from scripts.evaluate_recurrent_evidence import inputs, scored, summary
from ilm.visual_lm.book_prefix import observations
from scripts.train_visual_cursor import pad


@torch.inference_mode()
def main():
    data = Data()
    old = Corpus()
    _, held, targets, probes, tiles, chars, by = inputs(data, old)
    with gpu_owner():
        base = load_model("ink-pair", "cuda")
        net = load("dynamic", "cuda")
        x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        bank = net.ink(x)
        raw = []
        full = json.loads(
            gzip.decompress((WORK / "dynamic-private-answers.json.gz").read_bytes())
        )["answers"]
        for k in range(0, len(held), 16):
            rows = held[k : k + 16]
            q, s = [
                torch.as_tensor(pad([r[key] for r in rows]), device="cuda")
                for key in ("q", "s")
            ]
            prior = net.read(bank, q, s)[3].sigmoid().tolist()
            raw.extend(
                dict(
                    id=r["id"],
                    lang=r["lang"],
                    subset=r["subset"],
                    contiguous=t["single_source"],
                    prior=float(p),
                )
                for r, t, p in zip(rows, full[k : k + 16], prior, strict=True)
            )
        # The experimental intervention is a constant probability of zero on the one-span route.
        net.span.weight.zero_()
        net.span.bias.fill_(-1000)
        generated = []
        for k in range(0, len(held), 16):
            rs = held[k : k + 16]
            q = [r["q"] for r in rs]
            s = [r["s"] for r in rs]
            out = generate(
                base, net, x, old.n, q, s, [a + b for a, b in zip(s, q)], 160
            )
            generated.extend(
                scored(r, "".join(chars[i] for i in ids), end)
                for r, (ids, end) in zip(rs, out, strict=True)
            )
        stats = {}
        for lang in ("en", "zh"):
            rr = [r for r in raw if r["lang"] == lang and r["subset"] == "absent"]
            stats[lang] = dict(
                median_span_prior=float(np.median([r["prior"] for r in rr])),
                above_half=sum(r["prior"] > 0.5 for r in rr),
                disabled=summary(
                    [
                        g
                        for g in generated
                        if g["lang"] == lang and g["subset"] == "absent"
                    ]
                ),
            )
        examples = {}
        for r in probes:
            ((ids, end),) = generate(
                base, net, x, old.n, [r["q"]], [r["s"]], [r["s"] + r["q"]], 96
            )
            examples[r["id"]] = dict(output="".join(chars[i] for i in ids), stopped=end)
        dump(
            PUBLIC / "span-ablation.json",
            dict(summary=stats, original_prompts=examples, priors=raw),
        )
        print(stats, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
