"""Can better search alone recover the missing published words under the frozen model?"""

from zipfile import ZipFile
import numpy as np
import torch
from ilm.visual_lm.passage_selector import PassageWriter, MAX_WORD
from scripts.evaluate_passage_selector import ARCHIVE, SPLITS, read_ink
from scripts.train_passage_selector import load_bank
from scripts.prepare_passage_selector import WORK, PUB, base
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


@torch.inference_mode()
def main():
    configure_cuda_cli()
    results = {}
    with gpu_owner(), ZipFile(ARCHIVE) as z:
        bank = load_bank("cuda")
        model = PassageWriter("intact", "cuda").net
        known = {w: i for i, w in enumerate(bank.words)}
        for split in SPLITS:
            with np.load(WORK / (split + ".npz")) as data:
                source, marks, task, y = [
                    data[k] for k in ("source", "marks", "task", "target")
                ]
            emitted = read_ink(z, split + "/intact")
            reference = read_ink(z, split + "/gold0")
            records = []
            for start in range(0, len(y), 128):
                cut = slice(start, start + 128)
                src, mm, tt = [
                    torch.as_tensor(v[cut], device="cuda").long()
                    for v in (source, marks, task)
                ]
                q, a, g = model.observe(bank, src, mm, tt)

                def logmass(images):
                    codes = []
                    lengths = []
                    addresses = []
                    legal = []
                    for pixels in images:
                        glyphs = tuple(
                            bank.face.address.get(bytes(cell), 0) for cell in pixels
                        )
                        valid = 0 < len(glyphs) <= MAX_WORD and 0 not in glyphs
                        prefix = glyphs if valid else ()
                        legal.append(valid)
                        lengths.append(len(prefix))
                        addresses.append(known.get(glyphs, -1))
                        codes.append([*prefix, *([0] * (MAX_WORD + 1 - len(prefix)))])
                    mask = src == torch.tensor(addresses, device="cuda")[:, None]
                    copy = (a * mask).sum(1) * g
                    spell = model.spell_logprob(
                        q,
                        torch.tensor(codes, device="cuda"),
                        torch.tensor(lengths, device="cuda"),
                    )
                    total = torch.logaddexp(copy.log(), torch.log1p(-g) + spell)
                    return total.masked_fill(
                        ~torch.tensor(legal, device="cuda"), -torch.inf
                    ).cpu().numpy(), legal

                out, _ = logmass(emitted[cut])
                gold, legal = logmass(reference[cut])
                for j, (e, r) in enumerate(
                    zip(emitted[cut], reference[cut], strict=True)
                ):
                    diff = float(gold[j] - out[j])
                    records.append((bool(legal[j]), np.array_equal(e, r), diff))
            legal, correct, delta = map(np.array, zip(*records))
            within = (source == y[:, 0, None]).any(1)
            totals = {}
            for name, mask in [
                ("all", np.ones(len(y), bool)),
                ("present", within),
                ("absent", ~within),
            ]:
                errors = mask & legal & ~correct
                totals[name] = dict(
                    n=int(mask.sum()),
                    correct=int((mask & correct).sum()),
                    unreachable=int((mask & ~legal).sum()),
                    reference_below_emitted=int((errors & (delta < -1e-4)).sum()),
                    reference_above_emitted=int((errors & (delta > 1e-4)).sum()),
                    numerical_tie=int((errors & (np.abs(delta) <= 1e-4)).sum()),
                    median_wrong_log_reference_minus_emission=float(
                        np.median(delta[errors])
                    ),
                )
            results[split] = totals
            print(split, totals, flush=True)
    base.dump(
        PUB / "masses.json",
        dict(archive=base.sha(ARCHIVE), epsilon_nats=1e-4, diagnosis=results),
    )


if __name__ == "__main__":
    main()
