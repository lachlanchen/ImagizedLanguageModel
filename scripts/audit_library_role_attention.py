"""Measure source-role alignment of frozen literary writers on the two development books."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.role_attention import compare_assignments, source_role_mass, window_roles
from scripts.prepare_library_roles import LABELS, PUB, RUN
from scripts.prepare_teacher_translation import dump
from scripts.sample_visual_library import MODELS, ROOT, library, read, sha
from scripts.train_visual_prefix import material, pack
from scripts.write_ordered_translation import restore


@torch.inference_mode()
def audit(model, pairs, tags, optical):
    features, inventory, _ = optical
    sums = defaultdict(lambda: np.zeros(3))
    for t, language in enumerate(("en", "zh")):
        for start in range(0, len(pairs), 24):
            batch = pairs[start:start + 24]
            src = pack([r["cells"][1-t] for r in batch], "cpu")
            tgt = pack([r["cells"][t] + [1] for r in batch], "cpu")
            annotations = tags[start:start + len(batch)]
            source_roles = annotations[:, 1-t, :src.size(1)]
            target_roles = torch.zeros_like(tgt)
            width = min(tgt.size(1), annotations.size(2))
            target_roles[:, :width] = annotations[:, t, :width]
            fractions = window_roles(src, source_roles, model.stride)
            masses = []

            def measure(module, args):
                query, (keys, _), valid = args
                masses.append(source_role_mass(module, query, keys, valid, fractions))

            handle = model.dec[-1].cross.register_forward_pre_hook(measure)
            try:
                model(features, src, tgt, inventory[t], t)
            finally:
                handle.remove()
            mass, = masses
            if not torch.allclose(mass.sum(-1), torch.ones_like(tgt, dtype=mass.dtype), atol=2e-6):
                raise ValueError("Role fractions did not conserve attention")
            actual, null, valid = compare_assignments(mass, source_roles, target_roles)
            for j, row in enumerate(batch):
                for role, tag in LABELS.items():
                    take = valid[j] & target_roles[j].eq(tag)
                    values = [int(take.sum()), actual[j, take].double().sum().item(),
                              null[j, take].double().sum().item()]
                    for group in ("total", language, language + "/" + role, row["work"]):
                        sums[group] += values
    return {group: dict(positions=int(n), observed=a/n, expected=b/n, excess=(a-b)/n)
            for group, (n, a, b) in sums.items() if n}


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    evidence = read(PUB / "alignment.json")
    if sha(RUN / "dev.npy") != evidence["files"]["dev.npy"]:
        raise ValueError("Changed development annotations")
    pairs = library("dev")
    tags = torch.from_numpy(np.load(RUN / "dev.npy"))
    counts = evidence["splits"]["dev"]["eligible_target"]
    optical = material("cpu")
    results = {}
    for arm in ("repeat", "broad"):
        results[arm] = audit(restore("library-" + arm, "cpu"), pairs, tags, optical)
        for lang in ("en", "zh"):
            if results[arm][lang]["positions"] != counts[lang]["positions"]:
                raise ValueError("Mismatch with registered count")
    binding = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__).resolve(), ROOT / "ilm/visual_lm/role_attention.py", PUB / "alignment.json",
        MODELS / "repeat/weights.safetensors", MODELS / "broad/weights.safetensors",
        ROOT / "references/library_role_attention_protocol.md")}
    dump(PUB / "frozen-attention.json", dict(binding=binding, results=results))
    print({k: v["total"] for k, v in results.items()})


if __name__ == "__main__":
    main()
