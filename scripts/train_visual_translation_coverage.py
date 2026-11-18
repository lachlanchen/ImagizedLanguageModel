"""Register and fit two image writers with identical book replay and matched length budgets."""

import argparse
from collections import Counter
import fcntl
import hashlib

import numpy as np
import torch

from scripts.benchmark_visual_coverage_teacher import rows as test_rows
from scripts.prepare_visual_translation_coverage import DEST, WORK, ROOT, load, write_new
from scripts.review_visual_translation_coverage import roster, previously_reviewed
from scripts.train_screened_visual_targets import PARENT, DATA, SEED, Rasterizer, data_sets, fit, sha

ARMS = ("narrow", "broad")
REGISTER = DEST / "registration.json"
WEIGHTS = ROOT / "models/visual_edit_zh/coverage"


def final_roster():
    sources, _ = roster()
    decisions = previously_reviewed()
    if set(decisions) != {r["row"] for r in sources} or len(sources) != 1024:
        raise ValueError("Incomplete review")
    accepted = [r | decisions[r["row"]] for r in sources if decisions[r["row"]]["action"] != "exclude"]
    render = Rasterizer()
    def bitmap(s):
        return hashlib.sha256(b"".join(map(render.tile, s))).digest()
    held = {bitmap(r[k]) for r in test_rows() for k in ("source", "target")}
    for r in accepted:
        if any(bitmap(r[k]) in held for k in ("source", "target")):
            raise ValueError(f"Test overlap at training row {r['row']}")
    return accepted, Counter(r["action"] for r in decisions.values())


def plans(accepted):
    replay = np.random.default_rng(SEED).permutation(55872).reshape(1164, 48)
    result = {}
    for arm in ARMS:
        supplements = []
        for band in range(4):
            eligible = [55872 + i for i, r in enumerate(accepted)
                        if r["band"] == band and (arm == "broad" or r["narrow_subset"])]
            rng = np.random.default_rng([SEED, band])
            cycles = [rng.permutation(eligible) for _ in range((4656 + len(eligible)-1)//len(eligible))]
            supplements.append(np.concatenate(cycles)[:4656].reshape(1164, 4))
        result[arm] = np.concatenate([replay, *supplements], axis=1)
    return result


def experiment(train):
    accepted, counts = final_roster()
    plans_by_arm = plans(accepted)
    idx = [r["row"] for r in accepted]
    packed, original, corrected, same = data_sets(accepted)
    combined = tuple(np.concatenate((o, c[idx])) for o,c in zip(original, corrected))
    del original, corrected
    pads = np.maximum.reduce([np.stack([combined[k][b].max(axis=1) for k in (2,3)],axis=1)
                              for b in plans_by_arm.values()])
    files = [DEST/name for name in ("draft_manifest.json","sources.json","fresh_assessment.json","holdout.json",
              "isolation.json","sample_editorial_review.json","teacher_test_answers.json")] + sorted((DEST/"editorial").glob("*.json"))
    def digest(a):
        return hashlib.sha256(a.astype("<i8").tobytes()).hexdigest()
    ledger = dict(parent_sha=sha(PARENT), training_sha=sha(DATA/"training.npz"), files={str(p.relative_to(DEST)):sha(p) for p in files},
                  seed=SEED, decisions=dict(counts), unchanged_book_targets=same,
                  corrected_holdout_collisions=0, padding_sha=digest(pads), steps=1164,
                  shared_book_replay_sha=digest(plans_by_arm["narrow"][:,:48]), per_band_exposures=4656,
                  adamw_lr=0.0002, clip=1.0, per_arm_limits=dict(gpu_gib=10, available_host_gib_min=24, seconds=1800),
                  arms={a:dict(batches_sha256=digest(b), distinct_reviewed=len(np.unique(b[:,48:])),
                         source_cells=int(combined[2][b].sum()),target_cells=int(combined[3][b].sum()))
                         for a,b in plans_by_arm.items()})
    if not train:
        write_new(REGISTER,ledger)
        return
    if load(REGISTER) != ledger or ledger["parent_sha"] != "71053d75fd2c323c305b7a36c7f49d2fd575d3c99373ac3c5d184b43e70575da":
        raise ValueError("Registered comparison changed")
    WEIGHTS.mkdir(exist_ok=True, parents=True)
    for arm,b in plans_by_arm.items():
        fit(arm,packed,combined,combined[3],b,ledger|ledger["arms"][arm],out=WEIGHTS,evidence=DEST,padding=pads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","train"))
    args = parser.parse_args()
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        experiment(args.action=="train")
