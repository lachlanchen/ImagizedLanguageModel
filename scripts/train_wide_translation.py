"""Match the trained reference-prefix control while expanding its starting function."""

from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.wide_translation import widen
from scripts.reserve_wide_translation import (
    ROOT,
    BOOK,
    PUB,
    WORK,
    PREFIX,
    read,
    dump,
    sha,
    training_rows,
)
from scripts.train_visual_prefix import (
    streams,
    development,
    schedule,
    material,
    restore,
    gpu_owner,
    atomic_save,
)
from scripts.verse_gpu_runtime import configure_cuda_cli

MODEL = ROOT / "models/wide_translation"
CONTROL = ROOT / "models/visual_prefix/reference"


def fit():
    files = [
        ROOT / s
        for s in (
            "scripts/train_wide_translation.py",
            "ilm/visual_lm/wide_translation.py",
            "references/visual_writer_width_protocol.md",
            "references/evidence/wide_translation_v1/reservation.json",
            "models/book_breadth/rehearsal/weights.safetensors",
            "models/visual_prefix/reference/fit.json",
            "scripts/train_visual_prefix.py",
        )
    ]
    binding = {str(p.relative_to(ROOT)): sha(p) for p in files}
    if sha(WORK / "fresh.json") != read(PUB / "reservation.json")["sha256"]:
        raise ValueError("Reservation changed")
    ref = read(CONTROL / "fit.json")
    if any(sha(ROOT / p) != d for p, d in ref["binding"].items()):
        raise ValueError("Control training inputs changed")
    MODEL.mkdir(exist_ok=True)
    path = MODEL / "weights.safetensors"
    if (MODEL / "fit.json").exists():
        result = read(MODEL / "fit.json")
        if result["binding"] != binding or result["weights"] != sha(path):
            raise ValueError("Changed fit")
        return
    cells = training_rows()
    if (
        sha(PREFIX / "drafts.npz")
        != read(ROOT / "references/evidence/prefix_recovery_v1/drafts.json")["archive"]
    ):
        raise ValueError("Changed aligned supervision")
    with np.load(PREFIX / "drafts.npz") as z:
        inputs, targets = streams(cells, z, "reference")
    lengths = np.count_nonzero(cells, 2)
    counts = np.count_nonzero(targets, 2)
    steps = np.count_nonzero(inputs, 2)

    def gpu(a):
        return torch.from_numpy(a.astype(np.int32)).cuda()

    src, feed, truth = map(gpu, (cells, inputs, targets))
    del cells, inputs, targets
    torch.manual_seed(90231)
    model = widen(restore("book-rehearsal", "cpu")).cuda()
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.00016, fused=True)
    optical = material("cuda")
    features, inv, local = optical
    dev = read(BOOK / "dev.json")
    initial = development(model, dev, optical)
    history = []
    best_score = -1
    resume = WORK / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["binding"] != binding:
            raise ValueError("Resume inputs changed")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        history, best, best_score, selected = (
            saved[k] for k in ("history", "best", "score", "selected")
        )
        del saved
    for ep in range(len(history), 3):
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        model.train()
        opt.param_groups[0]["lr"] = (0.00016, 0.00010, 0.00006)[ep]
        order = schedule(lengths, 8 + ep)
        total_loss = 0.0
        tally = 0
        for idx in order:
            ii = torch.as_tensor(idx, device="cuda")
            source_width = lengths[idx].max(0)
            target_width = steps[idx].max(0)
            n = int(counts[idx].sum())
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                parts = []
                for t in (0, 1):
                    y = feed[ii, t, : target_width[t]].long()
                    x = src[ii, 1 - t, : source_width[1 - t]].long()
                    targets_local = local[t][truth[ii, t, : target_width[t]].long()]
                    logits = model(features, x, y, inv[t], t)
                    parts.append(
                        F.cross_entropy(
                            logits.float().transpose(1, 2),
                            targets_local,
                            reduction="sum",
                            ignore_index=-100,
                        )
                    )
                loss = sum(parts) / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            total_loss += float(loss.detach()) * n
            tally += n
            if (
                torch.cuda.max_memory_allocated() > 16 * 2**30
                or monotonic() - start + sum(h["seconds"] for h in history) > 3600
            ):
                raise RuntimeError("Width trial exceeded its resource limit")
        old = ref["epochs"][ep]
        if (len(order), tally, int(steps.sum())) != (
            old["updates"],
            old["target_positions"],
            old["feedback_positions"],
        ):
            raise ValueError("Exposure differs from narrow control")
        scores = development(model, dev, optical)
        score = float(np.mean(list(scores.values())))
        if score > best_score:
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_score, selected = score, ep + 1
        history.append(
            dict(
                epoch=ep + 1,
                nll=total_loss / tally,
                development=scores,
                updates=len(order),
                target_positions=tally,
                feedback_positions=int(steps.sum()),
                seconds=monotonic() - start,
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        atomic_save(
            lambda p: torch.save(
                dict(
                    binding=binding,
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    best=best,
                    history=history,
                    score=best_score,
                    selected=selected,
                ),
                p,
            ),
            resume,
        )
        print(history[-1], flush=True)
    atomic_save(lambda p: save_file(best, p), path)
    dump(
        MODEL / "fit.json",
        dict(
            binding=binding,
            initial_development=initial,
            selected_pass=selected,
            epochs=history,
            parameters=params,
            replicas=3,
            weights=sha(path),
        ),
    )


if __name__ == "__main__":
    configure_cuda_cli()
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 17 * 2**30:
            raise RuntimeError("Insufficient free GPU memory")
        fit()
