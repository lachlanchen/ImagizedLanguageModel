"""Learn a glyph-image adapter using typeface pairs, independently of sentence data."""

from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
from fontTools.ttLib import TTFont
from PIL import ImageFont
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.glyph_typeface_reader import TypefaceReader
from scripts.prepare_book_rehearsal import ROOT, BASE, WORK, read, dump, sha
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_fact_ranking import atomic_save
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

MODEL = ROOT / "models/book_breadth/typeface"
FONTS = Path("/usr/share/fonts")
FACES = [
    ("opentype/noto/NotoSansCJK-" + w + ".ttc", 2)
    for w in ("Regular", "Thin", "Medium", "Bold", "Black")
] + [
    ("truetype/arphic-gbsn00lp/gbsn00lp.ttf", 0),
    ("truetype/arphic-gkai00mp/gkai00mp.ttf", 0),
    ("truetype/dejavu/DejaVuSerif.ttf", 0),
    ("truetype/dejavu/DejaVuSans.ttf", 0),
    ("truetype/liberation/LiberationSerif-Regular.ttf", 0),
]


def render_face(file, index):
    face = Rasterizer()
    face.font = ImageFont.truetype(str(file), 26, index=index)
    with TTFont(file, fontNumber=index) as ft:
        face.coverage = set(ft.getBestCmap())
    return face


def prepare(palette):
    chars = read(BASE / "readback.json")
    pictures = [b for b in map(bytes, palette)]
    labels = list(range(len(palette)))
    for file, index in FACES:
        face = render_face(FONTS / file, index)
        for k in range(len(palette)):
            c = chars[str(k + 2)]
            if ord(c) not in face.coverage and not c.isspace():
                continue
            try:
                img = face.tile(c)
            except ValueError:
                continue
            pictures.append(img)
            labels.append(k)
    return np.frombuffer(b"".join(pictures), np.uint8).reshape(-1, 128), labels


def main():
    configure_cuda_cli()
    palette = np.load(BASE / "palette.npy")[2:]
    MODEL.mkdir(parents=True, exist_ok=True)
    meta = dict(
        palette=sha256(palette.tobytes()).hexdigest(),
        sources={
            str(f): sha(ROOT / f)
            for f in [
                "scripts/train_rehearsal_typeface.py",
                "ilm/visual_lm/glyph_typeface_reader.py",
                "references/rehearsal_font_protocol.md",
            ]
        },
        fonts={file: sha(FONTS / file) for file, _ in FACES},
        readback=sha(BASE / "readback.json"),
    )
    if (MODEL / "fit.json").exists():
        old = read(MODEL / "fit.json")
        if any(old[k] != v for k, v in meta.items()) or old["weights"] != sha(
            MODEL / "weights.safetensors"
        ):
            raise ValueError("Existing optical fit differs")
        return
    packed, labels = prepare(palette)
    with gpu_owner():
        torch.manual_seed(67185)
        device = "cuda"
        model = TypefaceReader(len(palette)).to(
            device, memory_format=torch.channels_last
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, fused=True)
        steps = 3000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, steps, eta_min=0.00015
        )
        data = torch.from_numpy(
            np.unpackbits(packed, axis=1).reshape(-1, 1, 32, 32)
        ).to(device)
        target = torch.tensor(labels, device=device)
        start = monotonic()
        torch.cuda.reset_peak_memory_stats()
        history = []
        for step in range(steps):
            ids = torch.randint(len(data), (192,), device=device)
            x = data[ids].float()
            # Fixed canvas: translations expose fresh white margins.
            dy, dx = torch.randint(-2, 3, (2,)).tolist()
            x = F.pad(x, (2, 2, 2, 2))[:, :, 2 + dy : 34 + dy, 2 + dx : 34 + dx]
            if step % 8 == 1:
                x = F.max_pool2d(F.pad(x, (0, 1, 0, 1)), 2, 1)
            if step % 3 == 0:
                x = torch.where(torch.rand_like(x) < 0.004, 1 - x, x)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(x.contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(logits, target[ids])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (step + 1) % 500 == 0:
                history.append(
                    dict(
                        step=step + 1,
                        batch_loss=float(loss.detach()),
                        batch_accuracy=float(
                            (logits.argmax(-1) == target[ids]).float().mean()
                        ),
                    )
                )
                print(history[-1], flush=True)
        torch.cuda.synchronize()
        meta.update(
            seconds=monotonic() - start,
            peak_allocated=torch.cuda.max_memory_allocated(),
            parameters=sum(x.numel() for x in model.parameters()),
            steps=steps,
            pictures=len(data),
            history=history,
        )
        atomic_save(
            lambda p: save_file(
                {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, p
            ),
            MODEL / "weights.safetensors",
        )
        dump(
            MODEL / "fit.json", dict(**meta, weights=sha(MODEL / "weights.safetensors"))
        )


if __name__ == "__main__":
    main()
