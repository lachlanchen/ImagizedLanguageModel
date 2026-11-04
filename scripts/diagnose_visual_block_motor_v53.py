#!/usr/bin/env python3
"""Post-result, target-supplied motor diagnostic. Never autonomous evidence."""

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.causal_visual_block_diffusion_v53 import (
    CausalVisualBlockDiffusion,
    ConditionalStripDenoiser,
    causal_visual_block_config_from_payload,
)
from ilm.visual_lm.visual_cell_data import load_v25_records, pack_visual_cells
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from ilm.visual_lm.visual_posterior_language_v51_evaluation import pixel_f1
from scripts.eval_causal_visual_block_diffusion_v53 import AUDIT_SEED
from scripts.train_causal_visual_block_diffusion_v53 import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_READER,
    load_frozen_reader,
)


@torch.no_grad()
def main():
    device = torch.device("cuda:0")
    out = Path(DEFAULT_OUTPUT) / "target_supplied_motor_diagnostic"
    out.mkdir(exist_ok=True)
    checkpoint = torch.load(
        Path(DEFAULT_OUTPUT) / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    reader = load_frozen_reader(DEFAULT_READER, device)
    writer = ConditionalStripDenoiser(
        causal_visual_block_config_from_payload(checkpoint["model_config"])
    )
    writer.load_state_dict(checkpoint["ema_denoiser"])
    model = CausalVisualBlockDiffusion(reader, writer).to(device).eval()
    records = load_v25_records(DEFAULT_MANIFEST, strict_manifest=True)
    render = CanonicalGlyphRenderConfig()
    statistics = build_visual_character_statistics(records)
    windows = build_visual_cell_audit_windows(
        records, statistics, count=512, continuation_cells=4, seed=AUDIT_SEED
    )
    dataset = CanonicalGlyphAuditDataset(windows[:64], statistics, render_config=render)
    loader = DataLoader(
        dataset, batch_size=16, collate_fn=canonical_glyph_audit_collate
    )
    bank = render_canonical_character_bank(statistics, render_config=render)
    fields = reader.field.encode_unit(bank.to(device))
    totals = {
        route: []
        for route in (
            "predicted",
            "exact_plan",
            "exact_plan_and_anchor",
            "inverse_only",
        )
    }
    gallery = []
    offset = 0
    for raw in loader:
        context = raw["context"][:, :60].to(device)
        target = raw["context"][:, 60].to(device)
        hidden, anchor, plan = model.read_condition(context)
        exact_field = reader.field.encode_unit(target)
        states = {
            "predicted": (hidden, anchor, plan),
            "exact_plan": (hidden, anchor, target),
            "exact_plan_and_anchor": (hidden, exact_field, target),
        }
        outputs = {}
        for route, state in states.items():
            rng = torch.Generator(device=device).manual_seed(
                AUDIT_SEED + offset * 104729
            )
            noisy = torch.randn(
                (len(context), 4, 1, 32, 32), device=device, generator=rng
            )
            schedule = model.sampling_schedule()
            for index, step in enumerate(schedule[:-1]):
                times = torch.full(
                    (len(context), 4), step, device=device, dtype=torch.long
                )
                null, _ = writer(
                    noisy,
                    torch.zeros_like(hidden),
                    torch.zeros_like(anchor),
                    torch.zeros_like(plan),
                    times,
                )
                conditional, _ = writer(noisy, *state, times)
                velocity = null + model.config.classifier_free_scale * (
                    conditional - null
                )
                clean, noise = model.velocity_to_clean_noise(noisy, velocity, times)
                alpha = (
                    model.alpha_bars[schedule[index + 1]]
                    if schedule[index + 1] > 0
                    else noisy.new_tensor(1.0)
                )
                noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            outputs[route] = (clean[:, 0] >= 0).float()
        outputs["inverse_only"] = reader.field.binary(exact_field)
        target_sim = exact_field @ fields.T
        represented = target_sim.max(-1).values >= 1 - 1e-6
        target_id = target_sim.argmax(-1)
        for route, output in outputs.items():
            selected = (reader.field.encode_unit(output) @ fields.T).argmax(-1)
            totals[route].append(
                torch.stack(
                    (
                        (selected == target_id).float() * represented,
                        pixel_f1(output, target),
                    ),
                    dim=-1,
                ).cpu()
            )
        gallery.append(torch.stack((target, *outputs.values()), dim=1).cpu())
        offset += len(context)
    report = {
        route: {
            "identity_top1": float(torch.cat(rows)[:, 0].mean()),
            "pixel_f1": float(torch.cat(rows)[:, 1].mean()),
        }
        for route, rows in totals.items()
    }
    report.update(
        {
            "examples": offset,
            "target_supplied": True,
            "autonomous_language_evidence": False,
            "gallery_columns": ["target", *totals],
        }
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    pack_visual_cells(torch.cat(gallery)[:16].flatten(0, 1), columns=5, gutter=2).save(
        out / "gallery.png"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
