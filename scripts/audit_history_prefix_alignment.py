#!/usr/bin/env python3
"""Audit retained image/language history correspondences without a model call."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


HELPER = Path("artifacts/reader_content_v77_20261106")
WRITER = Path("artifacts/native_chinese_v80_20261106/corpus_fit_v80r2")
PLAN = Path("references/history_prefix_alignment_protocol.md")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("L"), copy=True)


def same_prefix(prefix, reference, columns):
    return bool(np.array_equal(prefix[:, :columns], reference[:, :columns])
        and np.all(prefix[:, columns:] == 255))


def distance(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a = a / np.sqrt(np.mean(a * a) + 1e-6)
    b = b / np.sqrt(np.mean(b * b) + 1e-6)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def totals(cases):
    counts = Counter()
    for row in cases:
        for b in row["boundaries"]:
            kind = "initial" if b["position"] == 0 else "terminal" if b["position"] == row["blocks"] else "intermediate"
            counts[kind + "/boundaries"] += 1
            if b["canonical_prefix_indices"]:
                counts[kind + "/raster_realizable"] += 1
            if b["matches"]:
                counts[kind + "/source_state_available"] += 1
                counts[kind + "/matched_state_pairs"] += len(b["matches"])
            elif b["canonical_prefix_indices"]:
                counts[kind + "/raster_without_source_state"] += 1
            else:
                counts[kind + "/no_canonical_prefix_raster"] += 1
    return dict(sorted(counts.items()))


def run(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES='' for the CPU audit")
    import torch
    from tokenizers.decoders import ByteLevel
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
    from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
    from ilm.visual_lm.grounded_image_weights import MODEL_SHA

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("no CUDA context may be initialized")
    started = time.monotonic()
    generation = read(HELPER / "generation_report.json")
    hv = read(HELPER / "verification.json")
    prep = read(HELPER / "preparation.json")
    original = read(WRITER / "report.json")
    wv = read(WRITER / "verification.json")
    published_h = read("references/evidence/reader_content_v77_result.json")
    published_w = read("references/evidence/native_corpus_fit_v80_result.json")
    if (sha(HELPER / "generation_report.json") != hv["generation_report_sha256"]
            or hv["generation_report_sha256"] != published_h["generation_report_sha256"]
            or sha(HELPER / "preparation.json") != generation["preparation_sha256"]
            or sha(WRITER / "report.json") != wv["report_sha256"]
            or wv["report_sha256"] != published_w["source_report_sha256"]
            or sha(WRITER / "verification.json") != published_w["source_verification_sha256"]):
        raise ValueError("completed source reports differ from verified public identities")
    rows = [r for r in prep["cases"] if r["group"] == "training"]
    if len(rows) != 32:
        raise ValueError("all32 original training cases required")
    local = Path.cwd().parent / "LocalLLM/.local"
    gguf = load_gguf_reader_source(local / "tools/llama.cpp/gguf-py")
    archive_path = local / "models/ollama/blobs" / ("sha256-" + MODEL_SHA)
    # GGUFReader creates read-only mapped views. Only string metadata is used.
    archive = gguf.GGUFReader(archive_path, mode="r")
    if archive.fields["general.architecture"].contents() != "qwen3vl":
        raise ValueError("expected source tokenizer metadata")
    table = archive.fields["tokenizer.ggml.tokens"].contents()
    table_hash = hashlib.sha256(json.dumps(table, ensure_ascii=False).encode()).hexdigest()
    decoder = ByteLevel()
    native_inverse = {text: byte for byte, text in bytes_to_unicode().items()}
    eog = set(read(HELPER / "compiler.json")["eog_ids"])
    renderer = CompleteLineRenderer()
    for key in ("font_family", "font_size_px", "height_px", "padding_px", "primary_font_sha256", "pango_version"):
        if renderer.provenance[key] != prep["renderer"][key]:
            raise ValueError("original renderer changed: " + key)
    out.mkdir(parents=True, exist_ok=False)
    files, cases = {}, []

    def save_image(a, name):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(a).save(p)
        files[name] = sha(p)
        return name

    for row in rows:
        i = row["position"]
        trace_name = f"outputs/training_{i:02}_normal.pt"
        hp = HELPER / trace_name
        folder = WRITER / "predictions" / f"{i:02}"
        header_name = f"predictions/{i:02}/header.json"
        if sha(hp) != generation["raw_files_sha256"][trace_name] or sha(folder / "header.json") != original["files_sha256"][header_name]:
            raise ValueError("verified trace identity changed")
        header = read(folder / "header.json")
        for name in ("assisted_trace.pt", "reference.png"):
            if sha(folder / name) != header["files_sha256"][name]:
                raise ValueError("writer trace/reference changed")
        if (header["source"] != row["source"]
                or header["prompt_sha256"] != row["physical_tensor_sha256"]
                or header["prompt_memory_sha256"] != row["expected_memory_sha256"]):
            raise ValueError("owning prompts or reader memories differ")
        helper = torch.load(hp, map_location="cpu", weights_only=True)["result"]
        native = torch.load(folder / "assisted_trace.pt", map_location="cpu", weights_only=True)
        hs, ns = helper["hidden_states"].float(), native["states"][0].float()
        ids = helper["selected_ids"]
        if helper["stopped"]:
            if ids[-1] not in eog:
                raise ValueError("source termination differs")
            content_ids = ids[:-1]
        else:
            content_ids = ids
        pieces = [table[t] for t in content_ids]
        if decoder.decode(pieces) != helper["text"]:
            raise ValueError("metadata decoder does not reproduce source text")
        if len(hs) != len(ids) or hs.shape[1:] != (4096,) or ns.shape[1:] != (4096,):
            raise ValueError("full retained state sequences required")
        if not torch.isfinite(hs).all() or not torch.isfinite(ns).all():
            raise ValueError("finite states required")
        initial_equal = torch.equal(hs[0].to(torch.bfloat16), ns[0].to(torch.bfloat16))
        if not initial_equal:
            raise ValueError("same-prefill initial states differ after common BF16 rounding")
        reference = pixels(folder / "reference.png")
        rendered, meta = renderer.render(row["reference"], maximum_blocks=128)
        if not np.array_equal(reference, rendered) or reference.shape[1] % 64:
            raise ValueError("canonical full reference raster differs")
        blocks = reference.shape[1] // 64
        if len(ns) != blocks + 1:
            raise ValueError("all pre-block and terminal states required")
        prefix_records, prefix_images = [], []
        for k in range(len(row["reference"]) + 1):
            text = row["reference"][:k]
            image = np.full_like(reference, 255)
            if text.strip():
                layout, _ = renderer._layout(text)
                surface = renderer.cairo.ImageSurface(renderer.cairo.FORMAT_RGB24, reference.shape[1], 32)
                context = renderer.cairo.Context(surface)
                context.set_source_rgb(1, 1, 1); context.paint()
                context.set_source_rgb(0, 0, 0); context.move_to(meta["draw_x"], meta["draw_y"])
                renderer.PangoCairo.show_layout(context, layout); surface.flush()
                rgba = np.frombuffer(surface.get_data(), np.uint8).reshape(32, surface.get_stride() // 4, 4)
                image = np.where(rgba[:, :reference.shape[1], :3].mean(-1) >= 128, 255, 0).astype(np.uint8)
                if not (np.all(image[[0, -1]] == 255) and np.all(image[:, [0, -1]] == 255)):
                    raise ValueError("prefix layout reaches the original protected frame")
            prefix_images.append(image)
            prefix_records.append(dict(text=text, image=save_image(image, f"prefixes/{i:02}_{k:03}.png")))
        source_prefixes = []
        for k in range(min(len(content_ids) + 1, len(hs))):
            raw = bytes(native_inverse[c] for piece in pieces[:k] for c in piece)
            try:
                text = raw.decode("utf8", errors="strict"); complete = True
            except UnicodeDecodeError:
                text = raw.decode("utf8", errors="replace"); complete = False
            if text != decoder.decode(pieces[:k]):
                raise ValueError("strict byte reconstruction differs from source decoder")
            source_prefixes.append(dict(position=k, text=text, complete_utf8=complete))
        boundaries = []
        for j in range(blocks + 1):
            columns = j * 64
            canonical = ([0] if j == 0 else [k for k, a in enumerate(prefix_images) if same_prefix(a, reference, columns)])
            matches = []
            for sp in source_prefixes:
                if sp["complete_utf8"] and any(sp["text"] == prefix_records[k]["text"] for k in canonical):
                    matches.append(dict(source_position=sp["position"], text=sp["text"],
                        normalized_state_rmse=distance(ns[j].numpy(), hs[sp["position"]].numpy())))
            boundaries.append(dict(position=j, columns=columns, canonical_prefix_indices=canonical, matches=matches))
        arrays = f"states/{i:02}.npz"
        (out / "states").mkdir(exist_ok=True)
        np.savez_compressed(out / arrays, writer=ns.numpy(), source=hs.numpy())
        files[arrays] = sha(out / arrays)
        cases.append(dict(source=row["source"], position=i, reference_text=row["reference"],
            reference=save_image(reference, f"references/{i:02}.png"), blocks=blocks,
            physical_prompt_sha256=row["physical_tensor_sha256"], initial_bf16_state_equal=initial_equal,
            source_trace_sha256=sha(hp), writer_trace_sha256=sha(folder / "assisted_trace.pt"),
            source_text=helper["text"], source_stopped=helper["stopped"], source_pieces=pieces,
            source_prefixes=source_prefixes, canonical_prefixes=prefix_records, boundaries=boundaries, states=arrays))
    result = dict(cases=cases, totals=totals(cases), files_sha256=files,
        helper_generation_sha256=sha(HELPER / "generation_report.json"), writer_report_sha256=sha(WRITER / "report.json"),
        protocol_sha256=sha(PLAN), script_sha256=sha(__file__), tokenizer_table_sha256=table_hash,
        source_archive_declared_sha256=MODEL_SHA, source_archive_full_rehash_performed=False,
        source_weight_materialization=False, source_model_forward_calls=0, optimizer_updates=0,
        same_prefill_bf16_states=32, renderer=renderer.provenance,
        elapsed_seconds=time.monotonic() - started,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        training_cohort_only=True, language_quality_claim=False, general_ilm_goal_complete=False)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))


def verify(out):
    result = read(out / "result.json")
    for name, digest in result["files_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError("retained bytes changed: " + name)
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes, extra = list(visible), 256
    for byte in range(256):
        if byte not in visible:
            visible.append(byte); codes.append(extra); extra += 1
    inverse = {chr(code): byte for byte, code in zip(visible, codes)}
    checked = []
    for row in result["cases"]:
        reference = pixels(out / row["reference"])
        prefixes = [pixels(out / p["image"]) for p in row["canonical_prefixes"]]
        if ([p["text"] for p in row["canonical_prefixes"]]
                != [row["reference_text"][:k] for k in range(len(row["reference_text"]) + 1)]
                or any(p.shape != reference.shape for p in prefixes)):
            raise ValueError("canonical prefix labels or canvases differ")
        if not np.array_equal(prefixes[-1], reference) or np.any(prefixes[0] != 255):
            raise ValueError("complete/empty canonical prefix differs")
        with np.load(out / row["states"], allow_pickle=False) as arrays:
            source, writer = arrays["source"], arrays["writer"]
        expected_states = len(row["source_pieces"]) + int(row["source_stopped"])
        if (len(source) != expected_states or len(writer) != row["blocks"] + 1
                or len(row["source_prefixes"]) != min(len(row["source_pieces"]) + 1, len(source))
                or not np.isfinite(source).all() or not np.isfinite(writer).all()):
            raise ValueError("retained complete state sequences differ")
        def rounded_bf16_bits(a):
            raw = a.astype(np.float32).view(np.uint32)
            return ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
        initial_equal = bool(np.array_equal(rounded_bf16_bits(source[0]), rounded_bf16_bits(writer[0])))
        if not initial_equal or initial_equal != row["initial_bf16_state_equal"]:
            raise ValueError("initial state rounding does not reproduce the positive control")
        raw_text = bytes(inverse[c] for piece in row["source_pieces"] for c in piece)
        if raw_text.decode("utf8", errors="replace") != row["source_text"]:
            raise ValueError("complete retained helper response differs")
        decoded = []
        for k in range(len(row["source_prefixes"])):
            raw = bytes(inverse[c] for piece in row["source_pieces"][:k] for c in piece)
            try:
                text = raw.decode("utf8", errors="strict"); complete = True
            except UnicodeDecodeError:
                text = raw.decode("utf8", errors="replace"); complete = False
            if row["source_prefixes"][k] != dict(position=k, text=text, complete_utf8=complete):
                raise ValueError("independent byte-prefix decoding differs")
            decoded.append((k, text, complete))
        boundaries = []
        for b in row["boundaries"]:
            j, width = b["position"], b["columns"]
            canonical = [0] if j == 0 else [k for k, a in enumerate(prefixes)
                if not np.any(a[:, width:] != 255) and np.array_equal(a[:, :width], reference[:, :width])]
            matches = []
            for k, text, complete in decoded:
                if complete and any(text == row["canonical_prefixes"][p]["text"] for p in canonical):
                    a, t = writer[j].astype(np.float64), source[k].astype(np.float64)
                    a /= np.sqrt(np.dot(a, a) / len(a) + 1e-6)
                    t /= np.sqrt(np.dot(t, t) / len(t) + 1e-6)
                    matches.append(dict(source_position=k, text=text,
                        normalized_state_rmse=float(np.linalg.norm(a - t) / np.sqrt(len(a)))))
            if canonical != b["canonical_prefix_indices"] or len(matches) != len(b["matches"]):
                raise ValueError("independent physical correspondence differs")
            for a, t in zip(matches, b["matches"]):
                if a["source_position"] != t["source_position"] or a["text"] != t["text"] or not np.isclose(
                        a["normalized_state_rmse"], t["normalized_state_rmse"], rtol=1e-10, atol=1e-10):
                    raise ValueError("independent normalized state distance differs")
            boundaries.append(dict(b, matches=matches))
        checked.append(dict(row, boundaries=boundaries))
    if len(checked) != 32 or totals(checked) != result["totals"]:
        raise ValueError("complete correspondence totals differ")
    receipt = dict(cases=32, all_physical_correspondences_recomputed=True,
        native_byte_prefixes_reconstructed=True, matched_state_distances_recomputed=True,
        result_sha256=sha(out / "result.json"), model_forward_calls=0,
        renderer_reimplemented=False, contextual_models_replayed=False, general_ilm_goal_complete=False)
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    verify(args.output) if args.verify else run(args.output)


if __name__ == "__main__":
    main()
