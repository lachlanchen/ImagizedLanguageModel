#!/usr/bin/env python3
"""Package every actual answer for transcription without target/condition labels.

This is a review aid, not an independent judge or a model-quality score.
Original PNGs are copied verbatim. Contact sheets only add external labels.
The source mapping stays outside the reviewer packet.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", action="append", required=True, help="unique-label=/path/to/actual/outputs.json")
    p.add_argument("--seed", type=int, default=20269001)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("hide CUDA for this offline evidence preparation")
    from PIL import Image, ImageDraw, ImageFont
    from scripts.export_balanced_autonomous_writer import tensor_png
    rows, groups = [], {}
    for specification in args.index:
        label, separator, name = specification.partition("=")
        if not separator or not label or label in groups:
            raise ValueError("unique nonempty group label and input index required")
        path = Path(name).resolve(); actual = json.loads(path.read_text())
        if not actual or len({r["id"] for r in actual}) != len(actual):
            raise ValueError("complete unique nonempty source-output index required")
        groups[label] = dict(index=str(path), index_sha256=sha(path), count=len(actual))
        for i, row in enumerate(actual):
            answer = path.parent / row["path"]; evidence = row["result"]
            if sha(answer) != row["sha256"] or row["sha256"] != evidence["png_sha256"]:
                raise ValueError("actual source answer bytes differ")
            tensor_png(answer.parent / "pixels.pt", answer, evidence["pixels_sha256"], row["sha256"])
            with Image.open(answer) as image:
                if image.mode != "RGB" or image.size != (512, 256):
                    raise ValueError("this review layout requires complete native RGB answer pages")
                if image.info:
                    raise ValueError("source metadata requires inspection before blind packaging")
            rows.append(dict(group=label, source_position=i, source_id=row["id"],
                source_answer=str(answer), answer_sha256=row["sha256"],
                actual_tensor_sha256=evidence["pixels_sha256"], seed=row["seed"],
                input_sha256=row["input_sha256"]))
    random.Random(args.seed).shuffle(rows)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    packet = output / "packet"; (packet / "images").mkdir(parents=True); (packet / "pages").mkdir()
    public, key = [], []
    for i, row in enumerate(rows, 1):
        review_id = f"R{i:04}"
        name = "images/" + review_id + ".png"
        shutil.copy2(row["source_answer"], packet / name)
        if sha(packet / name) != row["answer_sha256"]:
            raise ValueError("review copy changed actual answer bytes")
        public.append(dict(review_id=review_id, image=name, image_sha256=row["answer_sha256"], size=[512, 256]))
        key.append(dict(review_id=review_id, **row))
    save_json(packet / "index.json", public)
    with (packet / "annotations.jsonl").open("x") as handle:
        for row in public:
            handle.write(json.dumps(dict(review_id=row["review_id"], transcription=None,
                legibility="unreviewed", visible_line_count=None, uncertain_regions=[], notes="")) + "\n")
    (packet / "READ_FIRST.txt").write_text(
        "Transcribe only the visible writing. Mark uncertain regions explicitly; do not infer a missing word.\n"
        "Do not open the private key, source files, target answers or condition labels during this pass.\n"
        "Every image is a complete actual model page. Contact sheets add only external review IDs.\n"
        "Do not repair, crop, replace, or omit extra writing. Leave unknown characters uncertain.\n"
        "This packet reduces label cues; it does not establish an independent or perfectly blinded review.\n")
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    contacts = []
    for offset in range(0, len(public), 16):
        selected = public[offset:offset+16]; columns = 4
        page = Image.new("RGB", (columns*512, ((len(selected)+columns-1)//columns)*280), "white")
        draw = ImageDraw.Draw(page)
        for i, row in enumerate(selected):
            x, y = (i % columns)*512, (i//columns)*280
            draw.rectangle((x,y,x+511,y+23), fill="#e5e9ed")
            draw.text((x+8,y+3),row["review_id"],font=font,fill="black")
            with Image.open(packet / row["image"]) as image:
                page.paste(image,(x,y+24))
                pasted = page.crop((x,y+24,x+512,y+280))
                if pasted.tobytes() != image.tobytes():
                    raise ValueError("contact sheet changed original answer pixels")
        name = f"pages/page_{1+offset//16:03}.png"; page.save(packet/name); contacts.append(name)
    save_json(output / "private_key.json", dict(groups=groups, seed=args.seed, cases=key))
    packet_files = {str(p.relative_to(packet)):sha(p) for p in sorted(packet.rglob("*")) if p.is_file()}
    result = dict(cases=len(rows), source_groups=len(groups), contact_sheets=len(contacts), seed=args.seed,
        implementation_sha256=sha(__file__), tensor_authenticator_sha256=sha(ROOT / "scripts/export_balanced_autonomous_writer.py"),
        private_key_sha256=sha(output / "private_key.json"), packet_files_sha256=packet_files,
        all_source_pngs_copied_verbatim=True, all_contact_sheet_answer_regions_pixel_identical=True,
        targets_or_condition_labels_in_packet=False, expected_answers_loaded=False,
        source_ids_and_group_names_in_private_key=True, preparer_has_source_access=True,
        independent_or_perfect_blinding_claim=False, raw_annotations_completed=False,
        quality_score_computed=False, neural_model_loaded=False, general_ilm_goal_complete=False)
    save_json(output / "receipt.json", result)
    print(json.dumps({k:result[k] for k in ("cases","source_groups","contact_sheets","all_source_pngs_copied_verbatim","all_contact_sheet_answer_regions_pixel_identical")},indent=2))


if __name__ == "__main__": main()
