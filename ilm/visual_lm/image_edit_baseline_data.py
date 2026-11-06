"""Prepare readable fixed Chinese/English question and reference pages on CPU."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from .image_edit_baseline import ROOT, save_json, sha

TRAIN = ROOT / "data/visual_qa/cmrc2018_short/train.jsonl"
TRAIN_SHA = "742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c"
COHORT = ROOT / "artifacts/native_chinese_v80_20261106/corpus_fit_v80r2/cases.json"
COPY_SOURCE = ROOT / "scripts/prepare_copy_transfer_v75.py"


def fixed_cases():
    tree = ast.parse(COPY_SOURCE.read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets)]
    if len(values) != 1:
        raise ValueError("original fixed copy cases missing")
    copies = ast.literal_eval(values[0])
    if len(copies) != 12 or [row[0] for row in copies] != ["zh"] * 4 + ["en"] * 4 + ["numeric"] * 4:
        raise ValueError("original copy case order differs")
    cases = [dict(position=i, id=f"copy/{i:02}", pair=f"copy/{i//2}", language=language,
        prompt=prompt, reference=answer) for i, (language, prompt, answer) in enumerate(copies)]
    if sha(TRAIN) != TRAIN_SHA:
        raise ValueError("original Chinese question data changed")
    cohort = json.loads(COHORT.read_text())
    if len(cohort) != 32 or any(row["position"] != i for i, row in enumerate(cohort)):
        raise ValueError("original fixed training cohort differs")
    selected = cohort[:8]
    if any(not row["source"].startswith("cmrc/") or not row["source"].endswith(".pt") for row in selected):
        raise ValueError("the selected original questions must be CMRC training records")
    if any(selected[i]["context_sha256"] != selected[i + 1]["context_sha256"] for i in range(0, 8, 2)):
        raise ValueError("original same-passage question pairs differ")
    wanted = {row["source"].removeprefix("cmrc/").removesuffix(".pt") for row in selected}
    found = {}
    with TRAIN.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                if row["id"] in found:
                    raise ValueError("duplicate source question ID")
                found[row["id"]] = row
    if set(found) != wanted:
        raise ValueError("owning original question not found")
    for i, item in enumerate(selected):
        name = item["source"].removeprefix("cmrc/").removesuffix(".pt")
        row = found[name]
        cases.append(dict(position=12 + i, id=name, pair=f"qa/{i//2}", language="zh",
            prompt=row["prompt"], reference=row["references"][0]["text"]))
    if any(not row["prompt"].strip() or not row["reference"].strip() for row in cases):
        raise ValueError("nonempty questions and expected answers required")
    return cases


def font_and_coverage(cases):
    description = subprocess.check_output(["fc-match", "-f", "%{file}\n%{index}\n%{charset}\n",
        "Noto Sans CJK SC"], text=True).splitlines()
    if len(description) != 3:
        raise ValueError("one declared CJK font face required")
    file, index, charset = description
    intervals = []
    for group in charset.split():
        bits = group.split("-")
        intervals.append((int(bits[0], 16), int(bits[-1], 16)))
    required = {char for row in cases for key in ("prompt", "reference") for char in row[key] if char != "\n"}
    missing = [char for char in sorted(required) if not any(lo <= ord(char) <= hi for lo, hi in intervals)]
    if missing:
        raise ValueError("source font lacks required characters: " + repr(missing))
    font = ImageFont.truetype(file, size=32, index=int(index))
    return font, dict(font_file_sha256=sha(file), font_face_index=int(index), font_size=32,
        charset_sha256=hashlib.sha256(charset.encode()).hexdigest(), missing_characters=0,
        unique_source_characters=len(required))


def page(text, size, font):
    """Wrap by physical advance, preserve all non-newline characters and ink."""
    width, height = size
    margin, pitch = 32, 48
    if "\r" in text or "\t" in text:
        raise ValueError("declare normalization before rendering unsupported control whitespace")
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            if line and font.getlength(line + char) > width - 2 * margin:
                lines.append(line); line = ""
            line += char
        lines.append(line)
    if "".join(lines) != text.replace("\n", ""):
        raise ValueError("line wrapping discarded source characters")
    image = Image.new("RGB", size, "white"); draw = ImageDraw.Draw(image)
    boxes = []
    for i, line in enumerate(lines):
        at = (margin, margin + i * pitch)
        box = draw.textbbox(at, line, font=font, anchor="lt")
        if box[0] < 0 or box[1] < 0 or box[2] > width - margin or box[3] > height - margin:
            raise ValueError("source page would clip writing")
        draw.text(at, line, font=font, anchor="lt", fill="black")
        boxes.append(list(box))
    if image.convert("L").getextrema() == (255, 255):
        raise ValueError("nonempty text rendered no ink")
    return image, dict(lines=len(lines), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        non_newline_characters=len(text.replace("\n", "")), boxes=boxes,
        all_characters_retained=True, writing_clipped=False)


def prepare_pages(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    cases = fixed_cases(); font, provenance = font_and_coverage(cases)
    rows, files = [], {}
    for case in cases:
        i = case["position"]
        prompt, geometry = page(case["prompt"], (1024, 1024), font)
        reference, reference_geometry = page(case["reference"], (512, 512), font)
        names = dict(normal=f"{i:02}_normal.png", blank=f"{i:02}_blank.png", reference=f"{i:02}_reference.png")
        prompt.save(out / names["normal"])
        Image.new("RGB", prompt.size, "white").save(out / names["blank"])
        reference.save(out / names["reference"])
        for name in names.values():
            files[name] = sha(out / name)
        rows.append(dict(position=i, id=case["id"], pair=case["pair"], language=case["language"],
            seed=20268301 + i // 2, files=names, prompt_geometry=geometry, reference_geometry=reference_geometry))
    report = dict(cases=rows, files_sha256=files, rendering=provenance,
        source_data_sha256={str(path.relative_to(ROOT)): sha(path) for path in (TRAIN, COHORT, COPY_SOURCE)},
        prompts=20, normal_control_calls_planned=40, model_calls=0, weights_loaded=False,
        original_case_order_preserved=True, question_pages=[1024, 1024], answer_pages=[512, 512],
        references_kept_separate=True, reused_exploratory_cases=True,
        sealed_test_used=False, language_quality_claim=False)
    save_json(out / "pages.json", report)
    return report
