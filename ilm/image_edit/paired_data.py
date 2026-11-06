"""Render complete source passages and separate answers for visual learning."""
import hashlib
import json
from pathlib import Path
import random

from .baseline import ROOT, save_json, sha
from .data import display_text, font_and_coverage, page

SEED = 20268320


def copy_cases():
    rng = random.Random(SEED)
    chinese = "天地人日月山水火木金土春夏秋冬风雨云雪花草树石江河海湖星光明白红青绿黑家门书画语言文字学习朋友东西南北大小多少长短高低早晚新旧左右上下中外心手口目耳足田禾牛羊鸟鱼"
    words = "river mountain garden morning evening bright quiet blue green small open close spring winter book window cloud stone paper silver".split()
    rows, seen = [], set()
    for split, count in (("train", 128), ("validation", 32)):
        for i in range(count):
            language = "zh" if i % 2 == 0 else "en"
            while True:
                if language == "zh":
                    text = "".join(rng.choices(chinese, k=rng.randint(4, 12)))
                    prompt = f"请只抄写引号内的内容：\n“{text}”"
                else:
                    text = " ".join(rng.choices(words, k=rng.randint(2, 4)))
                    prompt = f'Copy only the text in quotation marks:\n"{text}"'
                if text not in seen:
                    seen.add(text); break
            rows.append(dict(id=f"copy/{split}/{i:03}", split=split, task="copy", language=language,
                prompt=prompt, reference=text))
    return rows


def qa_candidates(source):
    groups = {}
    for line in Path(source).read_text().splitlines():
        row = json.loads(line)
        context = display_text(row["context"])
        digest = hashlib.sha256(context.encode()).hexdigest()
        groups.setdefault(digest, []).append(row)
    pairs = []
    for digest, group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r["id"])
        selected = []
        for row in group:
            reference = row["references"][0]["text"]
            if reference not in display_text(row["context"]) or reference in {r["reference"] for r in selected}:
                continue
            article = row["article_title"]
            split = "validation" if int(hashlib.sha256(article.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
            selected.append(dict(id=row["id"], split=split, task="qa", language="zh", article=article,
                pair=digest, prompt=f'资料：{display_text(row["context"])}\n问题：{display_text(row["question"])}\n回答：',
                reference=reference))
            if len(selected) == 2:
                pairs.append(selected); break
    random.Random(SEED + 1).shuffle(pairs)
    return pairs


def prepare(out, source=ROOT / "data/visual_qa/cmrc2018_short/train.jsonl"):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    copies = copy_cases()
    candidates = qa_candidates(source)
    all_rows = copies + [row for pair in candidates for row in pair]
    font, provenance = font_and_coverage(all_rows)
    chosen = list(copies)
    counts = dict(train=0, validation=0)
    rejected = []
    for pair in candidates:
        split = pair[0]["split"]
        if counts[split] >= dict(train=256, validation=32)[split]:
            continue
        try:
            for row in pair:
                page(row["prompt"], (1024, 1024), font)
                page(row["reference"], (512, 256), font)
        except ValueError as error:
            rejected.append(dict(pair=pair[0]["pair"], reason=str(error))); continue
        chosen.extend(pair); counts[split] += 2
    if counts != dict(train=256, validation=32):
        raise ValueError("not enough complete source pairs at the declared readable geometry")
    train_articles = {r["article"] for r in chosen if r["task"] == "qa" and r["split"] == "train"}
    val_articles = {r["article"] for r in chosen if r["task"] == "qa" and r["split"] == "validation"}
    if train_articles & val_articles:
        raise ValueError("article leakage")
    rendered = []
    for index, row in enumerate(chosen):
        result = {key: value for key, value in row.items() if key not in ("prompt", "reference")}
        result["reference_text"] = row["reference"]
        for key, text, size in (("input", row["prompt"], (1024, 1024)),
                                ("target", row["reference"], (512, 256))):
            image, geometry = page(text, size, font)
            name = f"{index:04}_{key}.png"; image.save(out / name)
            result[key] = name; result[key + "_sha256"] = sha(out / name)
            result[key + "_geometry"] = geometry
        rendered.append(result)
    (out / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rendered))
    report = dict(seed=SEED, source_sha256=sha(source), font=provenance, qa_counts=counts,
        copy_counts=dict(train=128, validation=32), total_pairs=len(rendered),
        input_canvas=[1024, 1024], answer_canvas=[512, 256], article_disjoint=True,
        full_passages=True, answer_aware_cropping=False, glyphs_clipped=False,
        geometry_rejections=rejected, manifest_sha256=sha(out / "pairs.jsonl"),
        validation_scope="held out from this optimizer; exploratory public data, not a pristine project test",
        quality_qualification=False, model_calls=0,
        implementation_sha256={name: sha(ROOT / name) for name in
            ("ilm/image_edit/paired_data.py", "ilm/image_edit/data.py")})
    save_json(out / "preparation.json", report)
    return report
