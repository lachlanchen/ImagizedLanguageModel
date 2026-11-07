#!/usr/bin/env python3
"""Prepare and calibrate bilingual registers with facts swapped across pages."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import save_json, sha
from ilm.image_edit.data import font_and_coverage, page
from score_native_visual_access import canonical
from score_native_visual_alignment import edit_distance

SEED = 20270521
SCENARIOS = {
    "en": [
        ("amber sample", "blue sample", "North Pier", "Cedar Valley"),
        ("atlas", "botanical guide", "East Cabinet", "West Cabinet"),
        ("bronze case", "wooden case", "River Depot", "Hill Station"),
        ("red tent", "green tent", "North Meadow", "South Beach"),
        ("red lamp", "green lamp", "Upper Hall", "Lower Hall"),
        ("pear seedling", "plum seedling", "River Garden", "Stone Garden"),
        ("small boat", "large boat", "East Harbor", "West Harbor"),
        ("plain bowl", "painted bowl", "Cedar Kiln", "Willow Kiln"),
    ],
    "zh": [
        ("青铜铃", "石印", "东侧展厅", "西侧展厅"),
        ("地理图册", "诗词选集", "北边书架", "南边书架"),
        ("红色小船", "蓝色小船", "上游码头", "下游码头"),
        ("梨树苗", "梅树苗", "东园", "西园"),
        ("松木箱", "竹木箱", "松林", "竹林"),
        ("木箱", "铁箱", "三号仓库", "五号仓库"),
        ("第一包裹", "第二包裹", "河畔驿站", "山脚驿站"),
        ("素面陶碗", "彩绘陶碗", "柳窑", "松窑"),
    ],
}


def read(path):
    return json.loads(Path(path).read_text())


def material():
    rng = random.Random(SEED)
    result = []
    for language, cases in SCENARIOS.items():
        for objects_a, objects_b, place_a, place_b in cases:
            objects, places = [objects_a, objects_b], [place_a, place_b]
            orientation = rng.randrange(2)
            order = rng.sample([0, 1], 2)
            versions = []
            for version in range(2):
                values = places[::-1] if orientation ^ version else places[:]
                if language == "en":
                    records = [f"The {objects[i]} was kept at {values[i]} during the move." for i in order]
                    context = ("This is a fictional register. Use only the details printed here.\n"
                        "A clerk prepared this record before a collection was moved. The two entries refer to separate items, and their labels were checked twice.\n"
                        + "\n".join(records) + "\n"
                        "Both items were packed on the same morning. These locations were recorded before the carriers arrived. Later transport did not change the entries.")
                    questions = [f"Where was the {obj} kept during the move?" for obj in objects]
                    prompts = [context + "\nQuestion: " + q + "\nAnswer:" for q in questions]
                else:
                    records = [f"记录写明，搬运期间，{objects[i]}放在{values[i]}。" for i in order]
                    context = ("以下是一份虚构记录。仅依据本页文字作答。\n"
                        "管理员在搬运开始前整理了清单。下面两条记录指向不同的物品，工作人员已把每个标签核对两次。\n"
                        + "\n".join(records) + "\n"
                        "两件物品在同一天上午完成包装。清单中的地点是在搬运人员抵达前登记的，后来发生的运输没有改变这些记录。请根据物品名称查找相应记录，不要根据现实常识猜测地点。")
                    questions = [f"搬运期间，{obj}放在哪里？" for obj in objects]
                    prompts = [context + "\n问题：" + q + "\n回答：" for q in questions]
                versions.append(dict(context=context, questions=questions, prompts=prompts, expected=values))
            if Counter(versions[0]["context"]) != Counter(versions[1]["context"]):
                raise ValueError("swapping facts changed the character inventory")
            if versions[0]["questions"] != versions[1]["questions"]:
                raise ValueError("same questions required across versions")
            result.append(dict(scenario=f"counterfactual/{language}/{len(result):02d}", language=language,
                objects=objects, places=places, orientation=orientation, record_order=order, versions=versions))
    return result


def prepare(args):
    scenarios = material()
    font, provenance = font_and_coverage([dict(prompt=p, reference=v["expected"][q]) for s in scenarios for v in s["versions"] for q, p in enumerate(v["prompts"])])
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "coarse", "references"):
        (args.output / name).mkdir()
    inputs, references, transcripts, checks = [], [], [], []
    hashes = {}

    def store(image, row):
        i = len(inputs)
        normal = f"inputs/{i:03d}.png"
        low = f"coarse/{i:03d}.png"
        image.save(args.output / normal)
        image.resize((432, 432), Image.Resampling.BICUBIC).save(args.output / low)
        hashes[normal], hashes[low] = sha(args.output / normal), sha(args.output / low)
        inputs.append(dict(row, input=normal, input_sha256=hashes[normal], coarse_input=low, coarse_input_sha256=hashes[low]))

    for i, scenario in enumerate(scenarios):
        common = dict(scenario=scenario["scenario"], language=scenario["language"], seed=20270321 + i, readout_seed=20270421 + i)
        erased_passages = {0: [], 1: []}
        all_top = []
        for version, value in enumerate(scenario["versions"]):
            _, geometry = page(value["context"], (1024, 1024), font)
            top = 32 + 48 * geometry["lines"]
            all_top.append(top)
            erased_questions = []
            for question, prompt in enumerate(value["prompts"]):
                image, layout = page(prompt, (1024, 1024), font)
                array = np.asarray(image).copy()
                identifier = f"{scenario['scenario']}/v{version}/q{question}"
                store(image, dict(common, id=identifier, condition="normal", version=version, question=question))
                answer, _ = page(value["expected"][question], (512, 256), font)
                name = f"references/{len(references):03d}.png"
                answer.save(args.output / name)
                hashes[name] = sha(args.output / name)
                references.append(dict(id=identifier, **common, version=version, question=question,
                    expected=value["expected"][question], prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    reference_image=name, reference_sha256=hashes[name], geometry=layout))
                transcripts.append(dict(id=identifier, text=prompt))
                removed = array.copy()
                removed[top:] = 255
                erased_questions.append(removed)
                missing = array.copy()
                missing[:top] = 255
                erased_passages[question].append(missing)
            if not np.array_equal(*erased_questions):
                raise ValueError("question-erased pages differ within a passage version")
            identifier = f"{scenario['scenario']}/v{version}/question_erased"
            store(Image.fromarray(erased_questions[0]), dict(common, id=identifier, condition="question_erased", version=version))
            transcripts.append(dict(id=identifier, text=value["context"]))
        if all_top[0] != all_top[1]:
            raise ValueError("fact swap moved question band; preserve preparation failure")
        for question, arrays in erased_passages.items():
            if not np.array_equal(*arrays):
                raise ValueError("question pixels or erased passage differ across fact versions")
            identifier = f"{scenario['scenario']}/q{question}/passage_erased"
            store(Image.fromarray(arrays[0]), dict(common, id=identifier, condition="passage_erased", question=question))
            q = scenario["versions"][0]["questions"][question]
            text = "Question: " + q + "\nAnswer:" if scenario["language"] == "en" else "问题：" + q + "\n回答："
            transcripts.append(dict(id=identifier, text=text))
        checks.append(dict(scenario=scenario["scenario"], question_band_top=all_top[0], question_pixels_unchanged=True,
            shared_question_erased_equal=True, shared_passage_erased_equal=True, same_character_multiset=True,
            same_seeds_all_eight_conditions=True))
    if len(inputs) != 128 or len(references) != 64 or len(transcripts) != 128:
        raise ValueError("complete factorial cohort required")
    for name, data in (("inputs.json", inputs), ("references.json", references), ("visible_text.json", transcripts), ("scenarios.json", scenarios)):
        save_json(args.output / name, data)
    save_json(args.output / "preparation.json", dict(scenarios=16, inputs=128, normal=64,
        independent_question_erased=32, independent_passage_erased=32, seed=SEED,
        inputs_sha256=sha(args.output / "inputs.json"), references_sha256=sha(args.output / "references.json"),
        visible_text_sha256=sha(args.output / "visible_text.json"), scenarios_sha256=sha(args.output / "scenarios.json"),
        files_sha256=hashes, font=provenance, checks=checks, implementation_sha256=sha(__file__),
        synthetic_diagnostic=True, natural_qa_benchmark=False, incorporated_into_training=False,
        case_selection_uses_model_or_ocr=False, model_calls=0, optimizer_updates=0,
        general_ilm_goal_complete=False))
    print("Prepared 16 fictional scenarios, 64 normal pages and 64 shared erasures; no model calls.")


def calibrate(args):
    p = read(args.prepared / "preparation.json")
    if sha(args.prepared / "inputs.json") != p["inputs_sha256"] or sha(args.prepared / "references.json") != p["references_sha256"]:
        raise ValueError("prospective cohort changed")
    for name, digest in p["files_sha256"].items():
        if sha(args.prepared / name) != digest:
            raise ValueError("source/control/reference raster changed")
    rows = {r["id"]: r for r in read(args.prepared / "inputs.json") if r["condition"] == "normal"}
    references = read(args.prepared / "references.json")
    args.output.mkdir(parents=True, exist_ok=False)
    observations = []
    started = time.monotonic()
    for ref in references:
        for kind, path, expected in (("input", args.prepared / rows[ref["id"]]["input"], ref["prompt"]),
                                      ("reference", args.prepared / ref["reference_image"], ref["expected"])):
            result = subprocess.run(["tesseract", str(path), "stdout", "-l", "eng" if ref["language"] == "en" else "chi_sim+eng", "--psm", "6"],
                env=dict(os.environ, OMP_THREAD_LIMIT="1"), capture_output=True, text=True, check=True)
            actual, target = canonical(result.stdout, ref["language"]), canonical(expected, ref["language"])
            observations.append(dict(id=ref["id"], kind=kind, language=ref["language"], image_sha256=sha(path),
                text=result.stdout, expected=expected, canonical_exact=actual == target,
                edit_errors=edit_distance(actual, target), reference_characters=len(target)))
            save_json(args.output / "observations.json", observations)
    groups = {}
    for language in ("en", "zh"):
        for kind in ("input", "reference"):
            selected = [r for r in observations if r["language"] == language and r["kind"] == kind]
            groups[language + "/" + kind] = dict(cases=len(selected), exact=sum(r["canonical_exact"] for r in selected),
                character_edit_error=sum(r["edit_errors"] for r in selected) / sum(r["reference_characters"] for r in selected))
    save_json(args.output / "report.json", dict(groups=groups, observations_sha256=sha(args.output / "observations.json"),
        inputs_sha256=p["inputs_sha256"], references_sha256=p["references_sha256"],
        elapsed_seconds=time.monotonic() - started, cases_filtered_using_ocr=False,
        student_conditioning_uses_ocr=False, model_calls=0, general_ilm_goal_complete=False))
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "calibrate"), required=True)
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_counterfactual/prepared_v2"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    {"prepare": prepare, "calibrate": calibrate}[a.stage](a)
