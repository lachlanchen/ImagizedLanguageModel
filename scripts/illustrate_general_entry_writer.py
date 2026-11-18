"""Assemble measured dictionary responses, attested rasters and a fixed two-entry failure."""
from hashlib import sha256
import shutil

import numpy as np

from scripts.audit_general_entry_writer import EVIDENCE, archived
from scripts.ask_visual_dictionary import source_record
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import Rasterizer, load, write_json
from scripts.prepare_source_plate_questions import CACHE
from scripts.train_general_entry_writer import ENTRIES, WORK
from scripts.train_paired_passage import unpack_sequences
import torch


def main():
    report = load(EVIDENCE / "result.json")
    examples, dimensions = {}, {}
    raster = Rasterizer()
    record, _ = source_record("言", None)
    train = load(ENTRIES / "train/index.json")
    train_forms = {m["raster"] for m in train}
    cache = torch.load(CACHE, weights_only=True, map_location="cpu")["pixels"]
    for stage, field, value, term in (("bronze", "本义", "talk", "金文"), ("oracle", "读音", "yán", "甲骨文")):
        src = WORK / "yan" / stage
        receipt = load(src / "receipt.json")
        original = next(g for g in record["glyphs"] if g["label"] == next(
                        item["label"] for item in receipt["sources"] if item["stage"] == stage))
        assert original["raster_sha256"] not in train_forms
        raw = np.packbits(cache[original["raster_index"]].numpy())
        assert sha256(raw).hexdigest() == original["raster_sha256"]
        reference = b"".join(raster.tile(c) for c in "言的"+field+"是"+value+"，"+term+"作")
        reference += np.bitwise_not(raw).tobytes() + raster.tile("。")
        name = "yan_" + stage
        shutil.copyfile(src / "answer.png", EVIDENCE / (name + ".png"))
        dimensions[name] = list(page((src / "answer.bin").read_bytes()).size)
        examples[name] = dict(exact=receipt["stopped"] and (src/"answer.bin").read_bytes()==reference,
                              reference_sha256=sha256(reference).hexdigest(), receipt=receipt)
    assert examples["yan_bronze"]["receipt"]["source_sha256"] == examples["yan_oracle"]["receipt"]["source_sha256"]
    receipt = load(WORK / "yan/from_image/receipt.json")
    assert receipt["answer_sha256"] == examples["yan_bronze"]["receipt"]["answer_sha256"]
    folder = ENTRIES / "development"
    meta = load(folder / "index.json")
    bank = np.load(folder / "pixels.npy")
    expected = [bank[s].tobytes() for s in unpack_sequences(folder / "answers.npz")]
    images, end = archived(WORK / "paired/simple.npz", report["archives"]["paired/simple.npz"])
    fixed = load(EVIDENCE / "selection.json")
    located_images, located_stops = archived(WORK / "visible_selection/simple.npz", fixed["modes"]["simple"]["sha256"])
    for i, (x, e, y, m) in enumerate(zip(images, end, expected, meta, strict=True)):
        siblings = [j for j, alt in enumerate(meta) if alt["card"] == m["card"] and
                    alt["stage"] == m["stage"] and alt["head"] != m["head"]]
        if e and siblings and x[-256:-128] == expected[siblings[0]][-256:-128]:
            examples["failure"] = dict(row=i, requested=m, copied_form=meta[siblings[0]]["glyph"],
                                       copied_head=meta[siblings[0]]["head"])
            for name, stream in (("failed_output", x), ("located_output", located_images[i]), ("failed_reference", y)):
                im = page(stream)
                im.save(EVIDENCE / (name + ".png"))
                dimensions[name] = list(im.size)
            examples["failure"]["located_exact"] = bool(located_stops[i] and located_images[i] == y)
            break
    assert "failure" in examples
    table = [["Writer","Simple / 1,136","New wording / 96","English / 1,136","DRCD pairs / 1,042","DuReader / 571"]]
    for key, label in (("parent","Parent"),("scrambled","Shuffled"),("paired","Matched")):
        s = report["arms"][key]
        table.append([label, *(str(s[k]["exact"]) for k in ("simple","independent","english")),
                      str(s["drcd"]["both"]), str(s["dureader"]["exact"])])
    scores = fixed["modes"]
    assert not any(fixed["natural_inputs_changed"].values())
    table.append(["Located", *(str(scores[k]["exact"]) for k in ("simple","independent","english")),
                  str(report["arms"]["paired"]["drcd"]["both"]),str(report["arms"]["paired"]["dureader"]["exact"])])
    counts = load(EVIDENCE / "audit.json")["form_selection"]["paired/simple"]
    series = [[counts["same_head_right_stage"], counts["other_head_right_stage"],
               counts["outside_four_forms_or_unstopped"]]]
    write_json(EVIDENCE / "figure.json", dict(table=table, dimensions=dimensions, examples=examples, series=series,
               image_input_reproduced=True, yan_train_forms=sorted({m["glyph"] for m in train if m["head"]=="言"})))
    print(examples["failure"])


if __name__=="__main__":
    main()
