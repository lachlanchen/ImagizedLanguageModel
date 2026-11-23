"""Test conditional interval scores on a reserved multilingual reading benchmark."""

from collections import defaultdict
import json
import torch
from safetensors.torch import load_file
from ilm.visual_lm.question_evidence_reader import predict
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead
from scripts.evaluate_bilingual_novel_writer import corpus
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.reserve_bilingual_interval_books import packets
from scripts.reserve_mlqa_pictures import CHECK, PACKET, PUBLIC, WORK
from scripts.train_bilingual_novel_writer import opposite
from scripts.train_question_evidence_head import bodies
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.train_span_choice import gpu_owner
from scripts.typeface_writer_cases import Styled


def mlqa():
    assert sha(PACKET) == json.loads(CHECK.read_text())["packet_sha"]
    face = Face(FACES[0])
    groups = dict(zh=[], en=[])
    distinct = []
    for entry in json.loads(PACKET.read_text()):
        p = face(entry["page"])
        qs = entry["questions"]
        pair = next(
            (
                (a["id"], b["id"])
                for i, a in enumerate(qs)
                for b in qs[i + 1 :]
                if opposite(a, b)
            ),
            None,
        )
        if pair:
            distinct.append(pair)
        for q in qs:
            refs = tuple(face(t) for t in q["refs"])
            for lang in groups:
                groups[lang].append(
                    Styled(qs[0]["id"], p, face(q[lang]), refs, refs, q["id"])
                )
    return groups, distinct


@torch.inference_mode()
def run():
    read, model = bodies()
    heads = {"parent": None}
    for arm in ("matched", "exchanged"):
        weights = WORK / (arm + ".safetensors")
        assert (
            sha(weights)
            == json.loads((PUBLIC / (arm + "_fit.json")).read_text())["sha256"]
        )
        head = QuestionEvidenceHead().cuda().eval()
        head.load_state_dict(load_file(str(weights)))
        heads[arm] = head
    official, pairs = mlqa()
    novels = corpus()
    other = packets()
    sets = {"mlqa": official, "book": {l: novels["book_" + l] for l in ("en", "zh")}}
    for key, gs in other.items():
        sets[key] = {l: gs["fresh_" + l] for l in ("en", "zh")}
    results = defaultdict(dict)
    for tag, groups in sets.items():
        counts = {a: dict() for a in heads}
        ids = [c.question_id for c in groups["zh"]]
        assert ids == [c.question_id for c in groups["en"]]
        paired = pairs if tag == "mlqa" else list(zip(ids[::2], ids[1::2], strict=True))
        for lang, rows in groups.items():
            out = {a: [] for a in heads}
            for i in range(0, len(rows), 48):
                streams = predict(
                    read, model, heads, [(r.p, r.q) for r in rows[i : i + 48]]
                )
                for a, ys in streams.items():
                    out[a].extend(ys)
            for a, ys in out.items():
                dest = WORK / f"{tag}_{lang}_{a}.npz"
                store_streams(dest, ys, [True] * len(ys))
                counts[a][lang] = [
                    bool(r.grade(y, True)[2]) for r, y in zip(rows, ys, strict=True)
                ]
                results[a].setdefault(tag, {})[lang] = dict(
                    exact=sum(counts[a][lang]), n=len(ys), sha256=sha(dest)
                )
        for a, flags in counts.items():
            by_id = dict(
                zip(
                    ids,
                    [z and e for z, e in zip(flags["zh"], flags["en"], strict=True)],
                    strict=True,
                )
            )
            results[a][tag]["four"] = dict(
                exact=sum(by_id[x] and by_id[y] for x, y in paired), n=len(paired)
            )
    m = results["matched"]["mlqa"]
    passed = (
        m["en"]["exact"]
        > max(results[k]["mlqa"]["en"]["exact"] for k in ("exchanged", "parent"))
        and m["zh"]["exact"] >= results["parent"]["mlqa"]["zh"]["exact"]
    )
    record(
        PUBLIC / "result.json",
        dict(scores=results, passed=passed, reservation_sha=sha(CHECK)),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
