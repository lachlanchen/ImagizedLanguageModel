"""Audit complete statements on the separately authored passages, with arm identities hidden."""

from collections import Counter, defaultdict
import json
from time import monotonic

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.local_translation_teacher import local_teacher
from scripts.prepare_ordered_book_replies import PUBLIC, WORK
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_ordered_book_replies import grades, items
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

MODEL = "qwen3:8b-q8_0"
ANSWER = "请根据给出的材料，按提问顺序，用两个完整的中文短句回答两个问题，共不超过48字。只输出答案。"
JUDGE = """核对一个学生的中文回答。给出材料、两个问题及其参考事实。只有回答依次正确表达两个事实、每个事实的角色和关系都正确、没有重要错误或无意义重复时判为完整。只列两个词而没有表达关系不算完整。允许自然改写。只返回一个整数：2表示完整，1表示至少一个问题回答正确，0表示无正确答案。"""
ARMS = (
    "parent_plain",
    "shifted_shifted",
    "shifted_matched",
    "matched_shifted",
    "matched_matched",
)


def assess():
    rows = list(items("new", "matched"))
    result = json.loads((PUBLIC / "result.json").read_text())
    face = Face(FACES[0])
    letters = set("，。！？、是的了在由也和与为没有而")
    for row, c, lang, rev, facts, _ in rows:
        letters.update(c["page"] + "".join(c["clauses"]))
        for q in c["qs"]:
            letters.update(q["zh"] + q["en"])
    # Decode visible ink only for the external assessment. The writer never receives text.
    for split in ("train", "valid"):
        from scripts.prepare_ordered_book_replies import BOOK

        for line in (BOOK / f"{split}.jsonl").read_text().splitlines():
            c = json.loads(line)
            letters.update(c["page"])
            for q in c["questions"]:
                letters.update(q["zh"] + q["en"])
    mapping = defaultdict(set)
    for letter in sorted(letters):
        if letter.isspace() or ord(letter) in face.coverage:
            mapping[face(letter)].add(" " if letter.isspace() else letter)
    streams = {}
    for arm in ARMS:
        p = WORK / f"new_{arm}.npz"
        assert sha(p) == result["scores"][arm]["new"]["streams_sha"]
        streams[arm] = load_streams(p)
    scores = defaultdict(Counter)
    report = []
    start = monotonic()
    with local_teacher(MODEL, WORK) as (ask, digest, size):
        for i, (_, c, lang, rev, facts, _) in enumerate(rows):
            order = [1, 0] if rev else [0, 1]
            qs = [c["qs"][k][lang] for k in order]
            prompt = c["page"] + "\n1. " + qs[0] + "\n2. " + qs[1]

            def run(system, user, limit):
                return ask(
                    system,
                    dict(role="user", content=user),
                    think=False,
                    options=dict(
                        temperature=0, seed=241209, num_ctx=2048, num_predict=limit
                    ),
                )

            text = run(ANSWER, prompt, 128)
            qwen = text["text"]
            ref = "、".join(c["qs"][k]["refs"][0] for k in order)
            seen = {}
            entries = {}
            for arm in (*ARMS, "text_qwen"):
                unknown = 0
                if arm == "text_qwen":
                    value = qwen
                    end = text["stop"] == "stop" and 0 < len(value) <= 48
                    try:
                        ordered = grades(face(value), end, facts, rev)
                    except ValueError:
                        ordered = False
                else:
                    ys, ends = streams[arm]
                    cells = in_cells(ys[i])
                    unknown = sum(len(mapping[g]) != 1 for g in cells)
                    value = "".join(
                        next(iter(mapping[g])) if len(mapping[g]) == 1 else "□"
                        for g in cells
                    )
                    end = bool(ends[i])
                    ordered = grades(ys[i], end, facts, rev)
                if value and end and not unknown:
                    if value not in seen:
                        v = run(
                            JUDGE,
                            prompt + "\n参考事实：" + ref + "\n学生回答：" + value,
                            8,
                        )
                        seen[value] = (
                            int(v["text"]) if v["text"] in ("0", "1", "2") else None
                        )
                    judge = seen[value]
                else:
                    judge = None
                key = scores[arm]
                key["requests"] += 1
                key["stopped_in_budget"] += int(end)
                key["ordered"] += int(ordered)
                key["complete_model_grade"] += int(judge == 2)
                key["partial_model_grade"] += int(judge == 1)
                key["ungraded"] += int(judge is None)
                entries[arm] = dict(
                    text=value, unknown=unknown, end=end, ordered=ordered, judge=judge
                )
            report.append(dict(key=c["key"], lang=lang, reverse=rev, entries=entries))
    record(WORK / "statement_assessment.json", report)
    record(
        PUBLIC / "statement_assessment.json",
        dict(
            counts=scores,
            model=MODEL,
            model_sha=digest,
            model_bytes=size,
            seconds=monotonic() - start,
            rows_sha=sha(WORK / "statement_assessment.json"),
            judge=JUDGE,
            limitations="A single text-model assessment, including self-assessment of the text baseline. Unstopped, uncertain and invalid grades receive no complete credit.",
        ),
    )
    # Ship the first page and the actual first forward and reverse Chinese replies.
    examples = PUBLIC / "examples"
    c = rows[0][1]
    for name, text in (
        ("page", c["page"]),
        ("first", c["qs"][0]["zh"]),
        ("second", c["qs"][1]["zh"]),
    ):
        write_strip(tensor(face(text)).numpy(), examples / (name + ".png"))


if __name__ == "__main__":
    with gpu_owner():
        assess()
