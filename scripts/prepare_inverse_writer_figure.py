"""Extract the fixed bronze examples and the first wrong-first-glyph natural example."""
from PIL import Image
from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_inverse_role_writer import challenge
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import Rasterizer
from scripts.train_inverse_role_writer import DRCD, PUBLIC, WORK, load, write_json


def prepare():
    result = load(WORK/"result.json")
    audit = load(PUBLIC/"audit.json")
    arms = result["arms"]
    render = Rasterizer()
    def drawn(text):
        return b"".join(render.tile(c) for c in text)
    def answers(test):
        return archived(WORK/"matched"/(test+".npz"),arms["matched"][test]["sha256"])[0]
    source = WORK/"matched"/"bronze0_premise.png"
    (PUBLIC/"bronze0_premise.png").write_bytes(source.read_bytes())
    for i,(_,query,_) in enumerate(challenge()["identify"][:2]):
        page(drawn(query)).save(PUBLIC/f"bronze_query{i}.png")
        page(answers("historic_identify")[i]).save(PUBLIC/f"bronze_answer{i}.png")
    record = audit["arms"]["matched"]["drcd"]["first_examples"]["wrong_first_glyph"]
    index = load(DRCD/"development/question_ids.json").index(record["id"])
    emitted = answers("drcd")[index]
    assert emitted == drawn(record["emitted"])
    fields = {"human_query":drawn(record["question"]), "human_answer":emitted,
              "human_reference":drawn(record["references"][0])}
    for name, image in fields.items():
        page(image).save(PUBLIC/(name+".png"))
    rows = [["Writer","Two names / 32","Four names / 96","New layout / 404","DuReader / 571"],
            ["Forward parent","0","0","297","0"]]
    for name in ("randomized","matched"):
        rows.append([name.capitalize()]+[str(arms[name][s]["exact"]) for s in
                       ("historic_identify","historic_four","new_layout_write","dureader")])
    controls = load(PUBLIC/"sentence_controls.json")["arms"]
    baseline = load(PUBLIC.parent/"natural_role_questions_20261119/baseline.json")["arms"]["passage_reader"]
    retrieved = load(PUBLIC/"evidence_sentences.json")["conditions"]["visual_overlap"]["exact"]
    singles = [arms["randomized"]["drcd"]["who"], arms["matched"]["drcd"]["who"], baseline["exact"]]
    scoped = [controls["randomized"]["exact"], retrieved, controls["passage_reader"]["exact"]]
    dimensions = {}
    for f in sorted(PUBLIC.glob("*.png")):
        with Image.open(f) as img: dimensions[f.stem] = img.size
    write_json(PUBLIC/"figure.json",dict(table=rows,series=[[round(100*x/n,2) for x in nums]
               for n,nums in ((387,singles),(387,scoped))],dimensions=dimensions,natural_id=record["id"]))


if __name__ == "__main__":
    prepare()
