"""Run the first improved and lost question pairs as physical PNGs on CPU."""

from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_competing_facts import PUB, MODEL, dump, read
from scripts.evaluate_competing_facts import dataset
from scripts.evaluate_visual_discourse import draw, unpack
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.write_paired_evidence import reader, generate
from scripts.write_visual_statements import editor, realize


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    cases = read(PUB / "comparison.json")["examples"]["contrast/parent"]
    rows, enc = dataset()
    folder = MODEL / "examples"
    folder.mkdir(parents=True, exist_ok=True)
    table = {(r["doc"], r["key"]): r for r in rows if r["test"] == "human"}
    sources = np.array([np.frombuffer(v, np.uint8) for v in enc.tiles])
    arms = ("parent", "likelihood", "contrast")
    frozen = {
        a: {
            (v["doc"], v["key"]): v
            for v in unpack(PUB / (a + "-outputs.json.gz"))
            if v["test"] == "human"
        }
        for a in arms
    }
    writer = Writer(device="cpu")
    statement = editor("cpu")
    nets = {a: reader("evidence" if a == "parent" else a, "cpu") for a in arms}
    space = next(i for i, v in enumerate(writer.palette) if i >= 2 and not v.any())
    receipts = []
    for kind in ("cmrc", "drcd", "squad", "coqa"):
        for reason in ("gain", "loss", "both"):
            if reason not in cases[kind]:
                continue
            doc = cases[kind][reason]
            for j in (0, 1):
                key = doc + "/" + str(j)
                r = table[doc, key]
                p = folder / kind / reason / str(j)
                p.mkdir(parents=True, exist_ok=True)
                draw(p, "question", sources[r["q"]])
                draw(p, "page", sources[r["s"]])
                q0 = read_image(p / "question.png", len(r["q"]))
                s0 = read_image(p / "page.png", len(r["s"]))
                ink, (q, s) = encode_pages(writer.palette, [q0, s0])
                x = torch.from_numpy(observations({"tiles": ink}, "ink"))
                out = {}
                for a in arms:
                    ((fact, stop),) = generate(
                        writer.model,
                        nets[a],
                        x,
                        len(writer.palette),
                        [q],
                        [s],
                        [s + q],
                        96,
                    )
                    wanted = frozen[a][doc, key]
                    bits = ink[np.asarray(fact, dtype=int)]
                    h = sha256(bits.tobytes()).hexdigest()
                    assert wanted["stopped"] == stop and wanted["physical"] == h, (
                        a,
                        key,
                    )
                    draw(p, a, bits)
                    out[a] = dict(fact=wanted["fact"], cells=len(fact), sha256=h)
                    if "answer_ink" in wanted:
                        ((ans, end),) = realize(
                            writer.model,
                            statement,
                            x,
                            len(writer.palette),
                            [q],
                            [fact],
                            space,
                        )
                        pixels = ink[np.asarray(ans, dtype=int)]
                        ah = sha256(pixels.tobytes()).hexdigest()
                        assert (
                            wanted["answer_ink"] == ah and end == wanted["answer_stop"]
                        ), (a, key, "statement")
                        draw(p, a + "-sentence", pixels)
                        out[a].update(
                            answer=wanted["answer"],
                            answer_cells=len(ans),
                            answer_sha256=ah,
                        )
                rec = dict(
                    kind=kind,
                    group=reason,
                    id=key,
                    question=r["question"],
                    query_cells=len(q),
                    source_cells=len(s),
                    refs=r["refs"],
                    outputs=out,
                )
                dump(p / "readout.json", rec)
                receipts.append(rec)
    dump(PUB / "physical-replays.json", receipts)


if __name__ == "__main__":
    main()
