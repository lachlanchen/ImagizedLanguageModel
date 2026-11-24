"""Archive both unedited replies from the first Chinese card yielding a correct historical mark."""

from hashlib import sha256
import torch
from scripts.evaluate_recurrent_history import historical_pages
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_open_exam import MODEL, PUB, read, dump
from scripts.write_open_exam import infer, load
from scripts.write_visual_discourse import Writer


def main():
    torch.set_num_threads(3)
    rows, pages, refs = historical_pages()
    saved = read(PUB / "matched-history.json")
    index = next(i for i, r in enumerate(saved) if r["lang"] == "zh" and r["correct"])
    start = index - index % 2
    writer, net = Writer(), load("matched", "cpu")
    for i in (start, start + 1):
        q, p = pages[2 * i : 2 * i + 2]
        out, stop = infer(writer, net, q, p)
        assert (
            sha256(out.tobytes()).hexdigest() == saved[i]["sha256"]
            and stop == saved[i]["stopped"]
        )
        dest = MODEL / "examples/first-history" / str(i - start + 1)
        dest.mkdir(parents=True, exist_ok=True)
        for k, data in (
            ("question", q),
            ("passage", p),
            ("reference", refs[i]),
            ("answer", out),
        ):
            draw(dest, k, data)
        dump(
            dest / "receipt.json",
            dict(
                saved[i],
                word=rows[i]["headword"],
                question_cells=len(q),
                page_cells=len(p),
            ),
        )


if __name__ == "__main__":
    main()
