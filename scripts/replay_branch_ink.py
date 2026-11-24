"""Replay newly authored role exchanges using the released writer and physical PNG inputs."""

from hashlib import sha256
import numpy as np
import torch
from scripts.prepare_branch_ink import MODEL, PUB, EXAM
from scripts.prepare_open_exam import dump
from scripts.evaluate_visual_discourse import draw
from scripts.write_visual_discourse import Writer
from scripts.write_open_exam import infer, load
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT

# Two independent questions on each page. The page edit exchanges the two facts.
CASES = [
    (
        "depart",
        "早报说，小陈乘车去{}，小林乘车去{}。两个人在车站告别。",
        "北京",
        "广州",
        "小陈要去哪里？",
        "小林要去哪里？",
    ),
    (
        "shelf",
        "闭馆记录写着，铜镜放在{}，石碑拓片放在{}。",
        "东柜",
        "西柜",
        "铜镜放在哪个柜子？",
        "石碑拓片放在哪个柜子？",
    ),
    (
        "poem",
        "课上读了两篇作品。《静夜思》的作者是{}，《春望》的作者是{}。",
        "李白",
        "杜甫",
        "《静夜思》的作者是谁？",
        "《春望》的作者是谁？",
    ),
    (
        "gloss",
        "讲义分别解释两个字，言的意思是{}，水的意思是{}。",
        "说话",
        "河流",
        "言是什么意思？",
        "水是什么意思？",
    ),
    (
        "room",
        "The notice lists the {} as the venue for the lecture and the {} as the venue for the concert.",
        "library",
        "garden",
        "Where is the lecture?",
        "Where is the concert?",
    ),
    (
        "loan",
        "Mara borrowed the {}. Jo borrowed the {}. Both books are due next week.",
        "atlas",
        "dictionary",
        "What did Mara borrow?",
        "What did Jo borrow?",
    ),
    (
        "work",
        "The notes credit {} with the drawing and {} with the engraving.",
        "Ellen",
        "Peter",
        "Who made the drawing?",
        "Who did the engraving?",
    ),
    (
        "hour",
        "The morning train leaves at {}. The evening train leaves at {}.",
        "nine",
        "seven",
        "At what hour does the morning train leave?",
        "At what hour does the evening train leave?",
    ),
]


def main():
    writer = Writer()
    records = []
    for arm, root in (("matched", EXAM), ("branch", MODEL)):
        net = load(arm, "cpu", root)
        for name, template, a, b, q1, q2 in CASES:
            for v, (u, w) in enumerate(((a, b), (b, a))):
                passage = template.format(u, w)
                for j, (question, answer) in enumerate(((q1, u), (q2, w))):
                    d = MODEL / "examples" / name / str(v) / str(j)
                    d.mkdir(parents=True, exist_ok=True)
                    inputs = []
                    for k, s in (
                        ("question", question),
                        ("passage", passage),
                        ("reference", answer),
                    ):
                        ink = read_text(s, DEFAULT_FONT)
                        if arm == "matched":
                            draw(d, k, ink)
                        actual = read_image(d / (k + ".png"), len(ink))
                        assert np.array_equal(ink, actual)
                        inputs.append(actual)
                    q, p, target = inputs
                    pix, end = infer(writer, net, q, p)
                    draw(d, arm, pix)
                    alphabet = {
                        bytes(t): c for t, c in zip(writer.palette, writer.transcript)
                    }
                    for tile, ch in zip(
                        np.concatenate([p, q]), passage + question, strict=True
                    ):
                        alphabet.setdefault(bytes(tile), ch)
                    r = dict(
                        arm=arm,
                        case=name,
                        version=v,
                        query=j,
                        question=question,
                        page=passage,
                        expected=answer,
                        cells=len(pix),
                        stopped=end,
                        actual="".join(alphabet.get(bytes(t), "�") for t in pix),
                        exact=end and np.array_equal(pix, target),
                        sha256=sha256(pix.tobytes()).hexdigest(),
                    )
                    records.append(r)
    dump(PUB / "authored.json", records)


if __name__ == "__main__":
    torch.set_num_threads(3)
    main()
