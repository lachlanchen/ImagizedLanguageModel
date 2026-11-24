"""Answer a pictured question from continuous Chinese prose about one supplied venue."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.prose_fields import select, name_position, pictured_name
from scripts.prepare_chinese_card_reply import DOMAINS, FIELDS, Rasterizer
from scripts.prepare_prose_fields import WIDTH, LIMIT
from scripts.write_chinese_card_reply import pack, card_image, display
from scripts.write_chinese_two_fact import TwoFactReply, keys_for, expand_plan
from scripts.train_prose_fields import load_model
from scripts.train_field_alignment import load_alignment
from scripts.train_field_cursor import cursor_actions, convert_plans

SCHEMA = {
    "酒店": ("地址", "评分", "酒店类型", "价格"),
    "景点": ("地址", "评分", "游玩时间", "门票"),
    "餐馆": ("地址", "评分", "人均消费", "营业时间"),
}


def render(q, domain, name, text, raster):
    if len(text) > LIMIT or not text or not name.strip():
        raise ValueError("Expected a named venue and one to 160 passage cells")
    return np.stack(
        [
            pack(s, raster)
            for s in [
                q,
                domain,
                name,
                *[text[i : i + WIDTH] for i in range(0, len(text), WIDTH)],
            ]
        ]
    )


def load_png(path):
    with Image.open(path) as image:
        if image.width != WIDTH * 32 or image.height not in (128, 160):
            raise ValueError(
                "Expected an 80-column PNG with question, domain, name and one or two prose rows"
            )
        grey = Image.alpha_composite(
            Image.new("RGBA", image.size, "white"), image.convert("RGBA")
        ).convert("L")
        cells = (
            (np.asarray(grey) < 128).reshape(-1, 32, WIDTH, 32).transpose(0, 2, 1, 3)
        )
        return np.packbits(cells.reshape(-1, WIDTH, 1024), axis=-1)


def trim(cells):
    ink = np.flatnonzero(cells.any(1))
    if not len(ink):
        raise ValueError("Empty line")
    return cells[: ink[-1] + 1]


class ProseReply:
    def __init__(
        self, device="cpu", mode="ink", model_dir=None, receipts=None, arm=None
    ):
        modes = {
            None: ("ink", "byte"),
            "entity-ink": ("ink",),
            "entity-erased": ("ink",),
            "entity-byte": ("byte",),
        }
        if mode not in modes.get(arm, ()):
            raise ValueError(
                "The selected input representation must match the fitted reader"
            )
        if (model_dir is None) != (receipts is None):
            raise ValueError(
                "Supply both the weights directory and its verification receipts"
            )
        self.writer = TwoFactReply(device)
        self.mode = mode
        self.device = device
        self.arm = arm
        paths = () if model_dir is None else (model_dir, receipts)
        if arm is None:
            self.segment = load_model(mode, device, *paths)
        else:
            from scripts.train_prose_entity import model

            self.segment = model(arm, device, *paths)
        self.plan = (
            self.writer.aligner if mode == "ink" else load_alignment("byte", device)
        )

    @torch.inference_mode()
    def encode(self, pages, raw=None):
        if not pages:
            raise ValueError("No pages")
        dom = []
        names = []
        lengths = []
        texts = []
        for p in pages:
            if (
                p.dtype != np.uint8
                or p.ndim != 3
                or p.shape[0] not in (4, 5)
                or p.shape[1:] != (WIDTH, 128)
            ):
                raise ValueError("Expected packed cells in the 80-column page layout")
            if not p[0].any():
                raise ValueError("Empty question")
            idx = self.writer.domains.get(p[1].tobytes())
            if idx is None:
                raise ValueError("Unrecognized pictured domain")
            dom.append(idx)
            names.append(trim(p[2]))
            text = trim(p[3:].reshape(-1, 128))
            texts.append(text)
            lengths.append(len(text))
        if self.mode == "ink":
            q = np.stack([p[0] for p in pages])
            x = np.stack(
                [np.pad(t, ((0, max(lengths) - len(t)), (0, 0))) for t in texts]
            )

            def observe(x):
                return observations({"tiles": x.reshape(-1, 128)}, "ink").reshape(
                    *x.shape[:2], 256
                )
        else:
            if raw is None or len(raw) != len(pages):
                raise ValueError(
                    "The byte comparator needs its original transcriptions"
                )

            def bytes_rows(lines, n):
                buf = np.zeros((len(lines), n, 4), np.uint8)
                for i, line in enumerate(lines):
                    for j, c in enumerate(line):
                        if j < n:
                            buf[i, j, : len(c.encode())] = list(c.encode())
                return buf

            q = bytes_rows([r["q"] for r in raw], WIDTH)
            x = bytes_rows([r["text"] for r in raw], max(lengths))

            def observe(x):
                return observations({"bytes": x.reshape(-1, 4)}, "byte").reshape(
                    *x.shape[:2], 256
                )

        def to(x):
            return torch.as_tensor(x, device=self.device)

        return dom, names, texts, to(dom), to(lengths), to(observe(q)), to(observe(x))

    @torch.inference_mode()
    def batch(self, pages, raw=None, *, erase=False):
        dom, names, texts, domains, lengths, q, x = self.encode(pages, raw)
        schema = torch.tensor(
            [[f in SCHEMA[list(DOMAINS)[d]] for f in FIELDS] for d in dom],
            device=self.device,
        )
        if self.arm:
            anchor = np.zeros((len(pages), x.size(1), 3), np.float32)
            for i, (text, name) in enumerate(zip(texts, names)):
                if self.mode == "byte":
                    string, needle = raw[i]["text"], raw[i]["name"]
                    start = string.index(needle) if string.count(needle) == 1 else None
                else:
                    start = pictured_name(text, name)
                if self.arm != "entity-erased":
                    anchor[i, : len(text)] = name_position(len(text), start, len(name))
            outputs = self.segment(
                x, lengths, domains, torch.as_tensor(anchor, device=self.device)
            )
        else:
            outputs = self.segment(x, lengths, domains)
        spans = select(outputs, lengths).tolist()
        plans = self.plan.choose(torch.zeros_like(q) if erase else q, domains, schema)
        w = self.writer
        seq = w.cursor.write(
            w.labels,
            domains + len(FIELDS),
            convert_plans(plans, len(w.labels)),
            w.embedding,
        )
        actions = cursor_actions(seq, w.alphabet)
        results = []
        for i, (plan, steps) in enumerate(zip(plans.tolist(), actions)):
            keys = keys_for(plan)
            intervals = {k: (a, b) for k, (a, b) in zip(FIELDS, spans[i]) if a >= 0}
            values = {k: texts[i][a:b] for k, (a, b) in intervals.items()}
            values["名称"] = names[i]
            missing = [k for k in keys if k not in intervals]
            pixels = (
                None
                if missing
                else expand_plan(steps, w.alphabet, w.images, keys, values)
            )
            results.append(
                dict(
                    keys=keys,
                    intervals=intervals,
                    missing=missing,
                    pixels=pixels,
                    actions=steps,
                )
            )
        return results


def main():
    import json

    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument("--input", type=Path)
    s.add_argument("--record", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--single-venue",
        action="store_true",
        help="Use the original reader fitted on one venue per passage",
    )
    args = p.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(2)
    if args.record:
        r = json.loads(args.record.read_text())
        page = render(r["q"], r["domain"], r["name"], r["text"], Rasterizer())
    else:
        page = load_png(args.input)
    model = ProseReply(args.device, arm=None if args.single_venue else "entity-ink")
    item = model.batch([page])[0]
    raw = item.pop("pixels")
    item.pop("actions")
    args.output.mkdir(parents=True)
    card_image(page).save(args.output / "input.png")
    if raw is not None:
        display(raw).save(args.output / "answer.png")
        (args.output / "answer.bin").write_bytes(raw.tobytes())
        item["transcript"] = model.writer.transcript(raw)
    (args.output / "result.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2) + "\n"
    )
    print(item.get("transcript") or "资料不足：" + "、".join(item["missing"]))


if __name__ == "__main__":
    main()
