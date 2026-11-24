"""Diagnose count errors and measure transfer from selected facts to English sentences."""

from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import best_path
from scripts.evaluate_multiple_ink_runs import corpus, grade
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.prepare_multiple_ink_runs import WORK, PUB, read, dump, normalize
from scripts.train_multiple_ink_runs import load
from scripts.train_paired_evidence import gpu_owner
from scripts.write_multiple_ink_runs import forward, sentences
from scripts.write_visual_discourse import Writer
from scripts.write_visual_statements import editor
from scripts.evaluate_competing_facts import CorpusFace


@torch.inference_mode()
def main():
    rows, base_tiles = corpus()
    confirm = read(WORK / "confirm.json")
    tiles = np.concatenate((base_tiles, np.load(WORK / "confirm-tiles.npy")))
    src = {r["id"]: r for r in rows + confirm}
    saved = {}
    for prefix in ("survival-", "survival-confirm-"):
        for r in unpack(PUB / (prefix + "answers.json.gz")):
            if r["mode"] == "count" and r["condition"] == "original":
                saved[r["id"]] = r
    diagnoses = []
    utterances = []
    with gpu_owner():
        net = load("cuda", True)
        bank = net.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda())
        visible = [
            r
            for r in src.values()
            if 2 <= len(r["refs"][0]) <= 16 and all(w in r["p"] for w in r["refs"][0])
        ]
        for i in range(0, len(visible), 12):
            g = visible[i : i + 12]
            roles, cs = forward(
                net, bank, [r["qi"] for r in g], [r["pi"] for r in g], survival=True
            )
            for j, r in enumerate(g):
                k = len(r["refs"][0])
                util = np.rint(
                    1000 * roles[j, : len(r["p"])].astype(np.float64)
                ).astype(np.int64)
                reward = np.zeros(k + 1, np.int64)
                reward[k] = 1 << 45
                tags, _ = best_path(util, reward)
                parts = []
                for ch, t in zip(r["p"], tags, strict=True):
                    if t == 1:
                        parts.append(ch)
                    elif t == 2:
                        parts[-1] += ch
                assert len(parts) == k
                raw = saved[r["id"]]["parts"]
                diagnoses.append(
                    dict(
                        id=r["id"],
                        lang=r["lang"],
                        k=k,
                        chosen=len(raw),
                        parts=parts,
                        observed=grade(raw, r["refs"][:1])["exact"],
                        oracle=grade(parts, r["refs"][:1])["exact"],
                    )
                )
        # The sentence writer has its own trained palette. Any new physical cells
        # are prompt-local additions. No Unicode labels enter its forward pass.
        writer = Writer(device="cuda")
        head = editor("cuda")
        face = CorpusFace()
        alphabet = {bytes(b): c for b, c in zip(writer.palette, writer.transcript)}

        def rendered(text):
            cells = []
            for c in text:
                try:
                    cell = face.tile(c)
                except ValueError:
                    cell = face.tile("�")
                cells.append(np.frombuffer(cell, np.uint8))
                alphabet.setdefault(cell, c)
            return np.stack(cells)

        conjunction = rendered(" and ")
        batches = []
        for r in rows + confirm:
            if r["lang"] != "en":
                continue
            question, page = tiles[r["qi"]], tiles[r["pi"]]
            for idx, c in zip(r["qi"] + r["pi"], r["q"] + r["p"], strict=True):
                alphabet.setdefault(bytes(tiles[idx]), c)
            source = saved[r["id"]]
            chosen = [page[a:b] for a, b in source["spans"]]
            assert [sha256(part.tobytes()).hexdigest() for part in chosen] == source[
                "ink"
            ]
            for mode, parts in (
                ("read", chosen),
                ("given", [rendered(x) for x in r["refs"][0]]),
            ):
                if (
                    len(question)
                    + sum(map(len, parts))
                    + len(conjunction) * max(0, len(parts) - 1)
                    + 2
                    <= 600
                ):
                    batches.append((r, mode, question, parts))
        for i in range(0, len(batches), 12):
            g = batches[i : i + 12]
            ans = sentences(
                writer, head, [v[2] for v in g], [v[3] for v in g], conjunction
            )
            for (r, mode, _, _), (pixels, end) in zip(g, ans, strict=True):
                txt = "".join(alphabet[bytes(cell)] for cell in pixels)
                norm = normalize(txt)
                target = r["refs"][0]
                utterances.append(
                    dict(
                        id=r["id"],
                        mode=mode,
                        answer=txt,
                        stopped=end,
                        all_present=bool(
                            end and all(normalize(w) in norm for w in target)
                        ),
                        sha256=sha256(pixels.tobytes()).hexdigest(),
                        cells=len(pixels),
                    )
                )
    packed(PUB / "given-count.json.gz", diagnoses)
    packed(PUB / "sentences.json.gz", utterances)
    result = {}
    for lang in ("en", "zh"):
        g = [r for r in diagnoses if r["lang"] == lang]
        result[lang] = dict(
            n=len(g),
            chosen_count_correct=sum(r["chosen"] == r["k"] for r in g),
            exact=sum(r["observed"] for r in g),
            given_count=sum(r["oracle"] for r in g),
        )
    for mode in ("read", "given"):
        g = [r for r in utterances if r["mode"] == mode]
        result[mode] = dict(
            n=len(g),
            stopped=sum(r["stopped"] for r in g),
            contains_all=sum(r["all_present"] for r in g),
        )
    dump(PUB / "composition.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
