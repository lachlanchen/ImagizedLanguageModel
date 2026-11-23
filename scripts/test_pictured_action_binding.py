"""Test a passage edit crossed with a single pictured action, using both candidate names and free generation."""

import argparse
from collections import Counter, defaultdict
import json
from time import monotonic

import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.audit_book_sequence_probability import score_batch
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_role_change_actions import ROOT, WORK as PREVIOUS, PUBLIC as LAST
from scripts.probe_role_change_wording import GLOSSES
from scripts.role_change_corpus import fictional, load
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

WORK = ROOT / "artifacts/action_binding_v1"
PUBLIC = ROOT / "references/evidence/action_binding_20261124"


def trials(scenes, face):
    for c in scenes:
        for w, page in enumerate(c["pages"]):
            for lang, qs in c["qs"].items():
                for action, question in enumerate(qs):
                    i = w ^ action
                    yield (
                        f"{c['key']}/{w}/{lang}/{action}",
                        lang,
                        face(page),
                        face(question),
                        face(c["names"][i]),
                        face(c["names"][1 - i]),
                    )


def groups():
    d = load()
    rewritten = [
        fictional(
            c["key"], list(zip(c["actions"], e, strict=True)), c["names"], c["kind"]
        )
        for c, e in zip(d["fresh"], GLOSSES, strict=True)
    ]
    return dict(stories=d["fresh"], rewrites=rewritten, credits=d["duie"])


@torch.inference_mode()
def evaluate(folder, arm, sets):
    started = monotonic()
    WORK.mkdir(exist_ok=True)
    writer = PreparedGlyphResponder(folder, "cuda")
    face = Face(FACES[0])
    summary = {}
    for group, scenes in sets.items():
        requests = list(trials(scenes, face))
        assert len(requests) == len(scenes) * 8 and len(
            {r[0] for r in requests}
        ) == len(requests)
        rows = [
            Row(group, key, p, q, y, b"")
            for key, lang, p, q, a, b in requests
            for y in (a, b)
        ]
        read, alphabet, pairs, ys = inputs(writer.net.writer, rows)
        scores = []
        for i in range(0, len(rows), 20):
            s, _ = score_batch(
                writer.net, read, alphabet, pairs[i : i + 20], ys[i : i + 20]
            )
            scores.extend(s)
        images = []
        ends = []
        for i in range(0, len(requests), 24):
            for im, end in writer.answer_many(
                [(tensor(p), tensor(q)) for _, _, p, q, _, _ in requests[i : i + 24]]
            ):
                images.append(im.numpy().tobytes())
                ends.append(bool(end))
        report = defaultdict(Counter)
        scene_scores = defaultdict(list)
        detail = []
        for (key, lang, p, q, a, b), right, wrong, im, end in zip(
            requests, scores[::2], scores[1::2], images, ends, strict=True
        ):
            exact = bool(end and im == a)
            preference = right > wrong
            # A longer phrasing can still identify the requested participant.
            cells = [im[k : k + 128] for k in range(0, len(im), 128)]

            def mentions(name):
                chars = len(name) // 128
                return any(
                    b"".join(cells[j : j + chars]) == name
                    for j in range(len(cells) - chars + 1)
                )

            mentions_only = end and mentions(a) and not mentions(b)
            report[lang].update(
                requests=1,
                preferred=preference,
                exact=exact,
                identifies=mentions_only,
                stopped=end,
            )
            scene_scores[key.rsplit("/", 3)[0], lang].append(
                (preference, exact, mentions_only)
            )
            detail.append(
                dict(
                    key=key,
                    log_margin=right - wrong,
                    exact=exact,
                    identifies=mentions_only,
                    stopped=end,
                )
            )
        for (k, lang), vs in scene_scores.items():
            assert len(vs) == 4
            report[lang].update(
                contrasts=1,
                all_preferred=all(v[0] for v in vs),
                all_exact=all(v[1] for v in vs),
                all_identified=all(v[2] for v in vs),
            )
        path = WORK / f"{arm}_{group}.npz"
        store_streams(path, images, ends)
        receipt = WORK / f"{arm}_{group}.json"
        record(receipt, detail)
        summary[group] = dict(
            counts=report, detail_sha=sha(receipt), images_sha=sha(path)
        )
    return dict(groups=summary, seconds=monotonic() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["aligned", "independent"])
    arg = parser.parse_args().arm
    with gpu_owner():
        torch.set_num_threads(2)
        folder = PREVIOUS / arg / "page"
        assert all(
            sha(folder / k) == v
            for k, v in json.loads((LAST / f"{arg}_fit.json").read_text())[
                "weights"
            ].items()
        )
        record(PUBLIC / f"single_{arg}.json", evaluate(folder, arg, groups()))
