"""Draw the object of an action named in a pictured question about one Chinese line."""

from pathlib import Path
import argparse
import json
import torch
from safetensors.torch import load_file
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.questioned_events import QuestionedEvents
from scripts.answer_pictured_action import restore_lookup
from scripts.answer_visual_gloss import load_focus
from scripts.write_visual_gloss import restore, load_adapter, respond, ROOT, DEFAULT
from scripts.read_optical_role import digest
from scripts.respond_pointing_glyphs import render_text

RELEASE = ROOT / "models/questioned_events"
COMPARATOR = ROOT / "models/visual_action_meaning/aligned"


def load_head(folder, device):
    spec = json.loads((folder / "weights.json").read_text())
    file = folder / "events.safetensors"
    if (
        digest(file) != spec["sha256"]
        or digest(COMPARATOR / "lookup.safetensors") != spec["reader"]
    ):
        raise ValueError("Changed event or comparison weights")
    model = QuestionedEvents().to(device)
    model.load_state_dict(load_file(str(file)))
    return model.eval()


def components(device):
    return (
        restore(device=device),
        load_focus(ROOT / "models/visual_clause_roles", device),
        restore_lookup(COMPARATOR, device),
        load_adapter(ROOT / "models/pronoun_writer", device),
    )


@torch.inference_mode()
def answer(parts, head, line, question):
    if not (2 <= len(line) <= 32 and len(question) > 0):
        raise ValueError("Expected a single line of 2–32 cells and a nonempty question")
    writer, syntax, lookup, adapter = parts
    device = next(writer.parameters()).device

    def features(raw):
        return cell_features(torch.as_tensor(raw, device=device)[None])

    lengths = torch.tensor([len(line)], device=device)
    h = encoding_windows(writer, features(line))
    prior = syntax.arc_scores(h, lengths)
    a, k = head.page(h, prior, lookup.predicates(h))
    q, m = lookup.questions(
        features(question), torch.tensor([len(question)], device=device)
    )
    verb, obj = head.select(a, k, q, m)
    v, o = int(verb[0]), int(obj[0])
    ink, stopped = respond(writer, line, o, context_adapter=adapter)
    return ink, dict(verb_cell=v, object_cell=o, stopped=stopped)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("line", "question"):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--" + name, type=Path)
        g.add_argument("--" + name + "-text")
    p.add_argument("--reader", type=Path, default=RELEASE / "aligned")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--transcribe", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)

    def draw(text):
        return render_text(text, DEFAULT, 2).numpy()

    line = read_image(args.line, None) if args.line else draw(args.line_text)
    question = (
        read_image(args.question, None) if args.question else draw(args.question_text)
    )
    ink, report = answer(
        components(args.device), load_head(args.reader, args.device), line, question
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(ink, args.output)
    if args.transcribe:
        alphabet = "abcdefghijklmnopqrstuvwxyz-"
        letters = {x.tobytes(): c for x, c in zip(draw(alphabet), alphabet)}
        report["text"] = "".join(letters.get(x.tobytes(), "�") for x in ink)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
