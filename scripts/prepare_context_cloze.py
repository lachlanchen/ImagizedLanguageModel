"""Reserve the remaining independent long CoQA passages for context discrimination."""

from scripts.prepare_dialogue_ink import ROOT
from scripts.prepare_word_canvas import reserved as canvas
from scripts.prepare_word_memory import write_reservation, reserved as load_reservation
from scripts.prepare_compose_dialogue import prepared

WORK = ROOT / "artifacts/context_cloze_v1"
PUB = ROOT / "references/evidence/context_cloze_v1"
MODEL = ROOT / "models/context_cloze"


def reserved():
    return load_reservation(WORK, PUB)


if __name__ == "__main__":
    write_reservation(
        547, 694, [canvas(), prepared()["fresh"], load_reservation()], WORK, PUB
    )
