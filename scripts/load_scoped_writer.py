"""Load a bitmap writer and restore its recorded question encoder."""
from safetensors.torch import load_file

from ilm.visual_lm.scoped_question import ScopedQuestion
from ilm.visual_lm.grounded_glyph_decoder import GroundedGlyphDecoder
from scripts.train_grounded_glyph_writer import load_writer
from scripts.prepare_pictured_entry_answers import load, sha


def load_visual_writer(directory, device):
    receipt=load(directory/"fit.json")
    if "question_architecture" not in receipt:
        return load_writer(directory, device)
    path=directory/"final.safetensors"
    mode=receipt["question_architecture"]
    if mode not in ("ordered","unordered") or sha(path)!=receipt["sha256"]:
        raise ValueError("Unsupported or changed visual question checkpoint")
    net=GroundedGlyphDecoder()
    net.question=ScopedQuestion(ordered=mode=="ordered")
    net.load_state_dict(load_file(str(path)), strict=True)
    return net.to(device).eval()
