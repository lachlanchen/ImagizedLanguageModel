"""Select answer pixels from a page using a cross-script question cue."""

from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.paired_question_encoder import PicturedQuestionEncoder
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead


class QuestionEvidenceReader:
    def __init__(self, reader_weights, question_weights, head_weights, device="cpu"):
        self.reader = FontInvariantReader.restore(Path(reader_weights), device).eval()
        self.questions = PicturedQuestionEncoder().to(device).eval()
        self.questions.load_state_dict(load_file(str(question_weights)))
        self.head = QuestionEvidenceHead().to(device).eval()
        self.head.load_state_dict(load_file(str(head_weights)))

    def answer(self, page, question):
        return predict(
            self.reader, self.questions, {"matched": self.head}, [(page, question)]
        )["matched"][0]


@torch.inference_mode()
def predict(read, model, heads, pairs):
    # Cover windows with 47 cells of shared context. Every valid interval is
    # represented at least once. Select the best global score across windows.
    vectors = model.encode([q for _, q in pairs])
    windows = []
    found = {a: [(-float("inf"), b"") for _ in pairs] for a in heads}
    for i, (p, q) in enumerate(pairs):
        cap = 512 - len(q) // 128 - 3
        if cap < 48:
            raise ValueError("Question too long for the 48-glyph support")
        for j in range(0, len(p) // 128, cap - 47):
            part = p[j * 128 : (j + cap) * 128]
            windows.append((i, part, q))
            if (j + cap) * 128 >= len(p):
                break
    for j in range(0, len(windows), 24):
        batch = windows[j : j + 24]
        states = [
            s[:, :770] for _, s in contextualize(read, [(p, q) for _, p, q in batch])
        ]
        h = pad_sequence(states, batch_first=True).to(read.markers.device)
        v = vectors[[i for i, _, _ in batch]]
        for arm, head in heads.items():
            scores = h[:, :, 768:770].float() if head is None else head(h, v)
            for k, (i, p, q) in enumerate(batch):
                left, right = scores[k, : len(p) // 128].unbind(1)
                logits = left[:, None] + F.pad(right, (0, 47), value=-torch.inf).unfold(
                    0, 48, 1
                )
                val, ix = logits.flatten().max(0)
                if float(val) > found[arm][i][0]:
                    a, b = divmod(int(ix), 48)
                    found[arm][i] = (float(val), p[a * 128 : (a + b + 1) * 128])
    return {a: [y for _, y in rs] for a, rs in found.items()}
