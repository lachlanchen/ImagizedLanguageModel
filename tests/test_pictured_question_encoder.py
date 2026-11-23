import torch
from ilm.visual_lm.paired_question_encoder import PicturedQuestionEncoder
from scripts.train_pictured_question_alignment import loss_for


def test_batch_order_and_local_bitmap_ids_do_not_change_question_vectors():
    torch.manual_seed(2)
    model = PicturedQuestionEncoder().eval()
    short = bytes([0x11]) * 128
    long = bytes([0x97]) * 128 + short * 4
    a = model.encode([short, long, short])
    b = model.encode([long, short], batch_size=1)
    torch.testing.assert_close(a[0], b[1])
    torch.testing.assert_close(a[1], b[0])
    torch.testing.assert_close(a[0], a[2])


def test_duplicate_visual_questions_count_towards_positive_mass():
    logits = torch.randn(3, 3, requires_grad=True)
    same = torch.zeros(3, dtype=torch.long)
    loss = loss_for(logits, same, torch.arange(3))
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits)
    )
