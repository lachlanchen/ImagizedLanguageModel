import torch
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead


def test_zero_residual_preserves_scores_and_question_can_change_endpoint_order():
    torch.manual_seed(3)
    head = QuestionEvidenceHead(4)
    h = torch.zeros(1, 2, 770)
    h[0, 0, 0] = 1
    h[0, 1, 0] = -1
    q = torch.zeros(1, 128)
    q[:, 0] = 1
    torch.testing.assert_close(head(h, q), h[:, :, 768:770])
    with torch.no_grad():
        head.page.weight[0, 0] = 1
        head.question.weight.zero_()
        head.question.weight[0, 0] = 1
    assert int(head(h, q)[0, :, 0].argmax()) == 0
    assert int(head(h, -q)[0, :, 0].argmax()) == 1
