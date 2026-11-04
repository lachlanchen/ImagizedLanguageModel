import torch

from ilm.visual_lm.question_assignment import assignment_objective


def test_assignment_cancels_answer_marginals_and_question_offsets():
    e=torch.tensor([[.14,.19],[.22,.13]],dtype=torch.float64,requires_grad=True)
    loss,values=assignment_objective(e,1.)
    shifted=e+torch.tensor([[.2],[-.1]])+torch.tensor([[.07,.11]])
    _,offset=assignment_objective(shifted,1.)
    assert torch.allclose(values['assignment_margin'],offset['assignment_margin'])
    assert torch.allclose(values['assignment_loss'],offset['assignment_loss'])
    loss.backward()
    assert e.grad[0,0]>0 and e.grad[1,1]>0 and e.grad[0,1]<0 and e.grad[1,0]<0
    _,marginal=assignment_objective(torch.tensor([[.1,.3],[.1,.3]]),1.)
    assert abs(float(marginal['assignment_margin']))<1e-7
