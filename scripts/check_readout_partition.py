"""Check that the grouped training loss includes the probability of all omitted pieces."""
import torch
from scripts.train_readout_objective import PUB, dump, group_logprob


def main():
    torch.manual_seed(1173)
    errors=[]
    gradient_errors=[]
    divergences=[]
    for count in (33,150,65001):
        teacher=torch.randn(4,count)
        score=torch.randn_like(teacher,requires_grad=True)
        p=teacher.softmax(-1)
        q=score.softmax(-1)
        take=teacher.topk(32,dim=1).indices
        tail=torch.ones_like(score,dtype=torch.bool).scatter_(1,take,False)
        logp=group_logprob(teacher,take)
        logq=group_logprob(score,take)
        reference=torch.cat((q.gather(1,take),(q*tail).sum(-1,keepdim=True)),1)
        errors.append(float((reference-logq.exp()).abs().max().detach()))
        grouped=(logp.exp()*(logp-logq)).sum()
        exact=(p*(teacher.log_softmax(-1)-score.log_softmax(-1))).sum()
        grad,=torch.autograd.grad(grouped,score)
        p_tail=(p*tail).sum(-1,keepdim=True)
        q_tail=(q*tail).sum(-1,keepdim=True)
        want=torch.where(tail,q*(1-p_tail/q_tail),q-p)
        gradient_errors.append(float((grad-want).abs().max().detach()))
        divergences.append([float(grouped.detach()),float(exact.detach())])
    assert max(errors)<1e-6 and max(gradient_errors)<1e-6
    assert all(g<=f+1e-5 for g,f in divergences)
    dump(PUB/"partition.json",dict(max_probability_error=max(errors),max_gradient_error=max(gradient_errors),
                                  vocabularies=[33,150,65001],grouped_and_full=divergences))


if __name__=="__main__":
    main()
