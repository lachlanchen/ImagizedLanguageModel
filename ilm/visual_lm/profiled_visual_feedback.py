"""Fixed-ridge variable projection for the visual reader's nonlinear map."""
import torch
from torch.nn import functional as F

from .nonlinear_visual_feedback import NonlinearVisualFeedback
from .weighted_feedback_ridge import WeightedFeedbackMoments


class Float32VisualFeedback(NonlinearVisualFeedback):
    """Use the same saved parameters with explicitly FP32 feedback execution."""
    def forward(self, query):
        with torch.autocast(device_type=query.device.type, enabled=False):
            normalized=self.original[0](query.float())
            return self.original[1](normalized)+self.project(F.gelu(self.expand(normalized)))


def feedback_features(head, query):
    """Differentiable fixed-normalization feature map, with a constant column."""
    with torch.autocast(device_type=query.device.type,enabled=False):
        normalized=head.original[0](query.float())
        hidden=F.gelu(head.expand(normalized))
        return torch.cat((normalized.new_ones((len(query),1)),normalized,hidden),dim=-1)


class FixedRidgeMoments(WeightedFeedbackMoments):
    @torch.no_grad()
    def solve_fixed(self, ridge):
        value=self.gram.new_tensor(ridge)
        if not self.rows or not bool(torch.isfinite(value) and value>0):
            raise ValueError('Observed rows and a fixed positive ridge are required')
        system=self.gram.clone();system.diagonal().add_(value)
        factor,info=torch.linalg.cholesky_ex(system)
        if int(info)!=0:raise ValueError('Fixed-ridge Cholesky failed')
        coefficient=torch.cholesky_solve(self.cross,factor)
        residual=(system@coefficient-self.cross).norm()/self.cross.norm().clamp_min(1e-30)
        if not bool(torch.isfinite(coefficient).all() and residual<1e-7):
            raise ValueError('Fixed-ridge normal-equation residual failed')
        return coefficient,dict(rows=self.rows,features=len(self.gram),outputs=self.cross.shape[1],
            ridge=float(value),normal_equation_relative_residual=float(residual))


def native_partial_loss(phi, coefficient, centered_target, weight):
    """Data term at a detached solved readout; retain the feature graph only."""
    if (phi.ndim!=2 or centered_target.ndim!=2 or weight.shape!=(len(phi),)
            or coefficient.shape!=(phi.shape[1],centered_target.shape[1])
            or len(centered_target)!=len(phi) or not len(phi)):
        raise ValueError('Profiled objective rows differ')
    return (weight.double()*(phi.double()@coefficient.detach().double()-centered_target.double()).square().sum(-1)).sum()


@torch.no_grad()
def install_native_readout(head, coefficient, scale):
    width=head.original[1].in_features
    expected=(1+width+head.hidden_width,head.original[1].out_features)
    if coefficient.shape!=expected or scale.shape!=(expected[1],) or not bool((scale>0).all()):
        raise ValueError('Native output coefficients or affine scale differ')
    normalized=coefficient.double()/scale.double()[None]
    head.original[1].bias.copy_(normalized[0].float())
    head.original[1].weight.copy_(normalized[1:width+1].T.float())
    head.project.weight.copy_(normalized[width+1:].T.float())
