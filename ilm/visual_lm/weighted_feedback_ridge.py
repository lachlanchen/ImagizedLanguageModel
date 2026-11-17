"""FP64 full-support weighted ridge readout fitting, with streamed moments."""
import torch


class WeightedFeedbackMoments:
    """Sufficient statistics for sum_i c_i ||phi_i K - y_i||² + lambda||K||²."""
    def __init__(self, features, outputs, device):
        self.gram=torch.zeros((features,features),dtype=torch.float64,device=device)
        self.cross=torch.zeros((features,outputs),dtype=torch.float64,device=device)
        self.rows=0

    @torch.no_grad()
    def add(self, phi, target, coefficient):
        if (phi.ndim!=2 or target.ndim!=2 or len(phi)!=len(target) or coefficient.shape!=(len(phi),)
                or phi.shape[1]!=len(self.gram) or target.shape[1]!=self.cross.shape[1] or not len(phi)):
            raise ValueError('Weighted native-coordinate rows differ')
        x,y,c=phi.double(),target.double(),coefficient.double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(c).all() and (c>0).all()):
            raise ValueError('Finite rows and strictly positive weights required')
        a=x*c.sqrt()[:,None];b=y*c.sqrt()[:,None]
        self.gram.add_(a.T@a);self.cross.add_(a.T@b);self.rows+=len(x)

    @torch.no_grad()
    def solve(self, ridge_scale=1e-4):
        if not self.rows or not 0<ridge_scale<1:
            raise ValueError('Require observed rows and a fixed positive ridge scale')
        ridge=self.gram.diag().mean()*ridge_scale
        if not bool(torch.isfinite(ridge) and ridge>0):
            raise ValueError('Degenerate feature energy')
        system=self.gram.clone();system.diagonal().add_(ridge)
        factor,info=torch.linalg.cholesky_ex(system)
        if int(info)!=0:raise ValueError('Regularized Gram matrix is not numerically positive definite')
        solution=torch.cholesky_solve(self.cross,factor)
        residual=(system@solution-self.cross).norm()/self.cross.norm().clamp_min(1e-30)
        if not bool(torch.isfinite(solution).all() and residual<1e-7):
            raise ValueError('Weighted normal-equation residual exceeds tolerance')
        return solution,dict(rows=self.rows,features=len(self.gram),outputs=self.cross.shape[1],
            ridge_scale=ridge_scale,ridge=float(ridge),normal_equation_relative_residual=float(residual))
