# models/stiefel.py
import torch
from torch import nn

class StiefelParameter(nn.Parameter):
    """Parameter on the Stiefel manifold."""
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

def orthogonal_projection(A, B):
    """Projects A onto the tangent space at B (assuming B is orthogonal)."""
    # You might want to import your utility symmetric function from utils.math_ops
    # For brevity, we define it inline here.
    def symmetric(X):
        size = list(range(len(X.shape)))
        temp = size[-1]
        size.pop()
        size.insert(-1, temp)
        return 0.5 * (X + X.permute(*size))
    return A - B @ symmetric(B.transpose(-1, -2) @ A)

def retraction(A, ref):
    """Retracts A back to the manifold."""
    # Using QR-based retraction.
    data = A + ref
    Q, R = torch.qr(data)
    sign = (R.diag().sign() + 0.5).sign().diag()
    return Q @ sign

class MixOptimizer(object):
    """
    Meta optimizer that remaps StiefelParameter updates to remain on the Stiefel manifold.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        # Pre-step: project gradients for StiefelParameter
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].copy_(p.data)
                    p.grad.data.copy_(orthogonal_projection(p.grad.data, p.data))
        # Optimizer update
        loss = self.optimizer.step(closure)
        # Post-step: retract parameters back to the manifold
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    p.data.copy_(retraction(p.data, self.state[id(p)]))
        return loss
