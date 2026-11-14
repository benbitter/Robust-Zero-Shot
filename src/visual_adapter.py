"""A residual MLP on frozen CLIP embeddings; contains no CLIP model."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class VisualAdapter(nn.Module):
    def __init__(self, dimension=512, hidden_dim=128, alpha=.2):
        super().__init__()
        if dimension < 1 or hidden_dim < 1 or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('Positive dimensions and 0 < alpha <= 1 required')
        self.dimension, self.hidden_dim, self.alpha = dimension, hidden_dim, float(alpha)
        self.residual = nn.Sequential(nn.Linear(dimension, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dimension))
        # Start exactly at the frozen CLIP embedding. The residual learns a correction.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        x = F.normalize(x.float(), dim=-1)
        return F.normalize(x + self.alpha*self.residual(x), dim=-1)

    def configuration(self):
        return {'dimension':self.dimension,'hidden_dim':self.hidden_dim,'alpha':self.alpha}


def view_scores(adapter, features, prototypes):
    """Adapt EACH view, normalize, score, then average scores across views."""
    if features.ndim != 3 or prototypes.ndim != 2 or features.shape[-1] != prototypes.shape[-1]:
        raise ValueError('Expected N x V x D features and C x D prototypes')
    return (adapter(features) @ F.normalize(prototypes.float(), dim=-1).T).mean(dim=1)
