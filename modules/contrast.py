import torch
from torch import nn
import torch.nn.functional as F


def generate_pos_mask(z, k):
    """Return a positive-pair mask with self-pairs and optional top-k neighbors."""
    N = z.shape[0]
    k = int(k)
    if k <= 0:
        return torch.eye(N, device=z.device, dtype=torch.float32)

    z = z.detach()
    z_norm = F.normalize(z, p=2, dim=-1)
    S = torch.mm(z_norm, z_norm.t())

    S.fill_diagonal_(-1e9)
    k = min(k, N - 1)
    topk_sim, topk_idx = torch.topk(S, k=k, dim=1)
    valid = topk_sim > 0

    pos_mask = torch.zeros(N, N, device=z.device, dtype=torch.float32)
    row_idx = torch.arange(N, device=z.device).unsqueeze(1).expand_as(topk_idx)
    pos_mask[row_idx[valid], topk_idx[valid]] = 1.0
    pos_mask.fill_diagonal_(1.0)
    return pos_mask


class Contrast(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        self.EPS = 1e-8

    def exp_sim_matrix(self, z1, z2, tau):
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)
        return torch.exp(torch.mm(z1_norm, z2_norm.t()) / tau)

    def loss_one_direction(self, exp_sim, mask):
        pos_score = (exp_sim * mask).sum(dim=1)
        sum_exp = exp_sim.sum(dim=1)
        return -(torch.log(pos_score + self.EPS) - torch.log(sum_exp + self.EPS)).mean()

    def forward(self, z1, z2, mask):
        mask = mask.float().to(z1.device)
        exp_sim_12 = self.exp_sim_matrix(z1, z2, self.tau)
        loss_12 = self.loss_one_direction(exp_sim_12, mask)
        loss_21 = self.loss_one_direction(exp_sim_12.t(), mask.t())
        return (loss_12 + loss_21) / 2.0
