"""
Let's compare MHA with MLA (with RoPe)

Implements a simple MHA and MLA model layer
while comparing memory usage and latency.
"""

import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(
            self,
            d_model,
            d_head,
            n_heads
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, d_head * n_heads)
        self.k_proj = nn.Linear(d_model, d_head * n_heads)
        self.v_proj = nn.Linear(d_model, d_head * n_heads)

        self.out_proj = nn.Linear(d_head * n_heads, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(x.shape[0], x.shape[1], self.n_heads, self.d_head)
        k = k.view(x.shape[0], x.shape[1], self.n_heads, self.d_head)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, self.d_head)
