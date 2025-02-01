from math import sqrt
import torch
import torch.nn as nn

def ring_attn(
        q,
        k,
        v,
        b_q,
        b_kv
):
    # WIP
    batch_size, seq_len, dim = q.shape
    output = torch.zeros_like(v)

    for i in range(seq_len):
        qi = q[:, i, :]
        current_max = -torch.inf
        current_sum = 0
        acc = 0

        # each GPU has a small copy of K, V
        # K and V are split into chunks each of size b_kv
        for start in range(0, seq_len, b_kv):
            end = start + b_kv 

            kj = k[:, start:end, :]
            vj = v[:, start:end, :]
            scores = qi@kj.transpose(-1, -2)

            chunk_max = scores.max()
            if chunk_max > current_max:
                old_max = current_max
                current_max = chunk_max
                acc = acc * torch.exp(old_max-current_max)
                current_sum = current_sum*torch.exp(old_max-current_max)

            exp_scores = torch.exp(scores-current_max)
            current_sum += exp_scores.sum()
            acc += exp_scores @ vj

        output[:, i, :] = acc/current_sum

    return output/sqrt(d)


