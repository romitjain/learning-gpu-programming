from math import sqrt
import torch
import torch.nn as nn

def simple_attn(q, k, v):
    """
    This function implements simple attention layer
    """

    d = q.shape[-1]
    s = q@k.transpose(-1, -2)
    s = nn.functional.softmax(s, dim=-1)

    return s@v

def flash_attn(q, k, v):
    seq_len, dim = q.shape
    out = torch.zeros_like(q)

    for i in range(seq_len):
        # single row
        qi = q[i, :]

        current_max = -torch.inf
        current_sum = 0
        acc = torch.zeros_like(qi)

        old_sum = current_sum
        old_max = current_max

        # single element of single out row
        for j in range(seq_len):
            kj = k[j, :]
            vj = v[j, :]

            xi = qi@kj # scalar value
            current_max = max(current_max, xi)
            current_sum = current_sum*torch.exp(old_max-current_max) + torch.exp(xi-current_max)

            acc = acc*( (old_sum * torch.exp(old_max-current_max))/current_sum ) + \
                vj*torch.exp(xi-current_max)/current_sum

            old_max = current_max
            old_sum = current_sum

        out[i] = acc

    return out

if __name__ == '__main__':
    s, d = 256, 756
    q_input = torch.randn((s, d), dtype=torch.float32)
    k_input = torch.randn((s, d), dtype=torch.float32)
    v_input = torch.randn((s, d), dtype=torch.float32)

    out_naive = simple_attn(q_input, k_input, v_input)
    out_flash = flash_attn(q_input, k_input, v_input)

    assert torch.allclose(out_naive, out_flash, rtol=0, atol=1e-4), "Attn out is not same"

    print(f"Attn impl. is correct")
