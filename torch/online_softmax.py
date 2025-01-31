import torch

def online_softmax(row):
    """
    In principle implementation of online softmax
    """

    current_max = -torch.inf
    current_exp = 0
    acc = torch.zeros_like(row)
    old_max = current_max

    for i in range(row.shape[-1]):
        current_max = max(current_max, row[i])
        current_exp = current_exp*torch.exp(old_max-current_max) \
            + torch.exp(row[i] - current_max)

        old_max = current_max

    for i in range(row.shape[-1]):
        acc[i] = torch.exp(row[i] - current_max)
        acc[i] = acc[i] / current_exp

    return acc


if __name__ == '__main__':
    inp = torch.randn(2048, dtype=torch.float32)
    out1 = torch.nn.functional.softmax(inp) 
    out2 = online_softmax(inp)

    assert torch.allclose(out1, out2), "Online softmax impl is not correct"

    print(f"Online softmax impl. is correct, {out1}, {out2}")
