import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.losses.cross_entropy import CrossEntropyLoss

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32] + ([torch.bfloat16] if is_sm8x else [])
)
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("inplace_backward", [False, True])
# @pytest.mark.parametrize("inplace_backward", [False])
@pytest.mark.parametrize("lse_square_scale", [0.0, 1e-2])
# @pytest.mark.parametrize("lse_square_scale", [1e-2])
@pytest.mark.parametrize("smoothing", [0.0, 0.9])
# @pytest.mark.parametrize("smoothing", [0.0])
@pytest.mark.parametrize("vocab_size", [50257, 128 * 1024])  # test vocab larger than 64k for split
# @pytest.mark.parametrize("vocab_size", [12])
def test_cross_entropy_loss(vocab_size, smoothing, lse_square_scale, inplace_backward, dtype):
    device = "cuda"
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 128
    x_pt = torch.randn(
        batch_size * seqlen, vocab_size, device=device, dtype=dtype, requires_grad=True
    )
    x = x_pt.detach().clone().requires_grad_()
    y = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    if batch_size * seqlen > 10:
        y[torch.randperm(batch_size * seqlen)[:10]] = -100
    model_pt = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    model = CrossEntropyLoss(
        label_smoothing=smoothing,
        lse_square_scale=lse_square_scale,
        inplace_backward=inplace_backward,
    )
    out = model(x, y)
    out_pt = model_pt(x_pt.float(), y)
    if lse_square_scale > 0.0:
        lse_pt = torch.logsumexp(x_pt.float(), dim=-1)
        out_pt += lse_square_scale * (lse_pt[y != -100] ** 2).mean()
    assert torch.allclose(out, out_pt, rtol=1e-5, atol=1e-6)

    g = torch.randn_like(out)
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
