import pytest
import torch
import torch.nn.functional as F

from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


def torch_forward(x, w):
    return F.conv2d(x, w, padding=w.size(3) // 2, groups=w.size(0))

def test_cuda_available():
    if not torch.cuda.is_available():
        pytest.exit("no cuda available")


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64, 192])
@pytest.mark.parametrize("kernel_size", [3, 7, 13, 31])
@pytest.mark.parametrize("resolution", [16, 32])
@pytest.mark.parametrize("seed", [0, 42])
def test_forward_fp32(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    torch.random.manual_seed(seed)
    with torch.cuda.device(0):
        x = torch.randn(batch_size, channels, resolution, resolution).cuda()
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size).cuda()
        y = m(x)
        y_ref = torch_forward(x, m.weight)
        assert y.dtype == torch.float
        assert torch.allclose(y, y_ref)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 7, 13])
@pytest.mark.parametrize("resolution", [16])
@pytest.mark.parametrize("seed", [0, 42])
def test_forward_fp16(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    torch.random.manual_seed(seed)
    with torch.cuda.device(0):
        x = torch.randn(batch_size, channels, resolution, resolution).cuda().half()
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size).cuda()
        with torch.cuda.amp.autocast():
            y = m(x)
            y_ref = torch_forward(x, m.weight)
        assert y.dtype == torch.half
        assert y_ref.dtype == torch.half
        assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-6), (y - y_ref).max()


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 7, 13])
@pytest.mark.parametrize("resolution", [16])
@pytest.mark.parametrize("seed", [0, 42])
def test_backward_fp32(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    torch.random.manual_seed(seed)
    with torch.cuda.device(0):
        x = torch.randn(batch_size, channels, resolution, resolution).cuda()
        x.requires_grad = True
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size).cuda()
        y = m(x)
        y.mean().backward()
        dx = x.grad.clone()
        dw = m.weight.grad.clone()
        x.grad = None
        m.weight.grad = None
        y_ref = torch_forward(x, m.weight)
        y_ref.mean().backward()
        dx_ref = x.grad.clone()
        dw_ref = m.weight.grad.clone()
        assert torch.allclose(dx, dx_ref), (dx - dx_ref).max()
        assert torch.allclose(dw, dw_ref, rtol=1e-4, atol=1e-6), (dx - dx_ref).max()


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 7, 13])
@pytest.mark.parametrize("resolution", [16])
@pytest.mark.parametrize("seed", [0, 42])
def test_backward_fp16(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    torch.random.manual_seed(seed)
    with torch.cuda.device(0):
        x = torch.randn(batch_size, channels, resolution, resolution).cuda().half()
        x.requires_grad = True
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size).cuda()
        with torch.cuda.amp.autocast():
            y = m(x)
            y.mean().backward()
        dx = x.grad.clone()
        dw = m.weight.grad.clone()
        x.grad = None
        m.weight.grad = None
        with torch.cuda.amp.autocast():
            y_ref = torch_forward(x, m.weight)
            y_ref.mean().backward()
        dx_ref = x.grad.clone()
        dw_ref = m.weight.grad.clone()
        assert dx.dtype == dx_ref.dtype
        assert dx.dtype == torch.half
        assert dw.dtype == dw_ref.dtype
        assert dw.dtype == torch.float
        assert torch.allclose(dx, dx_ref), (dx - dx_ref).max()
        assert torch.allclose(dw, dw_ref, rtol=1e-4, atol=1e-6), (dw - dw_ref).max()
