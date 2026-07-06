import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,
                 padding='same', lookahead=None):
    """
    lookahead=None -> original symmetric behavior (padding=`padding`).
    lookahead=L    -> keep past half + L future taps (peak on current bin),
                      renormalize, asymmetric-pad -> length-preserving,
                      out[t] uses inputs[t-p .. t+L]. L = look-ahead in bins
                      = L_algo. L=0 fully causal; L=p reproduces 'same'.
    """
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))   # K_full, symmetric

    K_full = gaussKernel.shape[0]
    p = K_full // 2                                               # past taps; peak index

    if lookahead is not None:
        assert 0 <= lookahead <= p, f"lookahead must be in [0, {p}]"
        gaussKernel = gaussKernel[: p + lookahead + 1]            # offsets -p..+lookahead
        gaussKernel = gaussKernel / gaussKernel.sum()

    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32,
                               device=device).view(1, 1, -1)

    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)                             # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)                    # [C, 1, K]

    if lookahead is not None:
        inputs = F.pad(inputs, (p, lookahead))                  # left=past, right=future
        smoothed = F.conv1d(inputs, gaussKernel, padding=0, groups=C)
    else:
        smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)

    return smoothed.permute(0, 2, 1)                            # [B, T, C]
