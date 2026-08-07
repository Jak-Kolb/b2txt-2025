import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

# C2 — the kernel depends only on (std, size, lookahead, channels, device) and was being rebuilt
# in scipy and re-copied host->device on EVERY call: once per batch in training, once per bin in
# streaming. Memoized; the weights are unchanged.
_KERNEL_CACHE = {}


def _grouped_kernel(smooth_kernel_std, smooth_kernel_size, lookahead, channels, device):
    key = (float(smooth_kernel_std), int(smooth_kernel_size),
           None if lookahead is None else int(lookahead), int(channels), str(device))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

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

    out = (torch.tensor(gaussKernel, dtype=torch.float32, device=device)
           .view(1, 1, -1).repeat(channels, 1, 1), p)             # [C, 1, K]
    _KERNEL_CACHE[key] = out
    return out


def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,
                 padding='same', lookahead=None):
    """
    lookahead=None -> original symmetric behavior (padding=`padding`).
    lookahead=L    -> keep past half + L future taps (peak on current bin),
                      renormalize, asymmetric-pad -> length-preserving,
                      out[t] uses inputs[t-p .. t+L]. L = look-ahead in bins
                      = L_algo. L=0 fully causal; L=p reproduces 'same'.
    """
    B, T, C = inputs.shape
    gaussKernel, p = _grouped_kernel(smooth_kernel_std, smooth_kernel_size,
                                     lookahead, C, device)

    inputs = inputs.permute(0, 2, 1)                             # [B, C, T]

    if lookahead is not None:
        inputs = F.pad(inputs, (p, lookahead))                  # left=past, right=future
        smoothed = F.conv1d(inputs, gaussKernel, padding=0, groups=C)
    else:
        smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)

    return smoothed.permute(0, 2, 1)                            # [B, T, C]


def time_mask(features, n_masks=20, max_frac=0.075):
    """C17 — SpecAugment-style time masking. features: [B, T, C].

    Parameterization note: "n=20, max_frac=0.075" is ambiguous, and the aggressive reading (20
    masks EACH up to 7.5 % of T) would blank ~75 % of a trial on average, which is not a
    regularizer but a lobotomy. This implements the sane reading: **max_frac is the total expected
    masked fraction**, shared across n_masks spans. Each span's width is drawn from
    U[0, 2*max_frac*T/n_masks], so E[total] = max_frac*T, in ~20 fine-grained chunks rather than a
    few large holes. For a typical 900-bin trial that is ~68 masked bins in spans of ~0-7.

    Masks are drawn independently per trial, and zero is the correct fill because the features are
    z-scored (zero is the channel mean, not an out-of-distribution value).
    """
    B, T, C = features.shape
    if n_masks <= 0 or max_frac <= 0 or T <= 1:
        return features

    max_w = max(1, int(round(2.0 * max_frac * T / n_masks)))
    w = torch.randint(0, max_w + 1, (B, n_masks), device=features.device)
    # Sample the start so the span stays in bounds; clamp guards T <= max_w.
    span = (T - w).clamp(min=1)
    start = (torch.rand((B, n_masks), device=features.device) * span).long()

    idx = torch.arange(T, device=features.device).view(1, 1, T)
    masked = (idx >= start.unsqueeze(-1)) & (idx < (start + w).unsqueeze(-1))   # [B, n_masks, T]
    keep = ~masked.any(dim=1)                                                  # [B, T]
    return features * keep.unsqueeze(-1).to(features.dtype)


def channel_mask(features, rate=0.10):
    """C18 — electrode/channel masking. features: [B, T, C].

    Zeroes each channel for a whole trial with probability `rate`, drawn independently per trial.
    Whole-trial (not per-timestep) is the point: it simulates an electrode being unavailable, which
    is the failure mode that actually occurs across sessions, and forces the model off any single
    channel. Per-timestep dropout would be a different and much weaker regularizer.
    """
    B, T, C = features.shape
    if rate <= 0:
        return features
    keep = (torch.rand((B, 1, C), device=features.device) >= rate).to(features.dtype)
    return features * keep
