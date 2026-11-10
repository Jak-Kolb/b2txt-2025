import torch
from torch import nn
import torch.nn.functional as F


def patch_frames(x, size, stride):
    '''[B, T, D] -> [B, (T - size) // stride + 1, size * D]: each frame concatenates bins [k*stride, k*stride + size).
    Same values and layout as the earlier unfold/permute/reshape, written as an index gather because torch.compile
    (inductor, torch 2.13) computes wrong gradients for a matmul followed by the overlapping unfold: the day-layer
    weight gradient was off by ~150% relative to eager and finite differences.'''
    n = (x.shape[1] - size) // stride + 1
    idx = (torch.arange(n, device=x.device)[:, None] * stride + torch.arange(size, device=x.device)[None, :]).reshape(-1)
    return x[:, idx].reshape(x.shape[0], n, size * x.shape[2])


def alibi_slopes(n_heads):
    '''Per-head linear distance penalties (ALiBi); relative, so no maximum sequence length.'''
    return torch.tensor([2 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)], dtype=torch.float32)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head, self.dropout = n_heads, d_model // n_heads, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def split(self, x):
        '''[B, T, D] -> q, k, v each [B, H, T, d_head]'''
        batch, steps, _ = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def merge(self, y):
        batch, _, steps, _ = y.shape
        return self.proj(y.transpose(1, 2).reshape(batch, steps, -1))

    def forward(self, x, bias):
        q, k, v = self.split(x)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.merge(y)


def drop_path(x, p, training):
    '''Stochastic depth: zero a residual branch for whole trials with probability p (rescaled when kept).'''
    if p == 0.0 or not training:
        return x
    keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= p
    return x * keep.to(x.dtype) / (1 - p)


class CausalConv(nn.Module):
    '''Conformer convolution module with a causal depthwise convolution (left padding only) and LayerNorm in place
    of BatchNorm, so padded batches and trial lengths do not change any valid frame.'''

    def __init__(self, d_model, kernel, dropout):
        super().__init__()
        self.kernel = kernel
        self.pointwise_in = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = F.glu(self.pointwise_in(x), dim=-1).transpose(1, 2)
        # depthwise convolution as unfold + weighted sum (same result as self.depthwise on the padded input):
        # inductor's conv backward specializes on the sequence length and recompiles for every new length
        windows = F.pad(y, (self.kernel - 1, 0)).unfold(-1, self.kernel, 1)          # [B, D, T, kernel]
        y = (windows * self.depthwise.weight[None]).sum(-1) + self.depthwise.bias[None, :, None]
        return self.drop(self.pointwise_out(F.silu(self.norm(y.transpose(1, 2)))))


class Block(nn.Module):
    '''Pre-norm Transformer block; conv_kernel > 0 adds a pre-norm causal convolution module after attention.'''

    def __init__(self, d_model, n_heads, ffn_mult, dropout, drop_path=0.0, conv_kernel=0, ffn_dropout=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # ffn_dropout adds dropout after the activation (as in the LightBeam code); off keeps the old parameter names
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 *([nn.Dropout(dropout)] if ffn_dropout else []),
                                 nn.Linear(ffn_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.drop_path = drop_path
        if conv_kernel:
            self.norm_conv = nn.LayerNorm(d_model)
            self.conv = CausalConv(d_model, conv_kernel, dropout)

    def forward(self, x, bias):
        x = x + drop_path(self.drop(self.attn(self.norm1(x), bias)), self.drop_path, self.training)
        if hasattr(self, 'conv'):
            x = x + drop_path(self.conv(self.norm_conv(x)), self.drop_path, self.training)
        return x + drop_path(self.drop(self.ffn(self.norm2(x))), self.drop_path, self.training)


class ConformerBlock(nn.Module):
    '''Macaron Conformer block (Gulati et al. 2020): half FFN, self-attention, causal convolution, half FFN, final
    LayerNorm. Its future context comes only from the attention bias it is given.'''

    def __init__(self, d_model, n_heads, ffn_mult, dropout, conv_kernel, drop_path=0.0):
        super().__init__()
        def ffn():
            return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_mult * d_model), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model), nn.Dropout(dropout))
        self.ffn1, self.ffn2 = ffn(), ffn()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = CausalConv(d_model, conv_kernel, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop_path = drop_path

    def forward(self, x, bias):
        p, training = self.drop_path, self.training
        x = x + drop_path(0.5 * self.ffn1(x), p, training)
        x = x + drop_path(self.attn_drop(self.attn(self.norm_attn(x), bias)), p, training)
        x = x + drop_path(self.conv(self.norm_conv(x)), p, training)
        x = x + drop_path(0.5 * self.ffn2(x), p, training)
        return self.norm_out(x)


class TransformerDecoder(nn.Module):
    '''
    Causal Transformer acoustic model with GRUDecoder's day-specific input layers and patching.

    Each layer attends to itself and the previous (window - 1) patch frames. Streaming keeps a
    per-layer key/value cache of that window, which reproduces the offline forward exactly
    (up to floating point) because earlier frames never change under causal attention.

    lookahead_frames R > 0: the last layer also attends to the next R frames, so the output for frame t depends on
    input up to frame t+R (total future context R, not R per layer). Offline only for now (no streaming path).

    block_type 'conv' adds a causal convolution module (conv_kernel frames, past only) to each pre-norm block;
    'conformer' uses ConformerBlock. Neither has a streaming path yet. drop_path: stochastic-depth rate, rising
    linearly from 0 at the first layer to this value at the last.

    shared_input: one input matrix shared by all sessions (day_weight_shared, identity init, in the trainer's day
    parameter group) with per-session biases kept, instead of a matrix per session. No streaming path yet
    (streaming_infer reads day_weights).
    no_day_input: no session-specific parameters at all (LightBeam, Feghhi et al. 2026): no day matrix, bias or
    softsign; patches go through a LayerNorm-Linear-LayerNorm embedding. No streaming path yet.
    inter_ctc_layer k > 0: in training mode forward returns (logits, intermediate logits), the latter from the output
    of block k through the shared final_norm and out layer (intermediate CTC, Lee & Watanabe 2021); the trainer
    weights the two CTC losses (model.transformer.inter_ctc_weight). Eval mode returns logits only.
    pos_bias 'relative': a learned bias per layer and relative distance (clamped to +-(max_rel_dist - 1)), shared by
    all heads, N(0, 1) init, in place of ALiBi (the LightBeam code's T5-style bias). No streaming path yet.
    window_min W > 0: in training mode each forward draws its attention window uniformly from [W, window] patch
    frames (LightBeam samples 5-20 s of left context per batch); eval mode always uses window.
    embed_dropout: dropout rate on the frame embeddings entering the first block (default: dropout). LightBeam drops
    only there, at its input-dropout rate, and not on the raw bins (set input_dropout 0 to match).
    ffn_dropout: also drop after the feed-forward activation. init 'torch': keep PyTorch's default initialization
    (LightBeam) instead of Xavier-uniform weights and zero biases.
    '''

    def __init__(self,
                 neural_dim,
                 n_days,
                 n_classes,
                 d_model=512,
                 n_layers=8,
                 n_heads=8,
                 ffn_mult=4,
                 dropout=0.2,
                 input_dropout=0.0,
                 patch_size=0,
                 patch_stride=0,
                 window=64,
                 mask_token=False,
                 lookahead_frames=0,
                 block_type='standard',
                 conv_kernel=9,
                 drop_path=0.0,
                 shared_input=False,
                 inter_ctc_layer=0,
                 no_day_input=False,
                 pos_bias='alibi',
                 window_min=0,
                 max_rel_dist=200,
                 embed_dropout=None,
                 ffn_dropout=False,
                 init='xavier',
                 ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.window = window
        if not 0 <= window_min <= window:
            raise ValueError("window_min must be in [0, window]")
        self.window_min = window_min
        if pos_bias not in ('alibi', 'relative'):
            raise ValueError(f"Unknown pos_bias: {pos_bias}")
        self.pos_bias, self.max_rel_dist = pos_bias, max_rel_dist
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames must be >= 0")
        self.lookahead_frames = lookahead_frames
        # Outputs near a trial's end attend to later frames, so padded batches need each trial's length
        self.uses_frame_lens = lookahead_frames > 0

        # Day-specific input layers, identical to GRUDecoder (same parameter names and init)
        self.day_layer_activation = nn.Softsign()
        self.shared_input = shared_input
        self.no_day_input = no_day_input
        if shared_input and no_day_input:
            raise ValueError("shared_input and no_day_input are exclusive")
        if no_day_input:
            pass
        elif shared_input:
            self.day_weight_shared = nn.Parameter(torch.eye(self.neural_dim))
        else:
            self.day_weights = nn.ParameterList(
                [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
            )
        if not no_day_input:
            self.day_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
            )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim * (self.patch_size if self.patch_size > 0 else 1)
        if no_day_input:
            self.input_proj = nn.Sequential(nn.LayerNorm(self.input_size), nn.Linear(self.input_size, d_model),
                                            nn.LayerNorm(d_model))
        else:
            self.input_proj = nn.Linear(self.input_size, d_model)
        self.input_drop = nn.Dropout(dropout if embed_dropout is None else embed_dropout)
        if init not in ('xavier', 'torch'):
            raise ValueError(f"Unknown init: {init}")
        if block_type not in ('standard', 'conv', 'conformer'):
            raise ValueError(f"Unknown block_type: {block_type}")
        self.block_type = block_type
        rates = [drop_path * i / max(n_layers - 1, 1) for i in range(n_layers)]
        if block_type == 'conformer':
            self.layers = nn.ModuleList([ConformerBlock(d_model, n_heads, ffn_mult, dropout, conv_kernel, r)
                                         for r in rates])
        else:
            self.layers = nn.ModuleList([Block(d_model, n_heads, ffn_mult, dropout, r,
                                               conv_kernel if block_type == 'conv' else 0, ffn_dropout)
                                         for r in rates])
        if not 0 <= inter_ctc_layer < n_layers:
            raise ValueError("inter_ctc_layer must be in [0, n_layers)")
        self.inter_ctc_layer = inter_ctc_layer
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_classes)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)
        if pos_bias == 'relative':
            self.rel_bias = nn.Parameter(torch.randn(n_layers, 2 * max_rel_dist - 1))

        for name, param in self.named_parameters() if init == 'xavier' else ():
            if 'day_' in name or name == 'rel_bias':
                continue
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)

        # Training-time time masking (Feghhi et al. 2025, Transformer variant): masked frames get a learned
        # embedding. Zero-filled inputs would leave exactly constant rows, whose LayerNorm backward scales
        # gradients by 1/sqrt(eps) per norm and overflows through the stack. Unused at inference.
        if mask_token:
            self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def front_end(self, x, day_idx, bin_mask=None, mask_fill=None):
        '''Day-specific layer, dropout, and patching; same computation as GRUDecoder.forward (incl. masked bins).'''
        if self.no_day_input:
            if self.input_dropout > 0:
                x = self.day_layer_dropout(x)
            if bin_mask is not None:
                x = torch.where(bin_mask.unsqueeze(-1), mask_fill.to(x.dtype), x)
            return patch_frames(x, self.patch_size, self.patch_stride) if self.patch_size > 0 else x
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        if self.shared_input:
            x = torch.matmul(x, self.day_weight_shared) + day_biases
        else:
            day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
            x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)
        if bin_mask is not None:
            x = torch.where(bin_mask.unsqueeze(-1), mask_fill.to(x.dtype), x)
        if self.patch_size > 0:
            x = patch_frames(x, self.patch_size, self.patch_stride)
        return x

    def attention_bias(self, steps, device, future=0, window=None, layer=0):
        '''[1, H, T, T] additive mask: -slope * |distance| (or the layer's learned relative bias, [1, 1, T, T])
        inside [t - window + 1, t + future], -inf outside.'''
        position = torch.arange(steps, device=device)
        offset = position[:, None] - position[None, :]
        distance = offset.float()
        allowed = (distance >= -future) & (distance < (self.window if window is None else window))
        if self.pos_bias == 'relative':
            index = offset.clamp(1 - self.max_rel_dist, self.max_rel_dist - 1) + self.max_rel_dist - 1
            return torch.where(allowed, self.rel_bias[layer][index].float(),
                               torch.full_like(distance, float('-inf')))[None, None]
        distance = distance.abs()
        # torch.where (not clamp + masked_fill) keeps this graph compilable by torch.compile/inductor
        return torch.where(allowed, -self.slopes[:, None, None] * distance,
                           torch.full_like(distance, float('-inf'))).unsqueeze(0)

    def forward(self, x, day_idx, states=None, return_state=False, frame_mask=None, frame_lens=None,
                bin_mask=None, mask_fill=None, return_hidden=False):
        '''frame_mask: optional [B, frames] bool; True frames are replaced by the learned mask token.
        frame_lens: optional [B] valid output frames per trial (look-ahead models): later frames are never attended to.'''
        if states is not None or return_state:
            raise NotImplementedError("TransformerDecoder streams through stream_step()")
        h = self.input_proj(self.front_end(x, day_idx, bin_mask, mask_fill))
        if frame_mask is not None:
            if not hasattr(self, 'mask_token'):
                raise ValueError("frame_mask needs a model built with mask_token=True")
            h = torch.where(frame_mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        h = self.input_drop(h)
        # The memory-efficient attention backward needs the bias row stride to be a multiple of 4 (it fails for e.g.
        # 403 frames, also when torch.compile fuses the bias), so run on a frame count padded to a multiple of 8.
        # Padded frames come last: causal layers never see them, and the last layer's look-ahead masks them.
        steps = h.shape[1]
        h = F.pad(h, (0, 0, 0, (steps + 7) // 8 * 8 - steps))
        # a tensor (not a Python int), so torch.compile does not recompile for every sampled window
        window = (torch.randint(self.window_min, self.window + 1, (), device=h.device)
                  if self.training and self.window_min else None)
        relative = self.pos_bias == 'relative'
        biases = [self.attention_bias(h.shape[1], h.device, window=window, layer=i)
                  for i in range(self.n_layers if relative else 1)] * (1 if relative else self.n_layers)
        if self.lookahead_frames:
            last = self.attention_bias(h.shape[1], h.device, future=self.lookahead_frames, window=window,
                                       layer=self.n_layers - 1)
            lens = frame_lens.to(h.device) if frame_lens is not None else torch.full((h.shape[0],), steps, device=h.device)
            keys = torch.arange(h.shape[1], device=h.device)[None, :] < lens[:, None]
            biases[-1] = torch.where(keys[:, None, None, :], last, torch.full_like(last, float('-inf')))
        inter = None
        for i, layer in enumerate(self.layers):
            h = layer(h, biases[i])
            if self.training and i + 1 == self.inter_ctc_layer:
                inter = h
        h = h[:, :steps]
        h = self.final_norm(h)
        if return_hidden:
            return h
        if inter is not None:
            return self.out(h), self.out(self.final_norm(inter[:, :steps]))
        return self.out(h)

    # -- streaming (eval mode) ------------------------------------------------------------------
    def initial_state(self):
        return [None] * self.n_layers

    def stream_step(self, patch, cache):
        '''One patch frame [1, 1, input_size] (day layer and patching already applied).'''
        if self.lookahead_frames:
            raise NotImplementedError("No streaming path for the look-ahead Transformer yet")
        if self.block_type != 'standard':
            raise NotImplementedError("No streaming path for convolution blocks yet")
        if self.no_day_input:
            raise NotImplementedError("No streaming path without a day input layer yet")
        if self.pos_bias != 'alibi':
            raise NotImplementedError("No streaming path for the learned relative bias yet")
        h = self.input_proj(patch)
        keep = self.window - 1
        new_cache = []
        for layer, kv in zip(self.layers, cache):
            q, k, v = layer.attn.split(layer.norm1(h))
            if kv is not None:
                k = torch.cat((kv[0], k), dim=2)
                v = torch.cat((kv[1], v), dim=2)
            distance = torch.arange(k.shape[2] - 1, -1, -1, device=h.device)
            bias = (-self.slopes[:, None] * distance[None, :])[None, :, None, :]
            h = h + layer.attn.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype)))
            h = h + layer.ffn(layer.norm2(h))
            new_cache.append((k[:, :, -keep:], v[:, :, -keep:]) if keep > 0 else None)
        return self.out(self.final_norm(h)), new_cache
