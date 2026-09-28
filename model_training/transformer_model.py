import torch
from torch import nn
import torch.nn.functional as F


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


class Block(nn.Module):
    '''Pre-norm Transformer block.'''

    def __init__(self, d_model, n_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Linear(ffn_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        x = x + self.drop(self.attn(self.norm1(x), bias))
        return x + self.drop(self.ffn(self.norm2(x)))


class TransformerDecoder(nn.Module):
    '''
    Causal Transformer acoustic model with GRUDecoder's day-specific input layers and patching.

    Each layer attends to itself and the previous (window - 1) patch frames. Streaming keeps a
    per-layer key/value cache of that window, which reproduces the offline forward exactly
    (up to floating point) because earlier frames never change under causal attention.
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

        # Day-specific input layers, identical to GRUDecoder (same parameter names and init)
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim * (self.patch_size if self.patch_size > 0 else 1)
        self.input_proj = nn.Linear(self.input_size, d_model)
        self.input_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([Block(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_classes)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

        for name, param in self.named_parameters():
            if 'day_' in name:
                continue
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)

    def front_end(self, x, day_idx):
        '''Day-specific layer, dropout, and patching; same computation as GRUDecoder.forward.'''
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)
        if self.patch_size > 0:
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride).squeeze(2).permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
        return x

    def attention_bias(self, steps, device):
        '''[1, H, T, T] additive mask: -slope * distance inside the causal window, -inf outside.'''
        position = torch.arange(steps, device=device)
        distance = (position[:, None] - position[None, :]).float()
        allowed = (distance >= 0) & (distance < self.window)
        # torch.where (not clamp + masked_fill) keeps this graph compilable by torch.compile/inductor
        return torch.where(allowed, -self.slopes[:, None, None] * distance,
                           torch.full_like(distance, float('-inf'))).unsqueeze(0)

    def forward(self, x, day_idx, states=None, return_state=False):
        if states is not None or return_state:
            raise NotImplementedError("TransformerDecoder streams through stream_step()")
        h = self.input_drop(self.input_proj(self.front_end(x, day_idx)))
        bias = self.attention_bias(h.shape[1], h.device)
        for layer in self.layers:
            h = layer(h, bias)
        return self.out(self.final_norm(h))

    # -- streaming (eval mode) ------------------------------------------------------------------
    def initial_state(self):
        return [None] * self.n_layers

    def stream_step(self, patch, cache):
        '''One patch frame [1, 1, input_size] (day layer and patching already applied).'''
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
