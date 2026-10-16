"""Masked (BERT-style) pretraining of the acoustic encoder on unlabelled neural data.

Spans of 20 ms bins are hidden after the day layer (replaced by a learned mask vector), so overlapping patches
cannot reveal them. From each output frame's hidden features, a per-participant linear head predicts that frame's
newest `patch_stride` bins of the smoothed input features. The loss is MSE on hidden, valid bins only. Labels in the
data are never read. Causal models must forecast hidden bins from the past (plus look-ahead frames, if any);
models with future context interpolate.

The checkpoint is loaded for fine-tuning with init_shared_from_checkpoint (+ init_shared_day_layers to carry
day layers over to sessions with the same name).

    python ssl_pretrain.py CONFIG.yaml
CONFIG: a training config (model/dataset sections as for train_model.py; dataset_dir = an SSL pool) plus
    ssl: {mask_ratio, span_min_bins, span_max_bins, n_batches, lr_max, lr_min, warmup, val_fraction, log_every,
          val_every, grad_clip, amp (bf16 autocast on CUDA; default true), objective (mse | gauss_nll | contrastive),
          temperature (contrastive; default 0.1)}
"""
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_augmentations import gauss_smooth
from dataset import BrainToTextDataset, train_test_split_indicies
from model_factory import build_model


def participant_of(session):
    return 1 if session.startswith("t12.") else 0          # t15 and t15v share a feature layout


def draw_bin_mask(lengths, T, ratio, span_min, span_max, generator):
    """[B, T] bool: spans of U[span_min, span_max] bins inside each trial until >= ratio of its valid bins are hidden."""
    mask = torch.zeros(len(lengths), T, dtype=torch.bool)
    for b, length in enumerate(lengths.tolist()):
        target = int(ratio * length)
        while int(mask[b, :length].sum()) < target:
            width = int(torch.randint(span_min, span_max + 1, (1,), generator=generator))
            start = int(torch.randint(0, max(1, length - width + 1), (1,), generator=generator))
            mask[b, start:start + width] = True
    return mask


class SSLHeads(torch.nn.Module):
    """Learned mask vector plus the objective's readouts (per participant where the feature layout matters).

    objective 'mse': hidden -> the frame's newest `stride` bins (squared error).
    objective 'gauss_nll': hidden -> mean and log-variance of those bins (Gaussian likelihood: noise-aware).
    objective 'contrastive': hidden -> a D-dim prediction, compared (cosine / temperature) with a learned embedding
        of the frame's true hidden bins against the other hidden frames of the same trial (InfoNCE, wav2vec-style).
    """

    def __init__(self, bin_dim, hidden, stride, n_features, objective="mse", n_participants=2, proj_dim=256):
        super().__init__()
        self.stride, self.n_features, self.objective = stride, n_features, objective
        self.mask_fill = torch.nn.Parameter(torch.randn(bin_dim) * 0.02)
        if objective == "contrastive":
            self.context = torch.nn.Linear(hidden, proj_dim)
            self.target = torch.nn.ModuleList([torch.nn.Linear(stride * n_features, proj_dim) for _ in range(n_participants)])
        else:
            width = stride * n_features * (2 if objective == "gauss_nll" else 1)
            self.readout = torch.nn.ModuleList([torch.nn.Linear(hidden, width) for _ in range(n_participants)])

    @staticmethod
    def pick(outputs, participant):
        out = torch.stack(outputs, dim=1)                                                # [B, P, ...]
        return out[torch.arange(out.shape[0], device=out.device), participant]

    def predict(self, hidden, participant):
        out = self.pick([head(hidden) for head in self.readout], participant)
        return out.view(hidden.shape[0], hidden.shape[1], self.stride, -1)               # [B, F, s, C or 2C]


def ssl_loss(model, heads, features, n_bins, days, participant, bin_mask, args):
    """Loss over hidden, valid bins that some frame predicts as one of its newest `stride` bins.

    Returns (loss, number of scored bins or frames, dict of diagnostics)."""
    patch, stride = int(args.model.patch_size), int(args.model.patch_stride)
    frame_lens = ((n_bins - patch) // stride + 1).clamp(min=0)
    kwargs = dict(frame_lens=frame_lens) if getattr(model, "uses_frame_lens", False) else {}
    hidden = model(features, days, bin_mask=bin_mask, mask_fill=heads.mask_fill, return_hidden=True, **kwargs)
    n_frames = hidden.shape[1]
    bins = (torch.arange(n_frames, device=features.device)[:, None] * stride + patch - stride
            + torch.arange(stride, device=features.device)[None, :])                     # [F, s] newest bins
    target = features[:, bins].float()                                                   # [B, F, s, C]
    hidden_bins = bin_mask.to(features.device)[:, bins]                                  # [B, F, s]
    valid = (torch.arange(n_frames, device=features.device)[None, :] < frame_lens[:, None])
    if heads.objective == "contrastive":
        temperature = float(args.ssl.get("temperature", 0.1))
        frames = hidden_bins.all(dim=-1) & valid                                          # every newest bin hidden
        c = torch.nn.functional.normalize(heads.context(hidden).float(), dim=-1)
        z = heads.pick([g(target.flatten(2)) for g in heads.target], participant)
        z = torch.nn.functional.normalize(z.float(), dim=-1)
        losses, correct, count = [], 0, 0
        for b in range(features.shape[0]):
            idx = torch.nonzero(frames[b]).squeeze(1)
            if idx.numel() < 2:
                continue
            if idx.numel() > 64:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:64]]
            logits = c[b, idx] @ z[b, idx].T / temperature                               # positives on the diagonal
            labels = torch.arange(idx.numel(), device=idx.device)
            losses.append(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(dim=1) == labels).sum()); count += idx.numel()
        if not count:
            return hidden.sum() * 0.0, 0, dict(acc=float("nan"))
        return torch.stack(losses).sum() / count, count, dict(acc=correct / count)
    weight = (hidden_bins & valid[:, :, None]).float()
    pred = heads.predict(hidden, participant).float()
    if heads.objective == "gauss_nll":
        mean, logvar = pred[..., :heads.n_features], pred[..., heads.n_features:].clamp(-8, 8)
        err = (0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))).mean(dim=-1)
        mse = ((mean - target) ** 2).mean(dim=-1)
    else:
        err = mse = ((pred - target) ** 2).mean(dim=-1)
    total = weight.sum().clamp(min=1)
    return (err * weight).sum() / total, int(weight.sum()), dict(mse=float(((mse * weight).sum() / total).detach()))


def main(config_path):
    args = OmegaConf.load(config_path)
    ssl = args.ssl
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "checkpoint").mkdir()
    OmegaConf.save(args, out_dir / "checkpoint" / "args.yaml")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s",
                        handlers=[logging.FileHandler(out_dir / "ssl_log"), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("ssl")
    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sessions = list(args.dataset.sessions)
    paths = [os.path.join(args.dataset.dataset_dir, s, "data_train.hdf5") for s in sessions]
    train_trials, val_trials = train_test_split_indicies(paths, test_percentage=float(ssl.val_fraction),
                                                         seed=int(args.dataset.seed))
    common = dict(days_per_batch=None, must_include_days=None, random_seed=int(args.dataset.seed), feature_subset=None)
    train_set = BrainToTextDataset(trial_indicies=train_trials, split="train", n_batches=int(ssl.n_batches),
                                   batch_size=int(args.dataset.batch_size),
                                   **{**common, "days_per_batch": int(args.dataset.days_per_batch)})
    val_set = BrainToTextDataset(trial_indicies=val_trials, split="test", n_batches=None,
                                 batch_size=int(args.dataset.batch_size), **common)
    train_loader = DataLoader(train_set, batch_size=None, shuffle=False,
                              num_workers=int(args.dataset.num_dataloader_workers), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False, num_workers=0)

    model = build_model(args).to(device)
    with torch.no_grad():
        probe = model(torch.zeros(1, int(args.model.patch_size) + int(args.model.patch_stride), int(args.model.n_input_features),
                                  device=device), torch.tensor([0], device=device), return_hidden=True)
    bin_dim = int(args.model.get("input_bottleneck", 0) or args.model.n_input_features)
    heads = SSLHeads(bin_dim, probe.shape[-1], int(args.model.patch_stride), int(args.model.n_input_features),
                     objective=str(ssl.get("objective", "mse"))).to(device)
    params = list(model.parameters()) + list(heads.parameters())
    optim = torch.optim.AdamW(params, lr=float(ssl.lr_max), betas=(0.9, 0.999), weight_decay=float(args.weight_decay))
    total, warmup, floor = int(ssl.n_batches), int(ssl.warmup), float(ssl.lr_min) / float(ssl.lr_max)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: s / max(1, warmup) if s < warmup else
                                              floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup)))))
    gen = torch.Generator().manual_seed(int(args.seed))
    transforms = args.dataset.data_transforms
    log.info(f"SSL pretraining: {len(sessions)} sessions, {sum(len(v['trials']) for v in train_trials.values())} train / "
             f"{sum(len(v['trials']) for v in val_trials.values())} val trials, hidden {probe.shape[-1]}, device {device}")

    def batch_tensors(batch):
        x = batch["input_features"].to(device).float()
        if transforms.smooth_data:
            x = gauss_smooth(x, device, transforms.smooth_kernel_std, transforms.smooth_kernel_size,
                             lookahead=transforms.get("smooth_lookahead"))
        n_bins = batch["n_time_steps"].to(device).long()
        days = batch["day_indicies"].to(device)
        participant = torch.tensor([participant_of(sessions[int(d)]) for d in batch["day_indicies"]], device=device)
        mask = draw_bin_mask(batch["n_time_steps"].long(), x.shape[1], float(ssl.mask_ratio), int(ssl.span_min_bins),
                             int(ssl.span_max_bins), gen).to(device)
        return x, n_bins, days, participant, mask

    start, step = time.time(), 0
    for step, batch in enumerate(train_loader):
        model.train(); heads.train()
        x, n_bins, days, participant, mask = batch_tensors(batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and bool(ssl.get("amp", True))):
            loss, n, extra = ssl_loss(model, heads, x, n_bins, days, participant, mask, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite SSL loss at batch {step}; stopping without saving a checkpoint")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, float(ssl.grad_clip))
        optim.step(); sched.step()
        if step % int(ssl.log_every) == 0:
            log.info(f"SSL batch {step}: loss {loss.item():.4f} {extra} grad norm {float(grad):.2f} scored {n} "
                     f"lr {sched.get_last_lr()[0]:.2e} ({time.time() - start:.0f} s)")
        if (step + 1) % int(ssl.val_every) == 0 or step + 1 == total:
            model.eval(); heads.eval()
            tot = cnt = 0.0
            extras = {}
            with torch.no_grad():
                for vb in val_loader:
                    x, n_bins, days, participant, mask = batch_tensors(vb)
                    l, n, extra = ssl_loss(model, heads, x, n_bins, days, participant, mask, args)
                    tot += float(l) * n; cnt += n
                    for k, v in extra.items():
                        extras[k] = extras.get(k, 0.0) + v * n
            summary = " ".join(f"{k} {v / max(cnt, 1):.4f}" for k, v in extras.items())
            log.info(f"SSL val after batch {step}: loss {tot / max(cnt, 1):.4f} {summary}")
    torch.save(dict(model_state_dict=model.state_dict(), ssl_state_dict=heads.state_dict(), sessions=sessions,
                    n_batches=step + 1), out_dir / "checkpoint" / "ssl_final")
    (out_dir / "ssl_summary.json").write_text(json.dumps(dict(batches=step + 1, seconds=round(time.time() - start),
                                                              checkpoint=str(out_dir / "checkpoint" / "ssl_final"))))
    log.info(f"Saved {out_dir / 'checkpoint' / 'ssl_final'}")


if __name__ == "__main__":
    main(sys.argv[1])
