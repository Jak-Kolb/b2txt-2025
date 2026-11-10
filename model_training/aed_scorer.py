"""A non-CTC second scorer on a frozen acoustic encoder (prepared 2026-11-10, NOT yet trained; brainstorm/PLAN.md
step 4).

A small attention decoder reads the frozen encoder's features (what its output layer reads) and predicts the phoneme
sequence one label at a time, each conditioned on the labels before it. CTC scores every frame independently, so this
is acoustic evidence with a different error pattern; the deadline ranker gets log P(candidate's phonemes | features
available at the deadline) as a feature (learning/live_eval/llm_live/aed_features.py). No end token: every training
target is a prefix, so prefix scores are what the head models.

Training examples: a training trial cut at a between-word point (data_prep/word_boundaries.py: bin, label count) plus
2-24 output frames, or the whole trial. The spread stands in for the gap between the bidirectional aligner's word end
and the frame at which a causal decider decides (its own later ' | ' emission + D). The encoder is frozen and run in
eval mode on input with the trainer's white-noise / offset augmentation.
Known limit: the encoder was trained on these trials, so its features are cleaner here than on new data and the head
may over-trust them; out-of-fold encoders (dataset.train_fold, step 5) remove that.
usage: aed_scorer.py --encoder REGISTERED_NAME --out DIR [--steps 20000] [--batch 32] [--device cuda] [--limit N]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model_training"))
from model_training.benchmark.common import load_model, smooth_features

N_TOKENS = 41                                   # 0 = start token on the input side; 1..40 phonemes (40 = ' | ')


def positions(length, dim, device):
    pos = torch.arange(length, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    out = torch.zeros(length, dim, device=device)
    out[:, 0::2], out[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return out


class AEDHead(nn.Module):
    def __init__(self, enc_dim, d_model=256, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(enc_dim, d_model)
        self.embed = nn.Embedding(N_TOKENS, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, N_TOKENS)

    def forward(self, memory, memory_lens, labels):
        """memory [B, T, enc_dim], memory_lens [B], labels [B, L] (1..40, 0 padded) -> log-probs [B, L, N_TOKENS] of
        each label given the labels before it."""
        mem = self.proj(memory) + positions(memory.shape[1], self.proj.out_features, memory.device)
        tokens = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=1)
        x = self.embed(tokens) + positions(tokens.shape[1], self.proj.out_features, memory.device)
        causal = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=memory.device), 1)
        pad = torch.arange(memory.shape[1], device=memory.device)[None, :] >= memory_lens[:, None]
        h = self.decoder(x, mem, tgt_mask=causal, memory_key_padding_mask=pad)
        return self.out(self.norm(h)).log_softmax(-1)

    def score(self, memory, memory_lens, labels):
        """Sum of label log-probs per row (padding ignored)."""
        logp = self(memory, memory_lens, labels)
        picked = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (picked * (labels > 0)).sum(1)


def load_encoder(name, device):
    entry = yaml.safe_load((REPO / "harness/registry/acoustic" / f"{name}.yaml").read_text())
    model, args, _ = load_model(str(REPO / entry["checkpoint_dir"]), device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, args


@torch.no_grad()
def encode(model, args, features, day, device, noise=False):
    """Frozen-encoder features [frames, enc_dim] of one trial ([T, C] raw features)."""
    x = features.to(device).unsqueeze(0)
    if noise:
        t = args["dataset"]["data_transforms"]
        x = x + torch.randn_like(x) * t["white_noise_std"] + torch.randn(1, 1, x.shape[2], device=device) * t["constant_offset_std"]
    x = smooth_features(x, args, device)
    return model(x, torch.tensor([day], device=device), return_hidden=True)[0].float()


def load_trials(args, boundaries, limit):
    trials = []
    for day, session in enumerate(args["dataset"]["sessions"]):
        path = REPO / "data/hdf5_data_final" / session / "data_train.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as h:
            for key in sorted(h.keys())[: limit or None]:
                g = h[key]
                ids = g["seq_class_ids"][: int(g.attrs["seq_len"])].astype(np.int64)
                trials.append(dict(day=day, features=torch.from_numpy(g["input_features"][:]), ids=torch.from_numpy(ids),
                                   bounds=boundaries.get(session, {}).get(str(int(key.split("_")[-1])), [])))
    return trials


def example(trial, hidden, rng, patch, stride):
    """(features, labels) for a random prefix of the trial, or the whole trial."""
    choice = rng.integers(len(trial["bounds"]) + 1)
    if choice == len(trial["bounds"]):
        return hidden, trial["ids"]
    cut, n_labels = trial["bounds"][choice]
    frames = min(hidden.shape[0], max(1, (cut - patch) // stride + 1 + int(rng.integers(2, 25))))
    return hidden[:frames], trial["ids"][:n_labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundaries", default=str(REPO / "results/word_boundaries/bigru768_pall_fall_20k_s10.json"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="trials per session (smoke)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    encoder, args = load_encoder(a.encoder, device)
    patch, stride = args["model"]["patch_size"], args["model"]["patch_stride"]
    trials = load_trials(args, json.loads(Path(a.boundaries).read_text()), a.limit)
    head = AEDHead(encoder.out.in_features).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    (out / "args.json").write_text(json.dumps(dict(vars(a), enc_dim=encoder.out.in_features, n_trials=len(trials))))
    print(f"{len(trials)} training trials; head {sum(p.numel() for p in head.parameters()):,} parameters", flush=True)
    t0 = time.time()
    for step in range(a.steps):
        mems, labs = zip(*(example(t, encode(encoder, args, t["features"], t["day"], device, noise=True), rng, patch, stride)
                           for t in (trials[i] for i in rng.integers(len(trials), size=a.batch))))
        memory = nn.utils.rnn.pad_sequence(mems, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labs, batch_first=True).to(device)
        lens = torch.tensor([m.shape[0] for m in mems], device=device)
        logp = head(memory, lens, labels)
        loss = F.nll_loss(logp.transpose(1, 2), labels, ignore_index=0)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step(); sched.step()
        if step % max(1, a.steps // 100) == 0 or step == a.steps - 1:
            print(f"step {step}: loss per label {loss.item():.3f} ({time.time() - t0:.0f} s)", flush=True)
    torch.save(dict(state_dict=head.state_dict(), encoder=a.encoder, enc_dim=encoder.out.in_features), out / "head.pt")
    print(f"saved {out / 'head.pt'}", flush=True)


if __name__ == "__main__":
    main()
