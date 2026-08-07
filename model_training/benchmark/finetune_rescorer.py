"""C29b — domain-adapt the rescorer on the TRAINING transcriptions.

Why
---
The rescorer is currently pure web text (OpenWebText-era GPT-2), while the prompts are everyday
conversational sentences from one participant's protocol. C28-A already showed how much in-domain
signal is worth here: a trigram over these same 8,072 sentences moved WER 17 points as a re-ranker
when the base LM was a 1-gram. As an *n-gram* it is now worthless (0.00 pts against the 4-gram,
§1.11) because a well-estimated 4-gram subsumes it. A fine-tuned neural LM is a different
proposition: it can pick up register, sentence length and phrasing without the coverage cliff that
sank the n-gram.

Honesty constraints
-------------------
* TRAIN transcriptions only. val-dev is scored, never trained on; val-test is untouched entirely.
* **6.94 % of val-dev sentences appear VERBATIM in train**, so fine-tuning will memorise them.
  `neural_rescore.py` already splits seen/unseen — the unseen row is the reportable number, and it
  is the one to watch for an inflated headline here.
* 8,072 sentences is small enough that catastrophic forgetting is the real risk: low LR, few
  epochs, and the general LM's knowledge is what supplies coverage.

Text format matches `neural_rescore.score_sentences` exactly (EOS as BOS, leading space,
lowercase), or the fine-tune teaches a distribution the scorer never queries.

    cd model_training && ../.venv/bin/python -m benchmark.finetune_rescorer --model gpt2
"""
import argparse
import math
import pathlib
import random
import re
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
_PUNCT = re.compile(r"[^\w\s']")


def load_sentences(path):
    """Normalise the way the decoder's candidates are normalised: lowercase, punctuation stripped."""
    out = []
    for line in pathlib.Path(path).read_text().split("\n"):
        s = " ".join(_PUNCT.sub(" ", line.lower()).split())
        if s:
            out.append(s)
    return out


def make_batches(sents, tok, batch_size, shuffle=True):
    idx = list(range(len(sents)))
    if shuffle:
        random.shuffle(idx)
    bos = tok.eos_token_id
    for start in range(0, len(idx), batch_size):
        chunk = [sents[i] for i in idx[start:start + batch_size]]
        # Same encoding as scoring: [EOS] + " sentence" + [EOS]
        enc = [[bos] + tok.encode(" " + s) + [bos] for s in chunk]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), bos, dtype=torch.long)
        labels = torch.full((len(enc), width), -100, dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, :len(e)] = torch.tensor(e)
            labels[r, 1:len(e)] = torch.tensor(e[1:])   # predict everything after the BOS
        yield ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--sentences", default=str(REPO / "results/train_sentences.txt"))
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--holdout", type=int, default=400, help="train sentences held out for ppl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = a.out or str(REPO / f"results/rescorer_ft_{a.model.replace('/', '_')}")
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sents = load_sentences(a.sentences)
    random.shuffle(sents)
    held, train = sents[: a.holdout], sents[a.holdout:]
    print(f"{len(train)} train / {len(held)} held-out sentences (from TRAIN split only)")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).to(device)
    print(f"{a.model}: {sum(p.numel() for p in model.parameters())/1e6:.0f} M params on {device}")

    @torch.inference_mode()
    def heldout_ppl():
        model.eval()
        tot_lp = tot_tok = 0
        for ids, labels in make_batches(held, tok, a.batch_size, shuffle=False):
            ids, labels = ids.to(device), labels.to(device)
            loss = model(ids, labels=labels).loss
            n = int((labels != -100).sum())
            tot_lp += float(loss) * n
            tot_tok += n
        return math.exp(tot_lp / max(tot_tok, 1))

    print(f"held-out ppl before: {heldout_ppl():.2f}")

    steps_per_epoch = math.ceil(len(train) / a.batch_size)
    total_steps = int(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.1)

    model.train()
    step = 0
    t0 = time.time()
    done = False
    while not done:
        for ids, labels in make_batches(train, tok, a.batch_size):
            ids, labels = ids.to(device), labels.to(device)
            loss = model(ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"  step {step}/{total_steps}  loss {float(loss):.3f}  "
                      f"{time.time()-t0:.0f}s", flush=True)
            if step >= total_steps:
                done = True
                break

    ppl = heldout_ppl()
    print(f"held-out ppl after:  {ppl:.2f}   ({time.time()-t0:.0f}s total)")

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")
    print(f"\nrescore with:  ../.venv/bin/python -m benchmark.neural_rescore --model {out_dir} \\\n"
          f"                 --gammas 0.5,0.25,0.1 --alphas 0.5,1,1.5,2 --betas 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
