"""Install causal_la0 / causal_la4 checkpoint archives copied from the original training machine.

Drop the zips in model_training/trained_models/_incoming/ and run:

    cd model_training && ../.venv/bin/python install_reference_checkpoints.py

Lands them in results/ — NOT model_training/trained_models/. The split is deliberate:

    results/                          reference checkpoints: finished, authenticated, immutable
    model_training/trained_models/    runs this machine is producing (may be mid-training)

benchmark/stability.py already defaults to results/causal_la0/checkpoint/val_metrics.pkl, and
writing into trained_models/causal_la0/ would collide with an in-flight training run that owns
that directory.

Accepts a .zip or an already-unzipped directory, tolerates any nesting, and rebuilds the
canonical layout the loaders expect:

    results/causal_laN/
        training_log
        train_val_trials.json
        checkpoint/{args.yaml, best_checkpoint, val_metrics.pkl}
"""
import hashlib
import pathlib
import pickle
import shutil
import sys
import tempfile
import zipfile

TRAINED = pathlib.Path(__file__).parent.parent / "results"          # reference checkpoints live here
INCOMING = pathlib.Path(__file__).parent / "trained_models" / "_incoming"
EXPECTED_LOOKAHEAD = {"causal_la0": 0, "causal_la4": 4}
EXPECTED_PER = {"causal_la0": 0.1004, "causal_la4": 0.1021}


def find_one(root, name):
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def install(src_path):
    """src_path is either a .zip archive or an already-unzipped checkpoint directory."""
    stem = src_path.stem.lower() if src_path.suffix == ".zip" else src_path.name.lower()
    tag = next((t for t in EXPECTED_LOOKAHEAD if t in stem.replace("-", "_")), None)
    if tag is None:
        print(f"  SKIP {src_path.name}: name does not contain causal_la0 or causal_la4")
        return False

    dest = TRAINED / tag
    if dest.exists():
        print(f"  SKIP {src_path.name}: {dest.name}/ already exists (delete it to reinstall)")
        return False

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        if src_path.suffix == ".zip":
            with zipfile.ZipFile(src_path) as z:
                z.extractall(tmp)
            staged = tmp
        else:
            staged = src_path  # copy out of the directory, leave the original intact

        ckpt = find_one(staged, "best_checkpoint")
        if ckpt is None:
            print(f"  FAIL {src_path.name}: no best_checkpoint found anywhere inside it")
            return False

        (dest / "checkpoint").mkdir(parents=True)
        shutil.copy2(str(ckpt), dest / "checkpoint" / "best_checkpoint")
        for name, sub in [("args.yaml", "checkpoint"),
                          ("val_metrics.pkl", "checkpoint"),
                          ("training_log", ""),
                          ("train_val_trials.json", "")]:
            found = find_one(staged, name)
            if found is not None:
                shutil.copy2(str(found), (dest / sub / name) if sub else (dest / name))
            else:
                print(f"    note: {name} not present")

    verify(tag, dest)
    return True


def verify(tag, dest):
    print(f"  installed {dest.relative_to(TRAINED.parent)}")
    ck = dest / "checkpoint" / "best_checkpoint"
    print(f"    best_checkpoint   {ck.stat().st_size / 1e6:.1f} MB")
    print(f"    sha256[:16]       {hashlib.sha256(ck.read_bytes()).hexdigest()[:16]}")

    args_p = dest / "checkpoint" / "args.yaml"
    if args_p.exists():
        from omegaconf import OmegaConf
        a = OmegaConf.load(args_p)
        d = a["dataset"]["data_transforms"]
        la = d.get("smooth_lookahead", "<ABSENT -> symmetric, NON-CAUSAL>")
        want = EXPECTED_LOOKAHEAD[tag]
        ok = "OK" if la == want else f"MISMATCH (expected {want})"
        print(f"    smooth_lookahead  {la}  {ok}")
        print(f"    smooth_kernel_std {d.get('smooth_kernel_std')}   seed {a.get('seed')}   "
              f"batches {a.get('num_training_batches')}")

    log = dest / "training_log"
    if log.exists():
        best = [l for l in log.read_text(errors="ignore").splitlines() if "New best test PER" in l]
        if best:
            per = float(best[-1].split("-->")[-1].strip())
            delta = abs(per - EXPECTED_PER[tag])
            print(f"    best val PER      {per:.4f}  (published {EXPECTED_PER[tag]:.4f}, "
                  f"{'OK' if delta < 0.0005 else f'DIFFERS by {delta:.4f}'})")

    vm = dest / "checkpoint" / "val_metrics.pkl"
    if vm.exists():
        with open(vm, "rb") as f:
            m = pickle.load(f)
        logits = m.get("logits")
        print(f"    val_metrics.pkl   avg_PER {m.get('avg_PER', float('nan')):.5f}  "
              f"avg_loss {m.get('avg_loss', float('nan')):.4f}")
        if logits is not None:
            # 'logits' is a list of PER-DAY BATCHES [n_trials, max_frames, n_classes], not
            # per-trial arrays. stability.py::load_val_logits unpacks them using 'n_time_steps'
            # (which holds adjusted_lens, i.e. FRAME counts) to trim each trial's padding.
            n_trials = sum(b.shape[0] for b in logits)
            print(f"                      {len(logits)} day-batches -> {n_trials} trials "
                  f"({'OK, full val split' if n_trials == 1426 else 'EXPECTED 1426'})")
            print(f"                      Group 1 runs off these; the acoustic model never re-runs")
        else:
            print("                      no 'logits' key — C13 will need to regenerate them")


def main():
    if not INCOMING.exists():
        print(f"missing drop zone {INCOMING}")
        return 1
    sources = sorted(INCOMING.glob("*.zip")) + sorted(
        d for d in INCOMING.iterdir() if d.is_dir() and any(t in d.name.lower() for t in EXPECTED_LOOKAHEAD)
    )
    if not sources:
        print(f"nothing to install in {INCOMING}")
        print("  expected either *.zip archives, or causal_la0/ and causal_la4/ directories")
        return 1
    print(f"found {len(sources)} source(s): {', '.join(s.name for s in sources)}\n")
    n = sum(install(s) for s in sources)
    print(f"\n{n} installed. Originals left in _incoming/ — delete them once you are satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
