"""Retraining: fork a registered model's config, smoke-test it, launch detached, auto-register.

Protocol (.claude/skills/train-model/SKILL.md): fresh output paths, a bounded smoke run of the
actual config first, the source revision/diff/config/question recorded, and completed runs
never modified. The trainer's checkpoint selection (val PER over all enabled val sessions) is
unchanged and recorded as selection data on the registered model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from . import REPO, registry
from .scopes import DATA_DIR
from .store import (DEFAULT_ROOT, git_provenance, pid_alive, replace_json, safe_name, utc_stamp,
                    write_new_json)

MODEL_TRAINING = REPO / "model_training"
TRAINED_MODELS = MODEL_TRAINING / "trained_models"
NAME = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
TRAIN_LINE = re.compile(r"Train batch (\d+): loss: (\S+)")
VAL_LINE = re.compile(r"Val batch (\d+): PER \(avg\): (\S+)")
SELECTION = dict(rule="rnn_trainer best greedy-CTC val PER (ties: val loss) over all enabled val sessions",
                 data=["val-dev", "former-val-test"])


def now():
    return datetime.now(timezone.utc).isoformat()


def trainer_command(config_path):
    return [sys.executable, "-u", "train_model.py", str(config_path)]


def set_state(train_dir, **fields):
    path = Path(train_dir) / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(fields, updated=now())
    replace_json(path, state)
    return state


def fork(base, name, overrides, question, root=DEFAULT_ROOT, registry_root=registry.REGISTRY_DIR,
         models_dir=TRAINED_MODELS, data_dir=DATA_DIR, smoke_batches=50, additions=()):
    """Write the forked full config and its smoke variant into a fresh train directory.

    `overrides` may only change existing keys (typos raise); `additions` explicitly introduce
    new keys, e.g. model.type=transformer and its model.transformer.* settings.
    """
    from omegaconf import OmegaConf
    if not NAME.match(name):
        raise ValueError("name must be 1-60 letters, digits, '_' or '-'")
    entries = registry.load_kind("acoustic", registry_root)
    if name in entries:
        raise FileExistsError(f"Acoustic model {name!r} is already registered")
    if not question or not question.strip():
        raise ValueError("State the question this training run answers (--question)")
    base_entry = entries[base]
    config = OmegaConf.load(registry.repo_path(base_entry["checkpoint_dir"]) / "args.yaml")
    if additions:
        added = OmegaConf.from_dotlist(list(additions))
        clashes = [key for key in additions if OmegaConf.select(config, key.split("=", 1)[0]) is not None]
        if clashes:
            raise ValueError(f"--add keys already exist (use --set): {clashes}")
        config = OmegaConf.merge(config, added)
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))  # unknown keys raise
    stamp = utc_stamp()
    model_dir = Path(models_dir).resolve() / f"{name}_{stamp}"
    train_dir = Path(root) / "train" / f"{stamp}_{safe_name(name)}"
    train_dir.mkdir(parents=True, exist_ok=False)
    if model_dir.exists():
        raise FileExistsError(model_dir)
    config.mode = "train"
    config.output_dir = str(model_dir)
    config.checkpoint_dir = str(model_dir / "checkpoint")
    config.dataset.dataset_dir = str(Path(data_dir).resolve())
    smoke = OmegaConf.merge(config, OmegaConf.create(dict(
        num_training_batches=int(smoke_batches), batches_per_val_step=int(smoke_batches),
        batches_per_train_log=max(1, int(smoke_batches) // 5),
        output_dir=str((train_dir / "smoke_model").resolve()),
        checkpoint_dir=str((train_dir / "smoke_model" / "checkpoint").resolve()))))
    with (train_dir / "config.yaml").open("x") as handle:
        handle.write(OmegaConf.to_yaml(config))
    with (train_dir / "smoke_config.yaml").open("x") as handle:
        handle.write(OmegaConf.to_yaml(smoke))
    write_new_json(train_dir / "request.json", dict(
        name=name, forked_from=base, overrides=list(overrides), additions=list(additions), question=question.strip(),
        smoke_batches=int(smoke_batches), created=now(), model_dir=str(model_dir),
        code=git_provenance(train_dir), selection=SELECTION))
    set_state(train_dir, state="forked", name=name, model_dir=str(model_dir))
    return train_dir


def parse_log(path):
    """Finite-ness and last values of train loss and val PER lines in a training_log."""
    text = Path(path).read_text() if Path(path).exists() else ""
    losses = [float(v) for _, v in TRAIN_LINE.findall(text)]
    pers = [float(v) for _, v in VAL_LINE.findall(text)]
    json_safe = lambda v: v if math.isfinite(v) else str(v)  # state files are strict JSON
    return dict(n_train_lines=len(losses), n_val_lines=len(pers),
                finite=bool(losses) and bool(pers) and all(math.isfinite(v) for v in losses + pers),
                n_nonfinite=sum(not math.isfinite(v) for v in losses + pers),
                last_loss=json_safe(losses[-1]) if losses else None,
                last_val_per=json_safe(pers[-1]) if pers else None,
                best_line="Best avg val PER achieved" in text)


def streaming_check(checkpoint_dir, data_dir, n_trials=2, device="cuda"):
    """Streamed vs offline logits on the first sample trials (skipped for non-causal smoothing)."""
    import torch
    from model_training.benchmark.common import (TrialSpec, configure_fp32_inference, load_model,
                                                 load_trial_features, offline_logits, select_device)
    from model_training.benchmark.streaming_infer import run_streaming_trial
    from .scopes import build_scope
    configure_fp32_inference()
    device = select_device(device)
    model, args, _ = load_model(checkpoint_dir, device)
    if args["dataset"]["data_transforms"].get("smooth_lookahead") is None:
        return dict(skipped="non-causal smoothing")
    sessions = list(args["dataset"]["sessions"])
    errors = []
    for trial in build_scope("val-dev-sample", data_dir)["trials"][:n_trials]:
        spec = TrialSpec(trial.session, sessions.index(trial.session), Path(trial.hdf5_path), trial.trial_key, 0)
        features, _ = load_trial_features(spec)
        streamed = run_streaming_trial(model, args, device, features, spec.day_idx, enable_timing=False)["logits"]
        with torch.inference_mode():
            offline = offline_logits(model, torch.as_tensor(features, device=device).unsqueeze(0),
                                     spec.day_idx, args, device)[0]
        errors.append(float((streamed.float().cpu() - offline.float().cpu()).abs().max()))
    return dict(max_logit_error=max(errors), n_trials=len(errors), passed=bool(max(errors) < 1e-3))


def run_smoke(train_dir, data_dir=DATA_DIR, trainer=trainer_command, device="cuda", timeout_s=3600):
    train_dir = Path(train_dir)
    set_state(train_dir, state="smoke_running")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (train_dir / "smoke.log").open("x") as log:
        done = subprocess.run(trainer((train_dir / "smoke_config.yaml").resolve()), cwd=MODEL_TRAINING,
                              stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout_s)
    checkpoint = train_dir / "smoke_model" / "checkpoint"
    result = dict(exit_code=done.returncode, log=parse_log(train_dir / "smoke_model" / "training_log"),
                  checkpoint=(checkpoint / "best_checkpoint").exists() and (checkpoint / "args.yaml").exists())
    passed = done.returncode == 0 and result["log"]["finite"] and result["checkpoint"]
    if passed:
        result["streaming"] = streaming_check(checkpoint, data_dir, device=device)
        passed = result["streaming"].get("passed", True)
    set_state(train_dir, state="smoke_passed" if passed else "smoke_failed", smoke=result)
    return passed, result


def launch(train_dir, then_bench=None, root=DEFAULT_ROOT):
    """Start the detached supervisor; it survives this CLI and the SSH session."""
    train_dir = Path(train_dir).resolve()
    set_state(train_dir, then_bench=then_bench)
    with (train_dir / "supervise.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "harness", "--root", str(root), "train-supervise",
                                    str(train_dir)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    set_state(train_dir, state="launched", supervisor_pid=process.pid)
    return process.pid


def register(train_dir, registry_root=registry.REGISTRY_DIR):
    """Add the finished checkpoint to the acoustic registry (a new file; never an edit)."""
    import yaml
    train_dir = Path(train_dir)
    request = json.loads((train_dir / "request.json").read_text())
    checkpoint_dir = Path(request["model_dir"]) / "checkpoint"
    if not (checkpoint_dir / "best_checkpoint").exists():
        raise FileNotFoundError(checkpoint_dir / "best_checkpoint")
    checkpoint = str(checkpoint_dir.relative_to(REPO)) if checkpoint_dir.is_relative_to(REPO) else str(checkpoint_dir)
    entry = dict(name=request["name"], checkpoint_dir=checkpoint,
                 description=f"Fork of {request['forked_from']}: {request['question']}",
                 training=dict(config=str(train_dir / "config.yaml"), log=str(Path(request["model_dir"]) / "training_log"),
                               forked_from=request["forked_from"],
                               overrides=request.get("additions", []) + request["overrides"],
                               git_head=request["code"]["head"], git_dirty=request["code"]["dirty"],
                               train_dir=str(train_dir)),
                 selection=request["selection"])
    path = Path(registry_root) / "acoustic" / f"{request['name']}.yaml"
    with path.open("x") as handle:
        yaml.safe_dump(entry, handle, sort_keys=False)
    return path


def supervise(train_dir, root=DEFAULT_ROOT, trainer=trainer_command, registry_root=registry.REGISTRY_DIR,
              bench_kwargs=None):
    train_dir = Path(train_dir)
    request = json.loads((train_dir / "request.json").read_text())
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (train_dir / "train.log").open("x") as log:
        process = subprocess.Popen(trainer((train_dir / "config.yaml").resolve()), cwd=MODEL_TRAINING,
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        set_state(train_dir, state="training", trainer_pid=process.pid, supervisor_pid=os.getpid())
        code = process.wait()
    log_info = parse_log(Path(request["model_dir"]) / "training_log")
    if code != 0 or not log_info["best_line"]:
        set_state(train_dir, state="failed", exit_code=code, log=log_info)
        return 1
    path = register(train_dir, registry_root)
    state = set_state(train_dir, state="registered", exit_code=code, log=log_info, registry_entry=str(path))
    if state.get("then_bench"):
        from . import bench
        from .store import flocked, lock_path
        spec = registry.resolve(state["then_bench"], acoustic=request["name"], root=registry_root)
        set_state(train_dir, state="benchmark_waiting")
        with flocked(lock_path(root, "job")):   # wait for any running benchmark
            pass
        run_dir = bench.run_standard(spec, root=root, **(bench_kwargs or {}))
        set_state(train_dir, state="benchmarked", benchmark_run=run_dir.name)
    return 0


def status(root=DEFAULT_ROOT):
    rows = []
    for path in sorted((Path(root) / "train").glob("*/state.json"), reverse=True):
        state = json.loads(path.read_text())
        request = json.loads((path.parent / "request.json").read_text())
        live_pid = state.get("trainer_pid") if state.get("state") == "training" else None
        if live_pid and not pid_alive(live_pid):
            state["state"] = "interrupted"
        log = parse_log(Path(request["model_dir"]) / "training_log") if state.get("state") in (
            "training", "interrupted", "failed", "registered", "benchmarked") else None
        rows.append(dict(id=path.parent.name, name=request["name"], forked_from=request["forked_from"],
                         overrides=request["overrides"], question=request["question"], state=state.get("state"),
                         updated=state.get("updated"), smoke=state.get("smoke"), log=log,
                         benchmark_run=state.get("benchmark_run"), registry_entry=state.get("registry_entry")))
    return rows


def active_training(root=DEFAULT_ROOT):
    """Harness-launched trainer processes still running (GPU/CPU contention for timing)."""
    active = []
    for path in (Path(root) / "train").glob("*/state.json"):
        state = json.loads(path.read_text())
        if state.get("state") == "training" and pid_alive(state.get("trainer_pid")):
            active.append(path.parent.name)
    return active
