"""Registry of acoustic models, LM graphs, and pipeline presets (YAML; no torch).

A resolved pipeline spec is plain JSON: it names every component by absolute path, carries all
eight native decode settings, and records preprocessing as trained versus effective so a
train/inference mismatch is explicit rather than silent.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import REPO
from .store import sha256_json

REGISTRY_DIR = Path(__file__).with_name("registry")
DECODE_KEYS = ("acoustic_scale", "blank_penalty", "beam", "lattice_beam", "max_active",
               "min_active", "length_penalty", "blank_skip_thresh")
INT_DECODE_KEYS = ("max_active", "min_active")
PREPROCESS_KEYS = ("smooth_data", "smooth_kernel_std", "smooth_kernel_size", "smooth_lookahead")
KINDS = {
    "acoustic": ({"name", "checkpoint_dir"}, {"description", "training", "selection"}),
    "lm": ({"name", "graph_dir", "default_decode"},
           {"description", "expected_rss_gb", "default_decode_tuned_on"}),
    "pipelines": ({"name", "acoustic", "lm"},
                  {"description", "decode", "decode_tuned_on", "preprocess_override"}),
    "rescorer": ({"name", "model", "nbest", "weights", "tuned_on"}, {"description"}),
}


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def load_kind(kind, root=REGISTRY_DIR):
    required, optional = KINDS[kind]
    entries = {}
    for path in sorted((Path(root) / kind).glob("*.yaml")):
        entry = yaml.safe_load(path.read_text())
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: expected a mapping")
        keys = set(entry)
        if not required <= keys or not keys <= required | optional:
            raise ValueError(f"{path}: keys {sorted(keys)}; required {sorted(required)}, "
                             f"optional {sorted(optional)}")
        if entry["name"] != path.stem:
            raise ValueError(f"{path}: name must match the file name")
        entries[entry["name"]] = entry
    return entries


def load_all(root=REGISTRY_DIR):
    return {kind: load_kind(kind, root) for kind in KINDS}


def check_decode(values, complete):
    unknown = set(values) - set(DECODE_KEYS)
    if unknown:
        raise ValueError(f"Unknown decode keys: {sorted(unknown)}")
    if complete and set(values) != set(DECODE_KEYS):
        raise ValueError(f"Decode settings need all of {DECODE_KEYS}")
    out = {}
    for key, value in values.items():
        out[key] = int(value) if key in INT_DECODE_KEYS else float(value)
    return out


def read_saved_args(checkpoint_dir):
    path = Path(checkpoint_dir) / "args.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved args: {path}")
    return yaml.safe_load(path.read_text())


def trained_preprocess(saved_args):
    transforms = saved_args["dataset"]["data_transforms"]
    return dict(smooth_data=bool(transforms.get("smooth_data", True)),
                smooth_kernel_std=transforms.get("smooth_kernel_std"),
                smooth_kernel_size=transforms.get("smooth_kernel_size"),
                smooth_lookahead=transforms.get("smooth_lookahead"))


def streaming_capable(preprocess):
    """Mirrors StreamingDecoder: smoothing needs an explicit (causal) lookahead."""
    return not preprocess["smooth_data"] or preprocess["smooth_lookahead"] is not None


def resolve(pipeline=None, *, acoustic=None, lm=None, decode=None, preprocess=None,
            root=REGISTRY_DIR):
    """Resolve a preset and/or ad-hoc components into a complete spec."""
    registry = load_all(root)
    preset = registry["pipelines"][pipeline] if pipeline else {}
    if pipeline is None and not (acoustic and lm):
        raise ValueError("Give a pipeline preset or both --acoustic and --lm")
    acoustic_name = acoustic or preset["acoustic"]
    lm_name = lm or preset["lm"]
    acoustic_entry = registry["acoustic"][acoustic_name]
    lm_entry = registry["lm"][lm_name]

    preset_decode = check_decode(preset.get("decode") or {}, complete=False)
    cli_decode = check_decode(decode or {}, complete=False)
    resolved_decode = check_decode({**lm_entry["default_decode"], **preset_decode, **cli_decode},
                                   complete=True)
    if cli_decode:
        tuned_on = None
    elif preset.get("decode_tuned_on") not in (None, "inherit"):
        tuned_on = preset["decode_tuned_on"]
    elif preset_decode:
        raise ValueError(f"Preset {pipeline} overrides decode settings without decode_tuned_on")
    else:
        tuned_on = lm_entry.get("default_decode_tuned_on")

    override = {**(preset.get("preprocess_override") or {}), **(preprocess or {})}
    unknown = set(override) - set(PREPROCESS_KEYS)
    if unknown:
        raise ValueError(f"Preprocess overrides are limited to {PREPROCESS_KEYS}; got {sorted(unknown)}")
    checkpoint_dir = repo_path(acoustic_entry["checkpoint_dir"])
    saved = read_saved_args(checkpoint_dir)
    trained = trained_preprocess(saved)
    effective = {**trained, **override}

    changed = bool(acoustic or lm or cli_decode or preprocess)
    name = pipeline if pipeline and not changed else "+".join(
        [acoustic_name, lm_name] + (["decode"] if cli_decode else []) + (["preprocess"] if override else []))
    model = saved["model"]
    return dict(
        name=name, preset=pipeline,
        acoustic=dict(name=acoustic_name, checkpoint_dir=str(checkpoint_dir),
                      selection=acoustic_entry.get("selection"), training=acoustic_entry.get("training")),
        lm=dict(name=lm_name, graph_dir=str(repo_path(lm_entry["graph_dir"])),
                expected_rss_gb=float(lm_entry.get("expected_rss_gb", 2.0))),
        decode=resolved_decode, decode_tuned_on=tuned_on,
        preprocess=dict(trained=trained, effective=effective, override=override or None),
        streaming_capable=streaming_capable(effective),
        model=dict(n_input_features=int(model["n_input_features"]), n_classes=int(saved["dataset"]["n_classes"]),
                   patch_size=int(model["patch_size"]), patch_stride=int(model["patch_stride"]),
                   sessions=list(saved["dataset"]["sessions"])),
        finalization=None)


def identity(spec, hasher):
    """Content identity: component file hashes plus effective settings."""
    checkpoint = Path(spec["acoustic"]["checkpoint_dir"])
    graph = Path(spec["lm"]["graph_dir"])
    shas = dict(best_checkpoint=hasher(checkpoint / "best_checkpoint"),
                args_yaml=hasher(checkpoint / "args.yaml"),
                tlg=hasher(graph / "TLG.fst"), words=hasher(graph / "words.txt"))
    acoustic_key = sha256_json(dict(best_checkpoint=shas["best_checkpoint"], args_yaml=shas["args_yaml"],
                                    preprocess=spec["preprocess"]["effective"]))
    pipeline_hash = sha256_json(dict(acoustic=acoustic_key, tlg=shas["tlg"], words=shas["words"],
                                     decode=spec["decode"], finalization=spec["finalization"]))
    return dict(sha256=shas, acoustic_key=acoustic_key, pipeline_hash=pipeline_hash,
                hash_method=getattr(hasher, "method", "sha256"))


def flags(spec, partition, limited=False):
    """Labels that must travel with every number produced by this spec on this partition."""
    out = []
    selection = (spec["acoustic"].get("selection") or {}).get("data") or ["unknown"]
    if partition in selection:
        out.append("acoustic_selected_on_scope")
    if "unknown" in selection:
        out.append("acoustic_selection_unknown")
    tuned = spec["decode_tuned_on"]
    if tuned is None:
        out.append("decode_tuning_unrecorded")
    else:
        if tuned.get("partition") == partition:
            out.append("decode_tuned_on_scope")
        if tuned.get("acoustic") != spec["acoustic"]["name"]:
            out.append("decode_tuned_for_other_acoustic")
    if spec["preprocess"]["effective"] != spec["preprocess"]["trained"]:
        out.append("preprocess_mismatch")
    if not spec["streaming_capable"]:
        out.append("offline_noncausal")
    if partition == "former-val-test":
        out.append("exposed_partition")
    if limited:
        out.append("smoke_limited")
    return out
