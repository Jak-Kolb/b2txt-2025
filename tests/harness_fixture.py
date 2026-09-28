"""Synthetic data, checkpoints, and registry for harness tests (Python 3.10 acoustic stack)."""
from pathlib import Path
import os

import numpy as np

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
FAKE_NATIVE = TESTS / "fake_native"
SESSIONS = ["t15.2099.01.01", "t15.2099.01.02", "t15.2025.01.10"]  # the last is a former val-test name
N_FEATURES, N_CLASSES = 4, 5


def encode(text, width=500):
    out = np.zeros(width, dtype=np.int32)
    codes = [ord(c) for c in text]
    out[:len(codes)] = codes
    return out


def write_split(path, trials, rng, labeled=True):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for index, (n_bins, text) in enumerate(trials):
            group = handle.create_group(f"trial_{index:04d}")
            group.create_dataset("input_features", data=rng.normal(size=(n_bins, N_FEATURES)).astype(np.float32))
            group.attrs["n_time_steps"] = n_bins
            group.attrs["block_num"] = 1
            group.attrs["trial_num"] = index
            if labeled:
                phones = np.zeros(500, dtype=np.int32)
                phones[:3] = [1, 2, 3]
                group.create_dataset("seq_class_ids", data=phones)
                group.create_dataset("transcription", data=encode(text))
                group.attrs["seq_len"] = 3


def make_data(root, seed=0):
    rng = np.random.default_rng(seed)
    data = Path(root) / "data"
    for s, session in enumerate(SESSIONS):
        write_split(data / session / "data_val.hdf5",
                    [(20 + 3 * t + s, ["w2 w3.", "w4", "w3 w2 w4", "w2"][t % 4]) for t in range(4)], rng)
        write_split(data / session / "data_train.hdf5", [(12, "W4!"), (15, "w9")], rng)
        write_split(data / session / "data_test.hdf5", [(18, "")], rng, labeled=False)
    # Block description beside the data directory, like data/t15_copyTaskData_description.csv
    rows = ["Date,Post-implant day,Block number,Number of sentences,Corpus,Split"]
    for session, corpus in zip(SESSIONS, ["Switchboard", "Random", "Harvard"]):
        rows.append(f"{session[4:].replace('.', '-')},1,1,4,{corpus},Val/Test")
    (Path(root) / "t15_copyTaskData_description.csv").write_text("\n".join(rows) + "\n")
    return data


def model_args(lookahead=0):
    transforms = {"smooth_data": True, "smooth_kernel_size": 8, "smooth_kernel_std": 1.0}
    if lookahead is not None:
        transforms["smooth_lookahead"] = lookahead
    return {"mode": "train", "output_dir": "unused", "checkpoint_dir": "unused/checkpoint",
            "num_training_batches": 1000, "batches_per_val_step": 100, "batches_per_train_log": 10,
            "model": {"n_input_features": N_FEATURES, "n_units": 8, "n_layers": 1, "patch_size": 3,
                      "patch_stride": 2, "rnn_dropout": 0.0, "input_network": {"input_layer_dropout": 0.0}},
            "dataset": {"sessions": list(SESSIONS), "n_classes": N_CLASSES, "data_transforms": transforms,
                        "dataset_dir": "unused"}}


def make_checkpoint(directory, lookahead=0, seed=3, transformer=False):
    import torch
    from omegaconf import OmegaConf
    from model_training.benchmark.common import build_model
    torch.manual_seed(seed)
    args = model_args(lookahead)
    if transformer:
        args["model"].update(type="transformer", transformer=dict(d_model=16, n_layers=2, n_heads=2,
                                                                  ffn_mult=2, dropout=0.0, window=4))
    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.save({"model_state_dict": build_model(args).state_dict()}, directory / "best_checkpoint")
    OmegaConf.save(OmegaConf.create(args), directory / "args.yaml")
    return directory


def make_registry(root):
    root = Path(root)
    causal = make_checkpoint(root / "models" / "tiny" / "checkpoint", lookahead=0)
    noncausal = make_checkpoint(root / "models" / "tiny_sym" / "checkpoint", lookahead=None)
    transformer = make_checkpoint(root / "models" / "tiny_tfm" / "checkpoint", transformer=True)
    graph = root / "lm_graph"
    graph.mkdir()
    (graph / "TLG.fst").write_bytes(b"fake graph")
    (graph / "words.txt").write_text("w2 2\nw3 3\n")
    registry = root / "registry"
    for kind in ("acoustic", "lm", "pipelines"):
        (registry / kind).mkdir(parents=True)
    selection = "selection: {rule: test, data: [val-dev, former-val-test]}\n"
    (registry / "acoustic" / "tiny.yaml").write_text(f"name: tiny\ncheckpoint_dir: {causal}\n{selection}")
    (registry / "acoustic" / "tiny_sym.yaml").write_text(f"name: tiny_sym\ncheckpoint_dir: {noncausal}\n{selection}")
    (registry / "acoustic" / "tiny_tfm.yaml").write_text(f"name: tiny_tfm\ncheckpoint_dir: {transformer}\n{selection}")
    (registry / "lm" / "fake.yaml").write_text(
        f"name: fake\ngraph_dir: {graph}\nexpected_rss_gb: 0.1\n"
        "default_decode: {acoustic_scale: 0.4, blank_penalty: 1.0, beam: 17, lattice_beam: 8, "
        "max_active: 7000, min_active: 200, length_penalty: 0, blank_skip_thresh: 1.0}\n"
        "default_decode_tuned_on: {partition: val-dev, acoustic: tiny, artifact: none}\n")
    (registry / "pipelines" / "tiny_fake.yaml").write_text(
        "name: tiny_fake\nacoustic: tiny\nlm: fake\ndecode: {}\ndecode_tuned_on: inherit\npreprocess_override: null\n")
    (registry / "pipelines" / "sym_fake.yaml").write_text(
        "name: sym_fake\nacoustic: tiny_sym\nlm: fake\ndecode: {}\ndecode_tuned_on: inherit\npreprocess_override: null\n")
    return registry


def fake_native_env():
    """Environment for LM subprocesses: the fake native module shadows any real one."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(FAKE_NATIVE), str(REPO)])
    return env
