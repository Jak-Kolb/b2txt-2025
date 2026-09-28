"""Stand-in for model_training/train_model.py in harness tests: writes a log and a tiny checkpoint
built from the given config (random weights; no training)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf

from model_training.benchmark.common import build_model

config = OmegaConf.load(sys.argv[1])
output = Path(config.output_dir)
output.mkdir(parents=True, exist_ok=False)
checkpoint = Path(config.checkpoint_dir)
checkpoint.mkdir(parents=True, exist_ok=False)
torch.manual_seed(0)
torch.save({"model_state_dict": build_model(config).state_dict()}, checkpoint / "best_checkpoint")
OmegaConf.save(config, checkpoint / "args.yaml")
loss = sys.argv[3] if len(sys.argv) > 3 else "2.50"
(output / "training_log").write_text(
    f"t: Train batch 0: loss: {loss} grad norm: 1.00 time: 0.100\n"
    "t: Val batch 0: PER (avg): 0.5000 CTC Loss (avg): 2.0000 time: 1.000\n"
    "t: Best avg val PER achieved: 0.50000\n")
