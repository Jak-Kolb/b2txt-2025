import os

# Batches pad to the longest of their 64 trials, so the input length varies every step. The default
# CUDA caching allocator keeps a separate freed block per shape; measured reservation reached
# 22.87 GB against 7.41 GB actually allocated. Once reservation exceeds physical VRAM the WSL2 WDDM
# manager pages GPU memory over PCIe instead of raising OOM: 0.169 -> ~1.5 s/batch, with no error.
# Must be set before torch is imported. Export the variable yourself to override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

args = OmegaConf.load('rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()