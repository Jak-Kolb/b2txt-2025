"""Real-time-feasible decoding extensions from grok_recs.md (branch grok_test).

Modules implement the conservative ship pipeline without changing the GRU
encoder architecture:

  R1  ensemble          — multi-seed logit averaging
  R2  streaming_decode  — blank-frame skip + entropy-gated adaptive rescoring hooks
  R3  phase_stagger     — sub-cadence emission via patch phase offsets
  S6  peak_delay        — measure CTC peak delay before emission regularizers
      stability         — word-level time-to-final / revisions-per-word
      checkpoint_avg    — SWA / last-k weight averaging (eval-only)
"""

from .ensemble import EnsembleDecoder, average_logits, load_ensemble
from .phase_stagger import PhaseStaggerConfig, phase_stagger_logits
from .streaming_decode import (
    AdaptiveRescoreConfig,
    StreamingBeamState,
    blank_skip_mask,
    frame_entropy,
    should_trigger_rescore,
)
from .stability import StabilityTracker, compute_stability_metrics
from .peak_delay import measure_peak_delays, summarize_peak_delays

__all__ = [
    "EnsembleDecoder",
    "average_logits",
    "load_ensemble",
    "PhaseStaggerConfig",
    "phase_stagger_logits",
    "AdaptiveRescoreConfig",
    "StreamingBeamState",
    "blank_skip_mask",
    "frame_entropy",
    "should_trigger_rescore",
    "StabilityTracker",
    "compute_stability_metrics",
    "measure_peak_delays",
    "summarize_peak_delays",
]
