---
paths:
  - "model_training/benchmark/**"
  - "model_training/splits.py"
---

# Measurement rules

- Former val-test is exposed and influenced checkpoint selection. New use needs an
  explicit revised scope; it cannot be an independent end-to-end holdout.
  Obsolete per-group instructions in splits.py and CLI help do not override this.
- State dataset, partition, session/participant/trial/word counts, selection policy,
  and seen/unseen definition. Unseen text is not an unseen participant/session.
  Keep B2T and ECoG scores separate; document task and label differences.
- Report WER and session/participant variation; seen/unseen where applicable.
  Use a stated text normalization policy. Keep diagnostic PER separate.
- Persist timestamped provisional, committed, and final output events. Count
  insertions, deletions/retractions, and substitutions. Distinguish online commitment
  from stabilization identified retrospectively.
- Keep future frames, references, and retrospective alignment out of online policies.
  Label dataset-supplied trial boundaries as oracle endpointing when used.
- Separate compute, queues, cadence, first output, commitment, endpointing, and
  finalization. Record input availability and output events on compatible clocks.
  Cache-only stage measurements are not integrated latency.
- Measure percentiles from paired events; never sum component p95s. State warmup,
  timing coverage, deadline misses, and observed maxima without claiming hard bounds.
- Report the latency boundary and alignment limitations. Feature replay excludes
  raw acquisition/extraction unless measured. Word-alignment proxies are not truth.
- Compare paired effects and session-aware uncertainty; training-seed variability
  is different. A replicate difference is not a noise floor.
- A causal LM on finished sentences remains utterance-final processing.
- Public B2T test labels are unavailable locally. Do not invent test WER or assume
  a leaderboard submission facility is available or authorized.
