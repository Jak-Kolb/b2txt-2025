# Streaming neural decoding plan — updated 2026-09-22

**Paced released-feature → GRU → native LM replay is implemented on JakPC.
A two-trial development smoke completed on September 19; artifacts and final
synthetic tests were verified September 22. This validates integration, not
general accuracy or speech-to-word latency. The research direction remains exploratory.**

## Objective and scope

Build an efficient real-time speech-to-text pipeline on B2T '25 and the Makin
lab's ECoG dataset, and use its development to investigate what accuracy,
stability, and delay cost. The class presentation remains the engineering frame.
A paper should explain a reproducible finding from this work; its central result
and novelty are not decided in advance.

Use modality-appropriate features and encoders. Share timing, output-event, and
evaluation conventions where meaningful; report each dataset separately.
B2T implementation can proceed now. ECoG is an intended second development track,
with access, task, data format, alignment, and baseline still unknown.

The [September review](../RESEARCH_ASSESSMENT.md) limits particular claims and
experiments; it does not freeze unrelated development. Original decisions and
gates remain in the [August record](../docs/history/PLAN_2026-08.md).
[RESULTS.md](../RESULTS.md) remains the source-linked observation summary.

## Development milestones

1. **Capture decoder output — implemented, 2026-09-19.**
   The cached-logit decoder now saves timestamped provisional/final tokens, original
   cache/session indices, configuration and source/cache hashes. Completed JSONL
   traces can be replayed into revision summaries. Token edits are separate from
   display-position changes; retractions and reappearances are retained.
   Synthetic tests cover timing order, empty outputs, finalization, reference
   independence, trace completeness, and existing-output protection in both Python
   environments. The legacy acoustic check runs only in its Python 3.10 stack.
   See [benchmark usage](../model_training/benchmark/README.md).
   This is unpaced decoder-stage instrumentation: input-feature availability is
   unknown, dataset endpoints are provided, and no commitment policy runs.

2. **Integrated replay implemented; bounded native validation complete.**
   Connect paced feature input, existing preprocessing/GRU, incremental decoding,
   and output events across the existing acoustic and LM environments. Include
   queueing and synchronization; label cached-component measurements separately.
   Record first output, provisional/committed/final accuracy, revisions,
   processing lag, endpoint behavior, deadline misses, memory, and throughput.
   Treat dataset-provided trial boundaries as an oracle condition unless an
   online endpoint detector actually produces them.
   The synchronous two-process baseline passed streamed/offline logit and final
   native-text checks on two development trials (one session, 409 frame updates).
   See results/paced_replay_smoke_20260919_02/ and RESULTS.md. Input is paced at
   20 ms/bin; clocks include queueing, CUDA completion, IPC, and native decoding.
   Dataset endpoints remain oracle and no online commitment policy is implemented.
   Broader session coverage and statistical baseline characterization remain open.

3. **Establish and integrate the lab ECoG track.**
   With the user/lab, identify authorized storage/access, participants and sessions,
   attempted/spoken task, sampling/features, labels/alignment, split constraints,
   and the lab's existing decoder. Reuse its appropriate frontend/baseline.
   Implement the smallest adapter supported by the real schema; do not build
   speculative loaders or assume B2T phoneme vocabulary/checkpoints will transfer.
   Completion: equivalent trace/metric reporting for an authorized ECoG baseline,
   with dataset-specific limitations explicit. This track can advance alongside B2T.

4. **Explore focused changes against the baseline.**
   Prioritize from measured bottlenecks and early traces, not guessed gains.
   Change one lever at a time initially; test interactions only when warranted.
   Candidate questions:
   - Temporal context: what does future context or chunk size buy in WER and delay?
   - Neural model: when do size, initialization/pretraining, or small ensembles
     justify their compute? Begin with the existing GRU.
   - LM search: how do beam size, LM choice, and rescoring schedule affect accuracy,
     memory, and deadline success?
   - Word commitment: how much waiting reduces revisions, and at what accuracy cost?
   - Runtime: where do caching, batching, precision, or CPU/GPU scheduling matter?
   Save configurations, paired comparisons, and negative results. A candidate is
   useful if it explains a tradeoff; improvement is not required. Compare modern
   relevant work such as LightBeam before claiming a novel decoding method.

5. **Choose the paper question with the lab, then confirm it.**
   Review the measured tradeoffs across available datasets and select a supported,
   useful contribution. Freeze the selected pipeline and evaluation protocol before
   independent evaluation. If independent data are unavailable, present a disclosed
   retrospective exploratory study and limit generalization claims.
   Keep final figures reproducible from saved events and result artifacts.

## Measurement decisions to settle during baseline work

The proposed 200 ms / 500 ms / 1.5 s conditions need an explicit input event, output
event, and failure rule. Measure feature-availability-to-output processing lag
separately from neural/speech-event-to-word delay. Public B2T word alignments are
not available here; inferred alignments are proxies, not ground truth. Report the
proxy and uncertainty if used. Raw acquisition/feature extraction remain outside
the replay boundary unless actually included.

The unconstrained configuration is an offline reference, not a guaranteed accuracy
ceiling. Preserve the old 140 ms criterion in historical results. Report empirical
deadline misses and timing coverage; observed maxima do not establish hard bounds.

## Issues with specific dependencies

- **Normalization experiment:** resolve preprocessing/clipping/reset provenance
  and amend R-D2/R-D3 before that treatment. Test future perturbation, scaling,
  constant/sparse channels, clipping, resets, and offline/online equivalence.
  If raw equivalence cannot be justified, limit claims to released-feature replay.
- **Evaluation:** former val-test participated in checkpoint selection. Existing
  artifacts can inform exploration with disclosure. New use of that partition
  needs an explicit revised scope; no silently created holdout. Define selection
  data and independent evaluation before confirmatory comparisons.
- **Uncertainty:** match conditions and use paired seeds when measuring training
  effects; report session/participant uncertainty separately from seed variation.
  Old single-replicate differences are not equivalence margins.
- **Lab track:** metadata/access questions block its data integration, not B2T work.
  No collaborator messages, data transfers, or downloads follow automatically.

**Next development milestone:** design a bounded, session-spanning development
baseline and an online word-commitment comparison. Define commitment/revision
metrics before running the comparison; retain the existing no-commitment baseline
and oracle endpoint condition. Choose timing proxies explicitly if word alignment
remains unavailable. The two-trial smoke is too small to select a method or support
a paper result. Lab access/schema and independent evaluation still need resolution.
