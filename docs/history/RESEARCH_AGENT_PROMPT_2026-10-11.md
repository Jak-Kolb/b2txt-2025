# Research prompt: novel directions for a real-time intracortical speech-to-text pipeline

You are a research agent. Your job is to find and propose ideas, especially non-obvious ones,
for improving a speech brain-computer interface (BCI) decoding pipeline, given what we have
already measured. Think broadly: architectures, training objectives, data use, decoding and
search, language-model integration, latency/commitment strategies, evaluation design, and
anything else you think matters. Do not limit yourself to the directions we mention; some of our
framing may be wrong, and telling us so is valuable.

Everything below is context. Numbers are our own measurements unless marked as from a paper.

---

## 1. The task and data

- **Dataset:** Brain-to-Text '25 (Card et al., NEJM 2024; Kaggle competition, ended Dec 31, 2025).
  One participant (T15, ALS, severe dysarthria) attempting to speak sentences.
  - 4 Utah arrays (64 electrodes each) in ventral 6v, area 4, 55b, dorsal 6v.
  - 512 features per 20 ms bin: threshold crossings (−4.5 x RMS) and spike-band power per electrode.
  - Released features are already normalized (consistent with per-block z-scoring, clipped at ±10;
    exact provenance unresolved).
  - Splits: 8,072 train / 1,426 val / 1,450 test trials across 45 sessions spanning ~20 months.
    Test labels are not available to us.
  - Sentence sources: Switchboard conversational sentences (most), frequent-word sentences, a 50-word
    set, OpenWebText, Harvard sentences, random word strings.
  - The participant is slow: about 1.6 s per word.
- **Our evaluation split ("val-dev"):** 35 val sessions, 1,253 trials, 8,277 words. Six other val
  sessions ("former val-test", 173 trials) were used for checkpoint selection and are treated as
  exposed. val-dev is also exposed: checkpoints are selected on val PER and decoder settings are
  tuned on val-dev. We have **no independent test set**, so every number is exploratory.
- **Other public data we have downloaded and can use:**
  - Willett et al. 2023 (Nature): participant T12, 256 electrodes (2 arrays in 6v, 2 in area 44),
    same feature types (threshold crossings at several thresholds plus spike-band power), 20 ms bins.
    9,680 labelled sentence trials over 24 sessions, plus 1,200 unlabelled held-out trials. Raw
    (not normalized).
  - Kunz et al. 2025 (Cell, inner speech): small (about 1,500 trials over T12, T15, T16), mostly
    imagined speech. Some T15 imagined-speech sessions are not in B2T '25. 10 ms bins for T15.
  - Other public intracortical data exists (handwriting, monkey reaching, about 367 h in total per
    the BIT paper); not downloaded.
- **A second track** (not yet started): an ECoG dataset from our lab with unknown schema. Ideas that
  would transfer across intracortical and ECoG are a plus, but do not assume shared inputs.

## 2. The current pipeline

1. **Preprocessing:** causal Gaussian smoothing (std 2 bins, 5 taps, past-only).
2. **Per-session input layer:** a 512x512 linear map (identity init) plus softsign, one per
   recording session ("day layers"). Each session's layer is trained only on that session's trials.
3. **Patching:** 14 bins (280 ms) per input frame, stride 4 bins (80 ms), giving a 7,168-dim input.
4. **Acoustic model:** a 5-layer unidirectional GRU, 768 units, dropout 0.4, about 32.5M shared
   parameters plus 11.8M in day layers.
5. **Output:** 41 classes (CTC blank, 39 ARPAbet phonemes without stress, word boundary), CTC loss.
6. **Training:**
   - AdamW (eps 0.1), cosine LR 0.005 → 0.0001, 120k batches of 64 trials drawn from 4 sessions
     per batch.
   - Augmentation: white noise, constant offset, random cut.
   - bf16. About 6.2 h on one RTX 5070 Ti.
7. **Decoding:**
   - A WFST beam search (Kaldi/WeNet-style TLG graph): CTC topology, lexicon, and a 4-gram LM.
   - The 4-gram is trained on OpenWebText, pruned to 125k words, about 8.5 GB RAM.
   - Two dials: acoustic scale (0.4) and blank penalty (log 90).
   - Partial results update every 80 ms.
8. **Finalization, optional and after the sentence ends:** GPT-2 Large re-ranks up to 100 n-best
   candidates.

## 3. What we have measured (val-dev, one participant, dataset-provided sentence boundaries)

**Accuracy:**
- **Reference causal GRU (la0):**
  - **8.51% WER live** [95% session-bootstrap interval 6.62–10.65].
  - Seen/unseen-sentence WER 5.41 / 8.74.
  - **6.87% after GPT-2 re-ranking.**
  - **Oracle best-of-100 candidates: 2.90%.**
- **Where la0's 704 word errors come from:**
  - 582 have wrong phonemes that the LM could not fix.
  - 93 have correct phonemes that the LM overrode.
  - 29 are insertions.
  - The network's greedy phonemes are exactly right for 80.8% of words.
- **By sentence source:** Switchboard 6.9%, frequent-word 3.5%, OpenWebText 12.9%, Harvard 28.6%,
  random word strings 38.5%. The LM does much of the work.
- **Session difficulty:** large between-session spread. Sentence mix explains about 61% of the
  between-session variance.

**Things that did not beat the causal GRU (single runs):**
- **Causal Transformer** (8 layers x 512, ALiBi, 64-frame window), with or without the published
  time-masking recipe: about 10.2–10.4% WER, worse at every stage.
- **Smoother changes:** 80 ms smoother look-ahead (8.80%) and a wider smoother (9.06%).
- **Training tweaks:** weak masking bundle (9.11%; its PER was *better* than la0's but its WER
  worse) and a different objective (8.36%, not distinguishable from la0).
- **The dataset authors' released model** with symmetric smoothing: 8.70%.

**Overfitting:** training CTC loss reaches about 0.02 per trial by batch 36k, while validation loss
bottoms out at batch 6k (16.8) and rises to about 22. Validation PER keeps slowly improving.

**Newest result (Oct 9): future context.**
- **Setup:** we trained a bidirectional 5x768 GRU (sees the whole sentence; 79M shared params) and a
  causal copy of la0 with new seeds, both stopped at 20k of the 120k-batch schedule.
- **WER at batch 20k, each at its best acoustic scale:**
  - Bidirectional **7.90%** vs causal **9.76%**, a difference of **−1.86 points** [−2.96, −0.84].
  - The bidirectional model prefers more LM weight (acoustic scale 0.3 vs 0.5).
- **PER at the same point differed by only about 0.4–0.65 points**, so PER badly understates WER
  effects here.
- **Seed-only PER variation** (one extra seed of the causal model) was about ±0.15 around la0's curve.
- **Not yet separated:** model capacity (79M vs 32.5M parameters) and WER seed variability.

**Latency and commitment:**
- **Timing:** first output 280 ms after speech-period start; updates every 80 ms; compute per update
  about 1 ms GPU plus about 13 ms p95 decoder. All replay of released features; no raw acquisition.
- **Word commitment** (locking words so they never change), replayed on recorded partials:
  - Lock once 3 newer words have appeared: 8.47% WER with 0.4% of locked words wrong, but about a
    5.4 s wait.
  - Lock when unchanged for 80 ms: 14.56%. Unchanged for 1.2 s: 11.05%.
  - 98.8% of words were already final a median 8 s before the sentence ended.
- **Proposed evaluation conditions** (not yet formally defined): 200 ms, 500 ms, 1.5 s, and
  unconstrained.
- **Word-level timing ground truth is not available** (no word alignments).

**Competition results (from papers; the test partition, not comparable one-to-one with val-dev):**
- **Leaderboard:** top private-leaderboard WER 1.54% (Infera-Neuro), then 1.66%, 1.77%, 1.87%. The
  organizers' causal RNN ensemble + 5-gram scored 3.89%.
- **BIT (arXiv 2511.21740):** 29 seeds of a 7M-param *bidirectional* Transformer, pretrained
  self-supervised on 367 h of human and monkey array data; 5-gram + OPT; GPT-4 merges the 29 outputs.
- **LightBeam (arXiv 2603.14002):** the baseline GRU + 5-gram + OPT scores 6.72%. LightBeam (LLM
  fused into the beam every 1.25 s) scores 6.47%. A non-causal Transformer x 10 seeds with a
  QLoRA-fine-tuned Llama-3.1-8B correcting the sentence from 10 candidates scores 2.08–2.36%.
- **BrainWhisperer (arXiv 2603.13321, 4th):** Whisper-based, windowed non-causal attention, trained
  across three datasets.
- **B2T '24 lessons paper (arXiv 2412.17227):** ensembles merged by a fine-tuned LLM were the largest
  gain; diphone targets and LR schedules helped; new architectures did not beat the RNN.
- **Our reading:** the big gains are whole-sentence (finalization) techniques with non-causal models
  and ensembles. None of these systems report live word latency.

## 4. Constraints

- **Compute:**
  - One RTX 5070 Ti (16 GB), 22 CPU cores, 27 GB RAM, running in WSL2. The WSL GPU layer has
    produced intermittent CUDA faults during long runs.
  - Practical budget: about 1–2 h per exploratory run (20k batches) and 6–12 h per full run.
    Roughly one long run per night.
- **The official 5-gram LM needs about 300 GB RAM and the 3-gram about 60 GB:** unavailable.
  LLMs up to about 8B (4-bit) fit on the GPU for finalization or rescoring.
- **Measurement rules we hold ourselves to:**
  - Report WER with session-level uncertainty.
  - Separate live output from finalization (whole-sentence rescoring is a finalization stage).
  - Keep future inputs and references out of online decisions.
  - Distinguish compute latency, buffering, look-ahead, commitment delay, and endpointing.
  - Never present val-dev as an independent test.
- **The project goal is not "beat the leaderboard".** It is an efficient real-time pipeline and a
  reproducible finding about what accuracy, stability, and delay cost. A clear negative result or a
  well-explained trade-off counts as success.

## 5. What we want from you

Research freely (papers, preprints, code, ASR and streaming-speech literature, other BCI work,
neuroscience) and come back with **ideas**. We especially want:

1. **Novel or under-explored ideas** that fit this pipeline and these constraints. Explain why the
   idea should help *here* (small data, one participant, slow speech, nonstationary recordings,
   strong LM dependence, CTC phoneme outputs), not just in general.
2. **Ideas about the future-context result:**
   - How to capture as much of the bidirectional gain as possible under a latency budget, whether
     through the acoustic model, a second pass, decoding, or commitment.
   - What the result tells us about where information lives in the signal. For example: is it
     coarticulation, late-arriving neural evidence, or word-level context?
   - What confounds we should rule out.
3. **Ideas about using more data:** cross-participant data (T12), unlabelled data (the B2T '25 test
   trials, T12 held-out trials), self-supervised objectives, and what could go wrong (sentence
   leakage, distribution shift, normalization mismatch).
4. **Ideas about the language side:**
   - Exploiting the 2.90% oracle n-best, live LLM integration on a small GPU, and LLM error
     correction trained without contaminating our evaluation data.
   - Smarter commitment rules that use more than word count.
5. **Ideas about the nonstationarity and session structure:** day layers, test-time adaptation,
   calibration-free decoding.
6. **Ideas about evaluation itself:** how to get a defensible confirmatory result without an
   independent test set, how to define the latency conditions, and how to estimate word timing
   without alignments.
7. **Anything we are not asking about but should be.**

## 6. Output format

For each idea (aim for 10–20, ranked by your own judgement of value versus cost):

- **Idea** in one or two sentences.
- **Mechanism:** why it should help in this specific setting.
- **Evidence:** primary sources (with links), and what they actually showed. Mark clearly what is
  established, what is extrapolation, and what is your own hypothesis. Do not invent effect sizes.
- **Novelty:** has it been done for speech BCIs? In ASR? What would be new here?
- **Cost:** code effort and GPU hours on a 16 GB card, and any new data or tools needed.
- **Test:** the smallest experiment that would support or refute it. What result would change our
  minds? (We can run 20k-batch truncated runs compared against logged baselines, and decode WER on
  val-dev with paired session bootstrap.)
- **Latency class:** live (and what added delay), finalization-only, or offline analysis.
- **Risks and confounds.**

Then add:

- **A critique of our current findings and plan:** what might be wrong, over-interpreted, or
  missing.
- **Three to five "if we could only do one thing next" recommendations,** with reasons.
