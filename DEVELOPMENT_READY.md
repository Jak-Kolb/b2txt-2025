# B2T environment and location — checked 2026-09-18

**Ready for development on JakPC: B2T streaming instrumentation first, with lab
ECoG integration as an intended second track. Specific validity issues constrain
the affected experiments and claims, not all development.**

Repository: /home/fishinjak/code/nejm-brain-to-text, branch main.
Access from the Mac: ssh jakpc with an explicit command. The interactive jakpc
function attaches a shared tmux session; leave it undisturbed.
Start with AGENTS.md, then brainstorm/PLAN.md. Integration status verified
2026-09-22: paced feature/GRU/native-LM replay and output/revision tracing work
across the two existing environments. The September 19 two-trial development smoke
completed; its source/result hashes and replayed summary were verified.
See results/paced_replay_smoke_20260919_02/ and RESULTS.md for qualified observations.
Final synthetic checks: 52 passed in .venv; 49 passed and 3 acoustic-stack checks
skipped in the Python 3.9 LM environment. Compile, CLI, and diff checks passed.
Lab access and schema remain open; no lab dataset integration is claimed.

## Git inventory

PC main and live GitHub main were 05500b870ccf96b072ff02cd7cde745807ff573d.
The Mac main is ded9959, four commits behind, and is a reference copy only.
GitHub grok_test at d893dbc preserves an alternative pipeline (one unique commit
versus 16 on main); it is not missing newer mainline development.
The old audit and causal-preprocessing refs are historical; their mainline work
is already incorporated. No branches were deleted, switched, reset, or merged.
Both machines had one worktree and no stashes.

PC research updates in AGENTS.md, RESULTS.md, and brainstorm/PLAN.md were already
uncommitted. Initial backups are in .git/development-prep/2026-09-18/.
The later documentation-cleanup backup is .git/context-cleanup/2026-09-18/,
with hashes and a manifest. Backups are on the PC disk, not independent storage.
Scope-alignment edits have a separate hashed backup in .git/scope-alignment/20260918T234204Z/.
The root CLAUDE.md pointer and the five scoped rule/skill files are eligible for
version control; assistant settings and local state remain ignored. These files
are still untracked until an intentional commit. Instruction-tracking backup:
.git/scope-alignment/20260918T234204Z/instruction-tracking/.
No commits or pushes were made.

## Verified runtime

- .venv/bin/python: Python 3.10.20, torch 2.13.0+cu130, RTX 5070 Ti with sm_120.
  Actual trainer/model/data/streaming imports and a tiny CUDA tensor operation passed.
- uv pip check: all 182 installed packages compatible. There is no pip module;
  use uv pip. Do not rerun the generic cu126 setup on this Blackwell GPU.
- LM: /home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python, Python 3.9.23,
  NumPy 1.24.4. lm_decoder import passed.
- Syntax parsing passed for 23 first-party Python files. Synthetic normalizer
  finite-output, warmup and future-perturbation checks passed; the later proposed
  fixed-floor invariance counterexample failed as documented in the assessment.
- Data inventory: 45 session directories, 127 HDF5 files. Anchor and R-D1a
  checkpoint files and the built 4-gram exist. Presence is not a scientific audit.
- At initial inspection: 701 GiB disk free, GPU about 1.2 GiB occupied, no Python
  training process observed. Recheck before jobs.
- rg is unavailable on the PC; use grep/find/Python.

The initial environment audit did not load models or evaluate data. Subsequent
authorized development loaded existing checkpoints/LM assets for the bounded
two-trial integration smoke; no training, downloads, or environment changes occurred.
The first attempt completed decoding but exceeded its 5-second worker shutdown
limit. A separate bounded 60-second teardown grace period and regression tests
resolved it; both attempt directories are preserved. Development backups are in
.git/paced-replay-development/20260919T172727Z/.
Historical watchdog was removed during cleanup because it would delete the
completed R-D1a directory. Its exact source is backed up.
