# Confirmatory 553-prompt validation (three fresh seeds)

Compares the schedule frozen by `../dense_calibration_200` with unchanged PSP on
all 553 official GenEval prompts, using three fresh candidate-pool base seeds and
paired per-prompt inference.

This directory contains only the implementation.  `prepare_protocol.py` will stop
unless a `FROZEN_SCHEDULE.json` exists in the calibration directory, so the
confirmatory stage cannot run without passing the freeze gate.

`run_all.sh` performs preflight, full generation (553 x 3 seeds x 2 methods),
HPS, official GenEval, and the prompt-level paired bootstrap.
