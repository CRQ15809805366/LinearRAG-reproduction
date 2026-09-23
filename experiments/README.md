# Experiment code

This directory contains the bounded Q1–Q8 experiment adapters. Each question's subdirectory holds its executable script and, where present, a short usage guide. The adapters define experiment-specific sampling and measurement around the reproduced implementation in `src/`; they are not a second implementation of LinearRAG.

Generated run evidence belongs under `data/output/experiment_results/<question>/`. Start with `project/agent/experiments/README.md` to identify completed runs, their result records, and the limits on interpreting them. Use a new experiment ID for every new run; do not overwrite recorded evidence.
