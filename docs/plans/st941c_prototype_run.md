# st941c prototype runs (local, uses precomputed PMD residuals)

This repo includes a heavy Plan A (per-gene mixed model) layer. For local development, you can validate wiring,
back-compat, and output organization without re-running the PMD bootstrap by pointing to a precomputed
`PMD_std_res.tsv` file via `--std-res-file`.

These instructions assume you have the related repo checked out locally (example path shown below).

## Paths

Example (local) source repo:

- `/home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_inputs/`
- `/home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_out/`

Key files:

- Counts: `.../guide_pmd_inputs/counts.tsv`
- Model matrix (dose-rank): `.../guide_pmd_inputs/model_matrix_dose_rank.tsv`
- Model matrix (one-hot): `.../guide_pmd_inputs/model_matrix_one_hot.tsv`
- Precomputed PMD residuals (dose-rank outputs): `.../guide_pmd_out/dose_rank/PMD_std_res.tsv`
- Precomputed PMD residuals (one-hot outputs): `.../guide_pmd_out/one_hot/PMD_std_res.tsv`

## Why Plan A can be slow on this dataset

If you include all non-intercept model-matrix columns as focal vars, Plan A may select a large number of
`(gene_id, focal_var)` tasks to fit (especially for nuisance covariates like `day`).

For faster iteration, explicitly set `--focal-vars` to the contrast(s) you actually care about.

## Dose-rank run (recommended dev default)

This runs:

- baseline guide-level outputs (byte-identical when using the same `--std-res-file`)
- all gene-level outputs (meta, Plan A LMM, QC, etc.)
- Plan A LMM with checkpoint/resume

```bash
guide-pmd-std-res \
  -in_file /home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_inputs/counts.tsv \
  -out_dir .tmp/st941c/dose_rank \
  -model_matrix_file /home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_inputs/model_matrix_dose_rank.tsv \
  -annotation_cols 2 \
  --std-res-file /home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_out/dose_rank/PMD_std_res.tsv \
  --focal-vars C1_dose_rank C2_dose_rank \
  --gene-progress \
  --gene-lmm-resume \
  --gene-lmm-checkpoint-every 200 \
  --gene-lmm-jobs 8
```

Resume behavior:

- If `gene_level/PMD_std_res_gene_lmm.partial.tsv` exists, rerunning the exact same command resumes from the last checkpoint.
- If any relevant inputs/configs change, the run fails fast with a checkpoint meta mismatch; delete the
  `PMD_std_res_gene_lmm.partial.*` files to restart.

After completion, summarize agreement/disagreement across methods:

```bash
python scripts/compare_gene_level_methods.py --out-dir .tmp/st941c/dose_rank
```

## One-hot run

```bash
guide-pmd-std-res \
  -in_file /home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_inputs/counts.tsv \
  -out_dir .tmp/st941c/one_hot \
  -model_matrix_file /home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_inputs/model_matrix_one_hot.tsv \
  -annotation_cols 2 \
  --std-res-file /home/ubuntu/bin/bfx-crispr-screen-de-paths/.assets/st941c/derived/guide_pmd_out/one_hot/PMD_std_res.tsv \
  --gene-progress \
  --gene-lmm-resume \
  --gene-lmm-checkpoint-every 200 \
  --gene-lmm-jobs 8
```

