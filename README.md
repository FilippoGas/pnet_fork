# P-Net (fork)

A biologically informed deep neural network for genomic prediction. P-Net wires its
layers to the **Reactome pathway hierarchy**, so every weight corresponds to a known
gene→pathway or pathway→pathway relation. This makes the model parameter-efficient and
directly interpretable: you can trace a prediction from a gene, up through the pathway
graph, to the output.

This repository is a working fork of the [van Allen Lab PyTorch P-Net](https://github.com/vanallenlab/pnet),
extended with covariate correction, stratified class balancing, a mask-weighted edge-sparsity
penalty, reproducible seeding, and the Reactome v98 graph.

---

## Model

### Architecture
- **Input.** A dict of `pd.DataFrame`s — one per data modality — with genes as columns and
  samples as rows. `PnetDataset` concatenates them on the gene axis, so `N` modalities over
  `G` genes give `N×G` inputs. In this project the modalities are polygenic score
  aggregations `score_type × {avg, sd, max, min, delta} × {hap1, hap2}` = **10 inputs/gene**.
- **Reactome backbone (`ReactomeNetwork`).** Reads `data/reactome/*` and emits layer masks.
  Depth is capped by `max_depth` (this project uses **`max_depth=7`**).
- **One node per gene + a residual node.** Every input gene is kept: genes present in the
  Reactome GMT attach to their pathways; genes absent from it attach to a shared `residual`
  node at the final gene layer. The model never silently drops genes.
- **`PNET_NN`.** Stacked `PNET_Block`s (masked gene + pathway layers) with a prediction head
  per level used for **deep supervision** (auxiliary heads), plus an optional linear
  **covariate head** (`n_covariates`) and an optional `Regulatory_Block`.
- **Training.** A custom loop (`train`) with Adam/AdamW, early stopping, and a ramped sparsity
  penalty (PyTorch Lightning step methods are also present for Trainer-based use).

### Regularization
- **Dropout** — `input_dropout=0.5` (input layer), `gene_dropout=0.2` (every gene→pathway
  branch, including deep blocks), `dropout=0.2` (pathway blocks). Dropout only masks usage.
- **Weight decay** — `weight_decay=1e-3`, genuine because losses are computed per-sample and
  averaged (`reduction='none'` → `.mean()`).
- **Edge sparsity** — `sparsity_penalty()` adds
  `l1_lambda·Σ|W⊙M| + group_lambda·Σ‖(W⊙M)[:,i]‖₂`, mask-weighted so only real edges are
  penalized (topology preserved) and the group term can prune whole nodes. Ramped over
  `sparsity_warmup`/`sparsity_ramp` epochs. **`l1_lambda=1e-5, group_lambda=1e-4`**, finalized
  on a pooled 10-fold BRCA/PLLR sweep and confirmed to transfer to GBM/PLLR.
- **Aux-loss scaling** — head weights are normalized to sum to `aux_loss_scale=0.5`.
- **Reproducibility** — `Pnet.run(seed=N)` seeds `torch`/`numpy`; expect ~±1 pt AUC drift on
  GPU from cudnn atomics.

---

## How this project uses it

Per-cancer **binary classification** across **33 tumor types** (`cancer_type_<TUMOR>.csv`),
over the available score types **PLL** and **PLLR**.

- **Class balancing** is done per fold *before* training via `util.balance_split` (keeps all
  positives, subsamples negatives, stratified by ethnicity + tumor type); the DataLoader does
  no balancing.
- **Evaluation** is 10-fold `StratifiedKFold(random_state=42)`; folds and seeds are fixed so
  real vs. shuffled runs are paired fold-for-fold.
- **Covariate correction** is enabled by passing a per-sample `covariates` DataFrame to
  `Pnet.run`, which adds a linear covariate head (used by the `*_cov_cor*` arms).
- **Production scripts** live in `scripts/all_cancers/all_scores/`:

  | Script | Covariates | Labels |
  |---|---|---|
  | `balanced.py` | no | real |
  | `balanced_cov_cor.py` | yes | real |
  | `balanced_shuffled.py` | no | shuffled (negative control) |
  | `balanced_cov_cor_shuffled.py` | yes | shuffled (negative control) |

  Each takes `tumor_type [gene_list] score_type output_dir` (the cov-cor variants omit
  `gene_list`) and writes per-fold and pooled predictions, ROC/PRC curves, gene/layer
  importances, and a Sankey diagram.
- **Sparsity calibration** (`sparsity_calibrate.py`) sweeps `l1:group` arms on the same pipeline
  and reports **pooled** metrics (fold-averaged predictions/importances) — decide λ on the pooled
  columns.

### Cluster execution

Jobs run under PBS (`cibioGPUQ`, `module load CUDA/12.9.0`, `conda activate pnet`) with wrappers
in `PBS_scripts/PNET/...` and Python drivers in `hpc_workflows/PNET/...` that loop
`tumor_types × score_types` and `qsub` each combination (one 10-fold run per job, one GPU).
`get_gpu.py` picks a free GPU per job. Production input/resource paths are hardcoded absolute
cluster paths, so these scripts only run on-cluster.

---

## Repository layout

```
src/pnet/       Pnet.py (model + run/train/interpret), pnet_loader.py, ReactomeNetwork.py,
                GenesetNetwork.py, CustomizedLinear.py (masked layers)
src/util/       util.py (losses, metrics, balance_split, EarlyStopper), sankey_diag.py
data/reactome/  pathway graph loaded at runtime by ReactomeNetwork (do not move repo layout)
test/           pytest-style tests (run from repo root)
scripts/        production pipelines (cluster-only hardcoded paths)
pnet.yml        conda environment (the only dependency manifest)
```

## Notes
- `src/pnet/PnetOG.py`, `src/pnet/ReactomeNetworkOG.py`, `src/util/hyperparam_search.py` are
  legacy — do not import them in new code.
- Old (pre-2026) checkpoints trained with summed-reduction losses and different aux weighting;
  they are not numerically comparable to current runs and are invalid after the v98 update.
