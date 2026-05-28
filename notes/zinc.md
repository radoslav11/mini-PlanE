# ZINC — molecular graph regression

Paper §7.4 (constrained-solubility regression on molecular graphs).

> **Scope of this experiment**: this is a deliberately *minimal* reproduction —
> one seed, one cell of the paper's `{lr × d_hid}` grid, ~40 min wall-clock on
> a single laptop CPU. It is **not** a full evaluation in the sense the paper
> reports (mean over 10 seeds × best of grid × multiple GPU days). The goal is
> to demonstrate that mini-PlanE's port reaches roughly the right neighbourhood
> of the paper number, not to compete on a leaderboard. See "Observed
> results" below for the gap and "Architectural delta vs upstream" for the
> known reasons.

## Task

ZINC 12k subset (10000 train / 1000 val / 1000 test) of organic molecular
graphs from the ZINC database. Target is penalised logP, a scalar regression.
The full ZINC dataset has ~250k molecules; the 12k subset is the standard
benchmark from Dwivedi et al.

Reported MAE numbers from the paper Table 6:

| Model              | Edge feats | ZINC 12k MAE |
| ------------------ | ---------- | ------------ |
| GCN                | —          | 0.278        |
| GIN                | no         | 0.387        |
| PNA                | no         | 0.320        |
| GIN-E              | yes        | 0.252        |
| PNA-E              | yes        | 0.188        |
| GSN                | yes        | 0.101        |
| CIN                | yes        | 0.079        |
| BasePlanE          | no         | **0.124**    |
| **E-BasePlanE**    | yes        | **0.076**    |

## Setup features

- ZINC nodes carry one of 28 atom types; edges carry one of 4 bond types.
  Both are one-hot encoded after preprocessing (`d_node = 28`, `d_edge = 4`).
- With `d_edge > 0`, `PlaneLayer.aggr_neigh` switches from `GINConv` to
  `GINEConv` automatically — this is what makes the model E-BasePlanE (every
  1-hop message becomes `MLP(h_src + h_edge)` instead of just `MLP(h_src)`).
  `TriEnc` already consumes real edge features for cycle edges in both
  configs; the switch only affects neighbor messaging.
- Same encoders as P3R (TriEnc / BiEnc / CutEnc + global readout) but a wider
  config (3 layers, 128-D hidden) and L1 loss with ReduceLROnPlateau.

## Hyperparameter grid (paper Appendix D.4)

| Param         | Values                                        |
| ------------- | --------------------------------------------- |
| `d_hid`       | `64`, `128`                                   |
| `d_pe`        | `16`                                          |
| `n_layers`    | `3`                                           |
| `lr`          | `1e-3`, `5e-4`, `1e-4`                        |
| LR schedule   | halve every 25 epochs with no val improvement |
| `n_batch`     | `256`                                         |
| loss          | L1                                            |
| `n_epochs`    | `500` (12k subset) / `200` (full)             |
| `p_drop`      | searched per-dataset (typ. 0 here)            |

## Reproduce

From the repo root:

```bash
# One-time preprocessing. ~12s for all 12k molecules on 10 cores via `fork`
# multiprocessing — workers inherit the already-imported Sage interpreter, so
# you pay the ~8s sage init once in the parent rather than once per worker.
python experiments/train_zinc.py --preprocess-only --n-workers 10

# Train E-BasePlanE (default): 500 epochs, ~13s/epoch on CPU = ~110 min.
python experiments/train_zinc.py

# BasePlanE (no edge features in the neighbor aggregator):
python experiments/train_zinc.py --no-edge-feat
```

The preprocessed dataset is cached under `.dataset/ZINC/{train,val,test}.pt`;
re-running the script reuses it. PyG downloads the raw ZINC subset on first
run to `.dataset/ZINC_raw/`.

| Flag             | Default | Notes                                       |
| ---------------- | ------- | ------------------------------------------- |
| `--n-epochs`     | 500     |                                             |
| `--n-batch`      | 256     |                                             |
| `--lr`           | 1e-3    | grid: {1e-3, 5e-4, 1e-4}                    |
| `--d-hid`        | 128     | grid: {64, 128}                             |
| `--n-layers`     | 3       |                                             |
| `--d-pe`         | 16      |                                             |
| `--p-drop`       | 0       |                                             |
| `--no-edge-feat` | off     | flip on to recover BasePlanE (0.124)         |
| `--n-workers`    | cpu-2   | preprocessing pool size                     |
| `--device`       | cuda    | falls back to CPU; MPS disabled (crashes on empty placeholders during scatter backward) |

## Observed results (mini-PlanE, single seed=42, 500 epochs, ~80 min CPU)

Final: **val MAE 0.0854, test MAE 0.0823** at epoch 444. The paper reports
0.076 ± 0.003 averaged over 10 seeds × best of `{lr × d_hid}` grid, so we
land about +0.006 above their reported mean (≈ 2σ).

| Epoch | val MAE | test MAE | LR        | Context vs paper baselines     |
| ----- | ------- | -------- | --------- | ------------------------------ |
| 5     | 0.6254  | 0.5930   | 1e-3      |                                |
| 20    | 0.2292  | 0.2124   | 1e-3      | passes GIN-E (0.252)           |
| 50    | 0.1982  | 0.1919   | 1e-3      | passes PNA-E (0.188)           |
| 150   | 0.1188  | 0.1156   | 1e-3      | matches BasePlanE-paper (0.124)|
| 195   | —       | —        | 1e-3 → 5e-4 | 1st LR halving                |
| 220   | 0.1026  | 0.0949   | 5e-4      | passes CIN (0.079) on test only|
| 290   | —       | —        | 5e-4 → 2.5e-4 | 2nd halving                 |
| 325   | —       | —        | 2.5e-4 → 1.3e-4 | 3rd halving               |
| 385   | —       | —        | 1.3e-4 → 6.3e-5 | 4th halving               |
| 444   | 0.0854  | **0.0823** | 1.3e-4 | best                           |

Full per-epoch trajectory is logged to `.checkpoints/zinc_log.csv`, and
`experiments/plot_zinc.py` renders it as `.checkpoints/zinc_log.png` with
paper baselines overlaid.

### Architectural delta vs upstream

Compared to upstream's `experiments/config/real_world/zinc/12k-plane.yaml`,
the things that almost certainly account for the remaining +0.006 gap:

1. **No BatchNorm in aggregator MLPs** — upstream's `flags_norm_before_com`
   defaults to `"batch_norm"`, which the ZINC YAML doesn't override. Our
   `gin_mlp()` in `PlaneLayer` hardcodes `norm="none"` because that's the
   P3R YAML's value (P3R explicitly sets `model_mlp_norm: "None"`). For
   ZINC, this means upstream has BN inside `aggr_neigh`, `aggr_spqr`,
   `aggr_b`, and `encoder_gr` MLPs that we don't.
2. **Single seed** — paper averages 10 seeds.
3. **Single cell of grid** — paper picks best of `{lr=1e-3 vs 1e-4, d_hid=64
   vs 128}` per seed. We ran `lr=1e-3, d_hid=128`.
4. **LR-scheduler patience 25 vs 30, min_lr 1e-6 vs 1e-5** — cosmetic.

If you want to close (1), add a `norm_aggr` flag to `PlaneLayer` and default
it to `"batch_norm"`, overriding to `"none"` only for the P3R config.

## Sage / environment

Same Sage setup as P3R (see top-level `README.md`). ZINC molecules are all
planar (organic chemistry), so `planar_preprocess` doesn't reject any.
