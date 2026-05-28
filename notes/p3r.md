# P3R — planar 3-regular graph classification

Paper §7.1.2 (synthetic expressiveness test for planar GNNs).

## Task

9 non-isomorphic planar 3-regular graphs on 10 nodes, each replicated 50 times
under different random node permutations → 450 graphs total, perfectly
class-balanced. The classifier must recover the isomorphism class from the
permuted edge list alone — node features are uninformative (all-ones).

Random accuracy is 1/9 ≈ 11%. Standard MPNNs cap at ~11% (GIN in the paper).
PPGN and BasePlanE both hit 100%.

## Setup features

- **BasePlanE** with all five aggregations on: `n_t_b_gr_cr` (1-hop neighbors,
  triconnected, biconnected, global readout, cut-subtree). The paper also
  reports a `t_b` ablation (triconnected + biconnected only) that already hits
  100%.
- No edge features (`d_edge = 0`), no dropout, all MLPs use BatchNorm only at
  the two output sites paper Appendix D.4 specifies.
- 10-fold cross-validation. Labels are pre-shuffled in `P3R.pkl`, so
  contiguous `fold * n_fold : (fold+1) * n_fold` slices are roughly
  class-balanced. `experiments/train_p3r.py` currently uses fold 0.

## Hyperparameter grid (paper sweep, upstream `plane.yaml`)

| Param         | Values            |
| ------------- | ----------------- |
| `lr`          | `1e-3`, `1e-4`    |
| `d_hid`       | `32`, `64`        |
| `plane_terms` | `t_b`, `n_t_b_gr_cr` |
| `n_layers`    | `2`               |
| `n_batch`     | `128`             |
| `n_epochs`    | `100`             |
| `p_drop`      | `0`               |
| `d_pe`        | `16`              |
| `seed`        | `0`               |
| split         | 10-fold (`0_10` … `9_10`) |

That's 2 × 2 × 2 × 10 = 80 runs. Best validation accuracy per cell is reported.

## Reproduce

From the repo root:

```bash
# Default — paper P3R config with 5 cosine-LR seeds, ~50s each on CPU.
python experiments/train_p3r.py
python experiments/train_p3r.py --seed 0
python experiments/train_p3r.py --seed 1
python experiments/train_p3r.py --seed 7
python experiments/train_p3r.py --seed 42
python experiments/train_p3r.py --seed 123

# Evaluate the saved best checkpoint:
python experiments/train_p3r.py --eval-only
```

Defaults differ slightly from the paper sweep cell (we add cosine LR schedule
and 200 epochs instead of plain SGD for 100) — these are friendlier on a
single seed without trading expected best-checkpoint accuracy.

| Flag             | Default | Paper sweep value                |
| ---------------- | ------- | --------------------------------- |
| `--n-epochs`     | 200     | 100                               |
| `--n-batch`      | 128     | 128                               |
| `--lr`           | 1e-3    | grid {1e-3, 1e-4}                 |
| `--d-hid`        | 64      | grid {32, 64}                     |
| `--n-layers`     | 2       | 2                                 |
| `--d-pe`         | 16      | 16                                |
| `--p-drop`       | 0       | 0                                 |
| `--scheduler`    | cos     | none                              |
| `--src-pickle`   | `../PlanE/.dataset_src/P3R.pkl` | (provide the upstream pickle path) |

## Observed results (mini-PlanE)

| Seed | Best test acc | Epoch |
| ---- | ------------- | ----- |
| 0    | 1.0000        | 87    |
| 1    | 1.0000        | 104   |
| 7    | 1.0000        | 89    |
| 42   | 1.0000        | 69    |
| 123  | 1.0000        | 134   |

`train_p3r.py` saves the **best-val** checkpoint along the way; final test
accuracy on the saved checkpoint is 1.0 on every seed tested.

## Sage / environment

`planar_preprocess` shells into Sage for the SPQR decomposition. The script
points `DOT_SAGE` / `SAGE_CACHE_DIR` at `.sage/` in the repo root so cache
writes don't hit the home directory. See top-level `README.md` for the
micromamba install line.
