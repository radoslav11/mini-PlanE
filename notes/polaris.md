# Polaris — modern molecular property prediction benchmarks

A more recent (and arguably more relevant) alternative to OGB-MolHIV / MoleculeNet
for drug-discovery property prediction. Polaris is curated by an industry
consortium ([polarishub.io](https://polarishub.io/)), with task definitions,
splits, and evaluation logic all version-controlled. The official scores are
computed server-side (the test labels are deliberately hidden), so leaderboard
entries are not gameable.

> **Scope of this experiment**: like the ZINC reproduction in
> [`zinc.md`](zinc.md), this is a minimal evaluation — one seed, one model
> config, all training on CPU on a single laptop. The goal is to validate that
> mini-PlanE composes with a third-party benchmark API end-to-end, not to
> compete on the leaderboard. The default task is one of the smaller ones
> (~900 molecules total); larger Polaris tasks fit the same pipeline.

## Default task — `tdcommons/caco2-wang`

| Field         | Value                                    |
| ------------- | ---------------------------------------- |
| Source        | TDC ADME panel, hosted via Polaris       |
| Inputs        | SMILES strings                            |
| Target        | Caco-2 cell-line apparent permeability (`log Papp`) |
| Task          | Single-task regression                    |
| Train / Test  | 728 / 182                                 |
| Split         | TDC scaffold split                        |
| Metric        | MAE (lower is better)                     |

## Pipeline

```
SMILES (str)
   |  rdkit.Chem.MolFromSmiles
Mol
   |  smiles_to_pyg(): atom indices, bond-type indices, edge_index
PyG Data    (1-D long x, 1-D long edge_attr, no targets stripped)
   |  planar_preprocess  (Sage SPQR + BC tree)
DataPlanE
   |  one-hot replace: x -> [N, 13],  edge_attr -> [E, 5]
batch
   |  PlanE forward
prediction (float)
```

- **Atom vocab** (13 categories): C, N, O, F, P, S, Cl, Br, I, B, Si, H + "other".
  All drug-like, fixed across tasks so the same model can transfer.
- **Bond vocab** (5 categories): single, double, triple, aromatic + "other".
- **Non-planar / unparseable**: caught with try/except in the worker, dropped
  and counted. For organic drug molecules the drop rate is ~0%.

## Hyperparameters

Defaults are conservative for a small (~700-graph) task:

| Flag             | Default | Notes                                   |
| ---------------- | ------- | --------------------------------------- |
| `--n-epochs`     | 500     | small dataset, lots of epochs OK         |
| `--n-batch`      | 64      | small to stabilise BN on tiny dataset    |
| `--lr`           | 1e-3    |                                         |
| `--d-hid`        | 128     |                                         |
| `--n-layers`     | 3       |                                         |
| `--d-pe`         | 16      |                                         |
| `--p-drop`       | 0.1     | a bit of dropout helps for small data    |
| `--no-edge-feat` | off     | E-BasePlanE by default                  |
| LR scheduler     | ReduceLROnPlateau (factor 0.5, patience 25, min_lr 1e-6) |
| Local val        | 20% of train, fixed perm by seed         |
| Test scoring     | `bench.evaluate(preds)` (server-side)    |

Train is internally split 80/20 into fit / local-val (Polaris doesn't expose
a separate val split). The best-local-val checkpoint's predictions are sent to
`bench.evaluate(...)` for the official score.

## Reproduce

From the repo root:

```bash
pip install polaris-lib rdkit             # one-time

# Default: tdcommons/caco2-wang.
python experiments/train_polaris.py

# Different benchmark (any single-task molecular regression on Polaris):
python experiments/train_polaris.py --benchmark tdcommons/lipophilicity-astrazeneca
python experiments/train_polaris.py --benchmark biogen/adme-fang-perm-reg-v2

# Just preprocess (cache to .dataset/polaris/<slug>/):
python experiments/train_polaris.py --preprocess-only --n-workers 10
```

## Observed results (mini-PlanE, single seed=42)

| Benchmark               | Best local-val MAE | Polaris test MAE  | Notes                                |
| ----------------------- | ------------------ | ----------------- | ------------------------------------ |
| `tdcommons/caco2-wang`  | 0.3094 @ ep 481    | **0.3137**         | 500 epochs, ~12 min CPU              |

Reference: the TDC `caco2-wang` leaderboard (also mirrored on Polaris) puts
the top reproducible methods in the MAE 0.27–0.33 band — see [tdcommons.ai →
ADMET → Caco2-Wang](https://tdcommons.ai/benchmark/admet_group/caco2_wang/).
mini-PlanE's single-seed entry of 0.3137 is mid-pack on this leaderboard with
no per-task tuning, no ensembling, and one seed.

The 500-epoch trajectory:

| Epoch | Local-val MAE | LR        | Notes                       |
| ----- | ------------- | --------- | --------------------------- |
| 1     | ~0.65         | 1e-3      | warm-up                     |
| ~100  | ~0.40         | 1e-3      | first plateau               |
| ~150  | ~0.36         | 5e-4      | first LR halving             |
| ~250  | ~0.33         | 2.5e-4    | second halving               |
| ~330  | 0.316         | 1.25e-4   | third halving                |
| ~370  | 0.312         | 3.1e-5    | fourth halving               |
| 481   | **0.309**     | 3.9e-6    | best (scheduler near floor) |
| 500   | 0.314         | 3.9e-6    | end                         |

Full per-epoch log: `.checkpoints/polaris_tdcommons_caco2-wang_log.csv`.

## Notes for future tasks

- Multi-task regression (e.g. `graphium/l1000-mcf7-v1`) needs `n_cls = n_targets`
  and a multi-target L1 / MSE loss — small change to `train_polaris.py`.
- Classification tasks need `nn.BCEWithLogitsLoss` and an AUROC tracker; the
  PlanE model already supports `n_cls > 1` so this is purely a head/loss swap.
- The atom/bond vocabularies above cover almost all drug-like molecules; for
  more exotic chemistry (organometallics, etc.) add to `ATOM_LIST` /
  `BOND_LIST` at the top of `experiments/train_polaris.py`.
