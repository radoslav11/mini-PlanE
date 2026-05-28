# mini-PlanE

**A minimal implementation of PlanE (Representation Learning over Planar Graphs).**

PlanE is a GNN for **planar graphs** that learns complete invariants while
remaining scalable, inspired by the Hopcroft–Tarjan planar graph isomorphism
algorithm. Each BasePlanE layer combines five signals: 1-hop neighbors,
triconnected components (via the SPQR tree + Weinberg canonical walk),
biconnected components (via bottom-up SPQR-tree message passing), a global sum
readout, and a cut-subtree encoding on the Block-Cut tree.

This minimal version is a port of upstream
[ZZYSonny/PlanE](https://github.com/ZZYSonny/PlanE) aligned with paper Section
5 (BasePlanE) and the per-dataset configs in Appendix D.4.

---

## Cite

```bibtex
@inproceedings{DimitrovZAC23,
  author    = {Radoslav Dimitrov and Zeyang Zhao and
               Ralph Abboud and
               {\.I}smail {\.I}lkan Ceylan},
  title     = {PlanE: Representation Learning over Planar Graphs},
  booktitle = {Proceedings of the Thirty-Seventh Annual Conference on
               Advances in Neural Information Processing Systems, {NeurIPS}},
  year      = {2023}
}
```

**Paper:** https://arxiv.org/abs/2307.01180
**Original repository:** https://github.com/ZZYSonny/PlanE

---

## Setup

mini-PlanE depends on **SageMath** for SPQR-tree decomposition (the canonical
triconnected-component encoder used by `TriEnc` requires Sage's
`TriconnectivitySPQR`; there is no good pure-Python equivalent). Sage is
**not on PyPI** — it must be installed through conda-forge.

The recommended setup uses [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
(a small, fast, drop-in conda/mamba replacement):

```bash
# 1. Install micromamba (one-time; macOS example, see link above for other OS).
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# 2. Create an environment with Sage + PyTorch + PyG + scatter.
micromamba create -n plane -c conda-forge \
    python=3.11 sage=9.6 \
    pytorch pytorch_geometric pytorch_scatter \
    numpy tqdm

# 3. Activate it.
micromamba activate plane

# 4. Install mini-PlanE in editable mode.
pip install -e .
```

Alternatives that work the same way (just swap the binary name):

```bash
mamba create -n plane -c conda-forge ...   # if you have mambaforge installed
conda create -n plane -c conda-forge ...   # slower but works
```

A pure-pip install **will not work** because Sage isn't on PyPI. The
`requirements.txt` in this repo lists pure-Python deps so that
`pip install -r requirements.txt` works inside an environment that already
has Sage.

### Verify the environment

```bash
python -c "import plane, torch, sage.all; print('ok')"
pytest tests/                                              # 4 tests, ~6s
```

---

## Usage

```python
import torch
from torch_geometric.data import Data
from plane import PlanE, planar_preprocess

# Build (or load) a planar PyG graph.
data = Data(
    x=torch.ones((4, 1)),
    edge_index=torch.tensor([[0,1,1,2,2,3,3,0],
                             [1,0,2,1,3,2,0,3]]),
    y=torch.tensor([0]),
)

# Preprocess: SPQR tree + BC tree + canonical codes + batch.
data = planar_preprocess(data)

model = PlanE(d_node=1, n_cls=4)
out = model(data)   # shape [1, n_cls]
```

### Constructor

| Argument   | Default | Description                                            |
| ---------- | ------- | ------------------------------------------------------ |
| `d_node`   | —       | input node feature dim (Linear projection of `x`)      |
| `n_cls`    | —       | output dim (classes or regression targets)             |
| `d_edge`   | 0       | input edge feature dim (>0 enables E-BasePlanE)        |
| `d_hid`    | 64      | hidden dim                                             |
| `n_layers` | 2       | number of PlaneLayers                                  |
| `p_drop`   | 0.0     | dropout probability                                    |
| `d_pe`     | 16      | positional-encoding dim inside `TriEnc`                |

When `d_edge > 0` the model switches to E-BasePlanE: `aggr_neigh` becomes
`GINEConv` so the 1-hop neighbor messages depend on edge features as well.

Naming follows Hungarian-style prefixes throughout the code: `d_*` for
dimensions, `n_*` for counts, `p_*` for probabilities, plus tensor shape
suffixes after `__` (e.g. `h_g__N_D` for node features of shape `[N, D]`).
See the legend at the top of `src/plane/model/layers.py`.

---

## Experiments

Each script in `experiments/` reproduces one task from the paper. See the
matching `.md` in `notes/` for the dataset description, the paper-config
hyperparameter grid, and the exact reproduction commands.

| Script                          | Notes                                | Task                                          |
| ------------------------------- | ------------------------------------ | --------------------------------------------- |
| `experiments/train_p3r.py`      | [`notes/p3r.md`](notes/p3r.md)       | P3R planar 3-regular classification (paper §7.1.2) |
| `experiments/train_zinc.py`     | [`notes/zinc.md`](notes/zinc.md)     | ZINC 12k regression (paper §7.4)              |
| `experiments/train_polaris.py`  | [`notes/polaris.md`](notes/polaris.md) | Any Polaris regression benchmark (default: TDC `caco2-wang`) — bring-your-own benchmark via `--benchmark <owner/slug>` |

Scripts can be run from any cwd — they resolve paths relative to the repo
root (`Path(__file__).resolve().parent.parent`). E.g. from the repo root:

```bash
python experiments/train_p3r.py
python experiments/train_zinc.py --preprocess-only --n-workers 10
python experiments/train_zinc.py
python experiments/train_polaris.py                                     # tdcommons/caco2-wang
python experiments/train_polaris.py --benchmark tdcommons/lipophilicity-astrazeneca
```

For the Polaris script: `pip install polaris-lib rdkit` (optional extras, not
in `requirements.txt`).

### Tests

```bash
pytest tests/        # 4 tests, ~6s wall clock
```

- `test_model.py` — forward shape, backward gradient flow, and **isomorphism
  invariance** on K4 (two node-relabelings of the same graph must produce
  identical model outputs).
- `test_p3r.py` — trains a tiny PlanE on a 3-class P3R subset (30 graphs) for
  50 epochs and asserts test accuracy ≥ 0.83.

---

## Layout

```
src/plane/
  model/
    model.py     # PlanE: node embed -> N x PlaneLayer -> JK readout
    layers.py    # PlaneLayer, TriEnc, BiEnc, CutEnc, make_mlp, PosEnc
  data/
    data_process.py            # planar_preprocess + DataPlanE
    data_process_classical.py  # SPQR / BC-tree canonical encoding (Sage)
experiments/
  train_p3r.py                 # P3R training / eval
  train_zinc.py                # ZINC training / eval, parallel preprocessing
  train_polaris.py             # Polaris benchmark (SMILES via rdkit -> PyG)
  plot_zinc.py                 # render ZINC CSV log to PNG
notes/
  p3r.md                       # P3R: task, grid, repro commands, results
  zinc.md                      # ZINC: task, grid, repro commands, results
  polaris.md                   # Polaris: pipeline, default task, results
tests/
  test_model.py                # unit tests (forward / backward / invariance)
  test_p3r.py                  # P3R-subset training test
```
