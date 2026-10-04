# OnMAR / OffMAR — CNN, clustering and video classification

A bundle of the three applications you asked for, taken from the `OnMAR`
repository (branch `extended-experimentation`) on 2026-10-04.

**The three are not at the same stage, and this matters for what you can run:**

| Application | State | Where |
|---|---|---|
| **CNN configuration** | Working and verified end to end under all four OnMAR/OffMAR variants | `applications/`, `OnMAR/`, `OffMAR/`, `experiments/` |
| **Clustering composition** | Implementation exists but is **not integrated** with the OnMAR/OffMAR framework | `clustering_composition/` |
| **Video classification** | TSN implementation exists but is **not integrated**, and needs a separate model zoo | `video_configuration/` |

Clustering and video were recovered from the repository's git history: both
were written before the framework refactor and removed by commit `68326bbf`
("Additional experimentation"). They are included because they are the only
implementations of those two applications that exist, not because they run
against the current framework. Neither has a `*_application.py` implementing
`BaseApplication`, which is what the meta-learning layer drives.

---

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements-min.txt
```

Python 3.10+ and about 2 GB of disk (datasets download on first use).

Note on `requirements.txt` in the full repo: it does **not** resolve as one
environment, because `auto-sklearn` pins `scikit-learn<0.25` which cannot
coexist with the rest. The AutoSklearn baseline needs its own Python 3.10
environment. `requirements-min.txt` here is the set the CNN application is
verified against.

---

## 2. CNN configuration — the part that works

### Check the installation

```bash
python experiments/test/verify_setup.py
```

### Verify the whole pipeline (recommended first run, ~10 minutes, CPU is fine)

```bash
python experiments/test/verify_cnn_pipeline.py --batches 20 --timesteps 6
```

This checks the design space round-trips through its fixed-length encoding,
that every sampled design builds, that the GA improves, that training state
persists across timesteps, that meta-feature vectors are the same length at
every timestep, and that all four approach variants run end to end. It printed
`ALL CHECKS PASSED` on 2026-10-04 with test accuracies of 0.959–0.982 on MNIST
at that (deliberately tiny) budget.

### Run a single experiment

```bash
cd experiments/test
python runner.py --application cnn --datasets mnist \
    --techniques onmar-accuracy --meta-learners knn \
    --timesteps 80 --runs 1
```

Results land in `experiments/test/results/` as JSON, a summary CSV, and a
thesis-format ranking table.

Datasets available: `mnist`, `fashion-mnist`, `cifar-10`, `cifar-100`.
Techniques: `onmar-accuracy`, `onmar-design`, `offmar-accuracy`,
`offmar-design`, `autosklearn` (the last needs its own environment).

### Run CIFAR-10 directly

```bash
python experiments/test/run_cifar10_onmar.py --timesteps 24 --batches 150
```

Measured on 10 CPU cores: **78.66% test accuracy** in 53 minutes
(133s/timestep) at 24 timesteps × 150 batches. `--batches 0` trains a full
epoch per timestep, which is the thesis configuration and much slower.

### Use it from Python

```python
from metalearner import load_approach
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# The approach directories contain a hyphen, so they load by path.
OnMAR = load_approach('onmar-accuracy').OnMARAccuracyPrediction

onmar = OnMAR(
    application=CNNConfigurationApplication('mnist', random_seed=42),
    meta_learner_type='knn',   # 'knn', 'rf' or 'xgboost'
    theta_t=None,              # N/2, per Algorithm 9
    theta_p=0.85,
)
results = onmar.run_onmar(dataset_name='mnist', timesteps=80)
print(results['performance_history'], results['test_performance'])
```

A design is per-layer, and the number of layers is part of the design:

```python
design = {
    'conv_layers': [
        {'filters': 32, 'batch_norm': 1, 'activation': 3, 'dropout': -1.0, 'max_pool': 2},
        {'filters': 64, 'batch_norm': 1, 'activation': 3, 'dropout': 0.25, 'max_pool': 2},
    ],
    'dense_layers': [{'nodes': 128, 'batch_norm': 1, 'activation': 3, 'dropout': 0.3}],
    'optimizer': 1,            # 1=Adam … 7=NAdam
    'learning_rate': 0.001,
}
```
`dropout: -1.0` means none; `max_pool: 0` means no pooling.

### HPC

```bash
qsub experiments/conf_cnn_hpc.job
```

Scoring GA candidates is 48% of a run's compute and is fully parallel, so it
is spread over worker processes: one per GPU on a multi-GPU node, otherwise a
share of the allocated CPUs with BLAS threads divided between them. Measured
1.59× on 10 CPU cores; the multi-GPU path is written but **has not been run on
a GPU** — no CUDA device was available.

Settings worth knowing:

- `candidate_workers` — pool size; `None` decides automatically, `0` runs serially
- `candidate_eval_threads` — pin threads per evaluation. Fitness is otherwise
  machine-dependent: CPU reductions are not associative, and a different core
  count changes results by up to ~0.06 probe accuracy. Pin it for runs that
  must reproduce across machines.
- `pretrained_init` / `pretrained_donor` — seed the first design's convolutions
  from an ImageNet donor (default `resnet18`). **This made MNIST worse**
  (0.79 vs 0.92 at timestep 0): ImageNet kernels are ~3× smaller in scale than
  a fresh layer's, and only ~4% of parameters are affected. Pass
  `pretrained_init=False` for from-scratch runs.
- `max_candidate_macs` — compute budget per candidate (250M MACs/image). Without
  it, one design with wide unpooled convolutions costs more than every other
  candidate in its generation combined. Raise it on a GPU.

---

## 3. Clustering composition — not integrated

`clustering_composition/` holds the thesis's clustering components, one module
per design option, matching the chromosome in
`reference/thesis_chromosomes.py`:

```
cluster_initialization.py  cluster_creation.py   cluster_addition.py
cluster_removal.py         cluster_merging.py    cluster_splitting.py
stopping_criteria.py       feature_extraction.py cluster.py
utils.py                   run.py
```

`run.py` is the entry point these were driven through. `sample_runs/` holds
output from six earlier runs (three static-design, three dynamic-design), which
is useful for seeing what the components produce.

**To use it with OnMAR/OffMAR** it needs a `ClusteringCompositionApplication`
subclassing `BaseApplication` — the same shape as
`applications/configuration/cnn/cnn_application.py`, implementing `load_data`,
`get_design_space`, `exec_design`, `evaluate` and `extract_meta_features`.
The design space is the eleven options in `reference/thesis_chromosomes.py`
under `clustering_composition`. Until that exists, the meta-learning layer has
nothing to drive.

---

## 4. Video classification — not integrated

`video_configuration/` is a Temporal Segment Networks (TSN) implementation in
PyTorch: `models.py`, `dataset.py`, `transforms.py`, `ops/`, `opts.py`, plus
`main.py`/`run.py`/`test_models.py` as its own training and testing entry
points.

One change was already applied: `main.py` used `tensor.cuda(async=True)`,
which stopped parsing when `async` became a reserved word in Python 3.7. It is
now `non_blocking=True`, PyTorch's rename for the same argument (2 occurrences).
Every module in `video_configuration/` parses under Python 3.12; none has been
*run*, since there is no video dataset loader in the repository.

Two things it still needs:

1. **The model zoo.** `models.py` lazily imports `tf_model_zoo` for
   BN-Inception and InceptionV3 base models. That directory is 442 files and
   51.8 MB, so it is **not** in this bundle. Recover it from the repository
   with:
   ```bash
   git checkout 20e8a896 -- applications/video_configuration/tf_model_zoo
   ```
   ResNet base models do not need it.

2. **An application wrapper**, as for clustering. The thesis design space for
   video is the seven options under `video_configuration` in
   `reference/thesis_chromosomes.py` (keyframe extraction, number of segments,
   base architecture, consensus function, learning rate, dropout, gradient norm
   clipping).

Video datasets (UCF101, HMDB51, LMTD) are referenced by the original CLI but
no loader for them exists in the repository — `datasets/image_datasets.py`
covers MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 only.

---

## 5. `reference/`

- `thesis_chromosomes.py` — the original `genetic_algorithm/individual.py`,
  holding the hand-written chromosome definitions for all three applications.
  The current framework replaced it with a generic, design-space-driven
  individual, so this file is the only record of the video and clustering
  design spaces. It does not run (it has a `_init_` typo and imports modules
  that no longer exist); keep it as a specification.
- `original_main.py` — the original CLI sketch, showing the intended
  three-application structure and timestep budgets (video 100, CNN 80,
  clustering 60). Does not parse; kept for the budgets and structure.
- `original_metalearner.py` — the original meta-learner with its three
  predictors (y-from-features, c-from-features, y-from-c).

---

## 6. Known limitations

Carried over from the full repository, so they are not surprises:

- **The GA matches but does not beat a hand-picked design.** On MNIST, seeded:
  0.9898 test versus 0.9912 for the hand-picked design. On CIFAR-10 it made
  **zero** design changes in 24 generations and 144 evaluations — the result is
  the seed design trained for 24 timesteps. The design algorithm needs a
  larger evaluation budget than a short run gives it.
- **θp = 0.85 is unreachable on CIFAR-10** at these budgets, so OnMAR's
  reuse-versus-recreate decision never fires and the approach degenerates to
  plain search. Set `--theta-p` to something attainable (0.65–0.70 for CIFAR-10
  here) for the mechanism to be exercised.
- **Pretrained initialisation is unproven.** It hurt on MNIST; the controlled
  CIFAR-10 comparison was started but not finished.
- GPU and multi-GPU paths are written but untested — no CUDA device was
  available.
