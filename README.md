# OnMAR / OffMAR — CNN, clustering and video classification

A bundle of the three applications you asked for, taken from the `OnMAR`
repository (branch `extended-experimentation`) on 2026-10-04.

All three are now integrated with the OnMAR/OffMAR framework and run end to
end under all four variants:

| Application | State | Where |
|---|---|---|
| **CNN configuration** | Working and verified end to end under all four OnMAR/OffMAR variants | `applications/configuration/cnn/`, `OnMAR/`, `OffMAR/`, `experiments/` |
| **Clustering composition** | Working and verified end to end under all four variants | `applications/composition/clustering/`, `clustering_composition/` |
| **Video classification** | Working and verified end to end under all four variants, against a generated dataset | `applications/configuration/video/`, `video_configuration/` |

Clustering and video were recovered from the repository's git history: both
were written before the framework refactor and removed by commit `68326bbf`
("Additional experimentation"), so they predate the `BaseApplication`
interface the meta-learning layer drives. Each now has an application wrapper
implementing it, and the components behind those wrappers needed real repairs
to run at all — §7 lists every one, because several were silently wrong rather
than loudly broken and the numbers they produced before cannot be compared
with the numbers they produce now.

Two caveats that do not go away:

- **Video has no real dataset here.** UCF101, HMDB51 and LMTD need to be
  downloaded and frame-extracted, and this repository has no loader that will
  do it. The video application ships a *generated* dataset in the same on-disk
  layout, so the pipeline is genuinely exercised — but it is a smoke test, not
  a benchmark. §4 covers pointing it at real data.
- **Some design options need extra libraries.** Both new applications drop the
  options they cannot execute from the design space and report them, rather
  than letting the search sample designs that crash. §1 lists the installs.

---

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements-min.txt
```

That covers the CNN and video applications. Clustering runs on it too, but
with a reduced design space — ten of its options need libraries that are not
in the minimal set:

```bash
pip install -r requirements-clustering.txt    # the full clustering design space
pip install -r requirements-video.txt         # video (mostly already satisfied)
```

Python 3.10+ and about 2 GB of disk (datasets download on first use); add
~1 GB if you install TensorFlow for the clustering feature extractors.

**Missing libraries narrow a design space, they do not break it.** Both new
applications check what is importable when they build their design space, drop
the options they cannot execute, warn once, and list them in
`get_run_statistics()['excluded_design_options']`. That is deliberate: the
design space is categorical and the GA samples across all of it, so an
unavailable option would otherwise surface as a crash somewhere inside
generation three. Pass `strict_design_space=True` to demand the full space and
fail at construction instead.

What needs what:

| Needs | For |
|---|---|
| `tensorflow`, `keras` | clustering `feature_extraction` 0–4 (ResNet50, InceptionV3, DenseNet121, Xception, VGG16) |
| `opencv-contrib-python` | clustering `feature_extraction` 6 (SIFT) and 8 (BRIEF). BRIEF needs `cv2.xfeatures2d`, which is in the contrib build only — plain `opencv-python` gives you SIFT but not BRIEF |
| `kneed` | clustering `cluster_creation` 2 (DBSCAN) |
| `kmedoids` | clustering `cluster_creation` 4 (k-medoids) |
| `markov_clustering`, `networkx` | clustering `cluster_addition` 4 |
| `tf_model_zoo` (not on PyPI) | video `base_architecture` 8 and 9 (BN-Inception, InceptionV3) — see §4 |

Note on `requirements.txt` in the full repo: it does **not** resolve as one
environment, because `auto-sklearn` pins `scikit-learn<0.25` which cannot
coexist with the rest. The AutoSklearn baseline needs its own Python 3.10
environment. `requirements-min.txt` here is the set the CNN application is
verified against.

---

## 2. CNN configuration

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

Re-run after the clustering and video integration — which touched
`datasets/image_datasets.py` and `OffMAR/design-prediction/offmar.py`, both
shared with this application — it still printed `ALL CHECKS PASSED`, at
`--batches 10 --timesteps 4`. That budget crashed before the k-NN fix in §7.

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

## 3. Clustering composition

A *composition* application: a design does not tune one algorithm's
hyperparameters, it selects which components the clustering algorithm is built
from. `clustering_composition/` holds those components, one module per design
option, matching the eleven-gene chromosome in
`reference/thesis_chromosomes.py`:

```
cluster_initialization.py  cluster_creation.py   cluster_addition.py
cluster_removal.py         cluster_merging.py    cluster_splitting.py
stopping_criteria.py       feature_extraction.py cluster.py
label_utils.py             optional_deps.py      utils.py        run.py
```

`applications/composition/clustering/clustering_application.py` wraps them in
`BaseApplication`. `sample_runs/` holds output from six earlier runs (three
static-design, three dynamic-design) — but see §7 before comparing anything
against it.

### Verify the pipeline (~15 minutes, CPU is fine)

```bash
python experiments/test/verify_clustering_pipeline.py
```

Seven checks: the encoding round-trips, **every value of every one of the
eleven genes runs for two timesteps**, each feature extractor really extracts
rather than silently falling back to raw pixels, cluster identifiers are
distinct, the clustering accumulates across timesteps and survives a change of
feature space, meta-feature vectors are a constant length, and all four
approaches run end to end. `--skip-options` drops the slowest check;
`--skip-approaches` drops the four runs.

### Run an experiment

```bash
cd experiments/test
python runner.py --application clustering --datasets mnist \
    --techniques onmar-accuracy --meta-learners knn \
    --timesteps 60 --runs 1
```

Datasets: `mnist`, `fashion-mnist`, `cifar-10`, `cifar-100`. The thesis
timestep budget for this application is 60.

### Use it from Python

```python
from metalearner import load_approach
from applications.composition.clustering.clustering_application import (
    ClusteringCompositionApplication,
)

OnMAR = load_approach('onmar-accuracy').OnMARAccuracyPrediction

onmar = OnMAR(
    application=ClusteringCompositionApplication(
        'mnist',
        random_seed=42,
        num_instances=128,       # instances clustered, not a batch size
        num_test_instances=128,  # held out for evaluate()
    ),
    meta_learner_type='knn',
    theta_t=None,                # N/2, per Algorithm 9
    theta_p=0.65,                # see below — 0.85 is unreachable here
)
results = onmar.run_onmar(dataset_name='mnist', timesteps=60)
```

A design is eleven integers, each selecting a component:

```python
design = {
    'distance_metric': 0,          # 0-7:  Euclidean, Manhattan, Minkowski, …
    'cluster_initialization': 2,   # 0-7:  none, single, k-means++, Birch, OPTICS, …
    'cluster_creation': 0,         # 0-6:  mini-batch k-means, mean shift, DBSCAN, …
    'cluster_addition': 0,         # 0-4:  nearest centroid, Gaussian, Markov, …
    'cluster_removal': 0,          # 0-5:  none, overlapping, largest-distance, …
    'cluster_merging': 0,          # 0-3:  none, by distance, fewest members, random
    'cluster_splitting': 0,        # 0-6:  none, Ward, eigenvalue, …
    'membership_limit': 0,         # 0 = one cluster per instance, 1 = overlap allowed
    'stopping_criteria': 0,        # 0 = assignments settled, 1 = centroid variance settled
    'step': 0,                     # which operator runs this timestep (0-4, see below)
    'feature_extraction': 5,       # 0-8:  CNN features, PCA, SIFT, t-SNE, BRIEF
}
```

The option ranges above are the thesis's. The recovered components implement
more than the thesis declares for three genes — 8 cluster-creation options,
16 cluster-splitting options, and a tenth raw-pixel feature extractor — and
all of them run; pass `extra_options=True` to search them too. The default is
the thesis space so that results stay comparable with it.

Two genes deserve attention:

- **`step`** is what makes this a composition application. It selects which
  *single* composition operator is applied at this timestep — 0 creation,
  1 addition, 2 removal, 3 merging, 4 splitting. The other operator genes
  still choose *how* their operator would behave, but only the one `step`
  names actually runs.
- **`feature_extraction`** dominates the cost. A ResNet50 pass over the
  instance sample takes seconds; PCA takes milliseconds. Features are cached
  per extractor, so the GA pays for each extractor once per run rather than
  once per candidate — without that cache, candidate scoring was the whole
  run. The seeded default design uses PCA for the same reason.

### What a run looks like

Measured on an 8-core laptop (MNIST, 48 instances, 8 timesteps,
OnMAR-accuracy with k-NN, θp = 0.60): clustering accuracy rose 0.583 → 0.667
over the run, held-out accuracy 0.396, 4 design-algorithm calls and 4 reuses,
32 fitness evaluations, 19.5 s total. The GA moved the design from PCA
features to a 2048-dimensional CNN extractor over those timesteps.

`verify_clustering_pipeline.py` at the same budget (θp = 0.65, 6 timesteps)
printed `ALL CHECKS PASSED` on 2026-10-04, with held-out accuracy 0.4375
(OnMAR-accuracy), 0.4531 (OnMAR-design), 0.2031 (OffMAR-accuracy) and 0.2812
(OffMAR-design) — against 0.1 for chance on MNIST's ten classes.

θp is **0.65** in `CLUSTERING_CONFIG`, not the thesis's 0.85, and that is a
deliberate departure. θp is the predicted-accuracy threshold above which OnMAR
reuses its design instead of searching again, so it has to sit inside the range
the application actually reaches. Clustering accuracy measured here runs to
about 0.60–0.67 on MNIST — at 0.85 the reuse branch would never fire and OnMAR
would degenerate into plain search, which is the same failure §6 records for
CIFAR-10. Raise it toward 0.85 only if a larger instance sample lifts the
accuracies with it.

### Read held-out clustering accuracy carefully

Held-out accuracy can sit far below training accuracy for a reason that is
about design coherence, not overfitting, and it is worth understanding before
reporting either number.

A cluster's centroid is an arithmetic mean (`cluster.set_centroid`), which is
the *Euclidean* representative point. But several creation components —
mini-batch k-means, mean shift — are Euclidean-only in scikit-learn and ignore
the `distance_metric` gene when forming clusters. And training accuracy is
computed from stored membership, so it never consults the metric either.
Held-out assignment, which puts each new instance with its nearest centroid,
is the first place the gene is actually used.

Measured on MNIST at 128 instances with the default design, varying only the
metric:

| `distance_metric` | Training accuracy | Held-out accuracy | Held-out ARI |
|---|---|---|---|
| 0 Euclidean | 0.6406 | 0.5703 | +0.352 |
| 4 Cosine | 0.6406 | 0.5547 | +0.348 |
| 1 Manhattan | 0.6406 | 0.5234 | +0.264 |
| 3 Hamming | 0.6406 | 0.1250 | +0.000 |
| 6 Canberra | 0.6406 | 0.0938 | +0.006 |

Training accuracy is **identical for every metric**. Held-out accuracy spans
0.09 to 0.57. Two consequences:

- The design algorithm optimises training accuracy, so it receives *no signal
  at all* about this gene and leaves it to drift. A search can therefore
  return a design whose held-out accuracy is near chance while its training
  accuracy looks healthy — which is exactly what one 6-timestep OnMAR run
  did, ending on Canberra with PCA features: training 0.5703, held-out
  0.0703.
- Held-out assignment honours the gene, which is faithful — the framework's
  own `add_to_cluster_using_centroids` uses it the same way — but it means a
  held-out number partly reflects a gene the search could not optimise.

`evaluate()` therefore also returns **`test_accuracy_euclidean`**: the same
assignment under Euclidean distance. Compare the two. A large gap means the
design's metric is incoherent with the geometry its clusters were built in,
not that the clustering failed. This is a property of the thesis's setup
rather than something introduced here, so the fitness function is left alone —
but it is reported rather than buried.

### Design changes and the feature space

A design change can change `feature_extraction`, and then the clusters carried
over from the previous timestep describe points in a feature space that no
longer exists. The application handles this the way the CNN application handles
an architecture change: a cluster *is* fundamentally a set of member instances,
and membership is a property of the data rather than of the representation, so
membership is kept and the vectors and centroids are recomputed in the new
space. A change of representation costs the run its centroid geometry but not
the composition it has built up.

---

## 4. Video classification

`video_configuration/` is a Temporal Segment Networks (TSN) implementation in
PyTorch — `models.py`, `dataset.py`, `transforms.py`, `ops/`, `opts.py` — now
with `keyframes.py` (the five segment-sampling strategies the design space
asks for), `synthetic.py` (a generated dataset) and `run.py` (which was a
zero-byte file).
`applications/configuration/video/video_application.py` wraps it in
`BaseApplication`.

### Verify the pipeline (~15 minutes, CPU is fine)

```bash
python experiments/test/verify_video_pipeline.py
```

Seven checks: the encoding round-trips, the generated dataset and its loader
agree, all five keyframe strategies return in-range indices, every base
architecture and every consensus function builds and backpropagates, training
state accumulates and transfers across a design change, meta-feature vectors
are a constant length, and all four approaches run end to end.
`--skip-architectures` skips downloading several hundred MB of pretrained
weights; `--skip-approaches` drops the four runs.

It printed `ALL CHECKS PASSED` on 2026-10-04 with `--skip-architectures`
(24 train / 8 held-out videos, 4 timesteps, θp = 0.85): held-out accuracy
1.000 from all four approaches, in 134 s, 352 s, 170 s and 126 s
respectively. The generated task is easy, so that is a liveness result rather
than a comparison between the approaches. The architecture check was run
separately — all five ResNets and both VGGs build and run forward (11.2 M to
58.2 M parameters); BN-Inception and InceptionV3 were skipped, since
`tf_model_zoo` is not bundled.

### Run an experiment

```bash
cd experiments/test
python runner.py --application video --datasets synthetic \
    --techniques onmar-accuracy --meta-learners knn \
    --timesteps 100 --runs 1
```

The thesis timestep budget for this application is 100.

### Use it from Python

```python
from metalearner import load_approach
from applications.configuration.video.video_application import (
    VideoConfigurationApplication,
)

OnMAR = load_approach('onmar-accuracy').OnMARAccuracyPrediction

onmar = OnMAR(
    application=VideoConfigurationApplication('synthetic', random_seed=42),
    meta_learner_type='knn',
    theta_t=None,
    theta_p=0.85,
)
results = onmar.run_onmar(dataset_name='synthetic', timesteps=100)
```

A design is the seven options of the thesis video chromosome:

```python
design = {
    'keyframe_extraction': 1,       # 1-5: random, centre, uniform, dense, strided
    'num_segments': 3,              # 2-10
    'base_architecture': 1,         # 1-9: resnet18/34/50/101/152, vgg16/19,
                                    #      BNInception, InceptionV3
    'consensus_function': 1,        # 1-5: avg, max, topk, identity, weighted
    'learning_rate': 0.001,         # 0.0009 - 0.01
    'dropout': 0.5,                 # 0.4 - 0.65
    'gradient_norm_clipping': 20.0, # 1.0 - 20.0
}
```

Two options need a word of explanation:

- **`consensus_function` 4 (`identity`)** applies no consensus, so the model
  returns one score vector *per segment* rather than per video. The
  application then supervises each segment separately — the mean of the
  per-segment cross-entropies — and predicts by averaging the per-segment
  probabilities. That is deliberately *not* the same objective as
  cross-entropy of the averaged scores, which is what `avg` gives, so the two
  options stay genuinely distinct.
- **`consensus_function` 5 (`weighted`)** is a learnable per-segment
  weighting. The original command line called its fifth option `rnn`, but no
  recurrent consensus was ever implemented here — `ConsensusModule` aliased
  `rnn` to `identity`, making two of the five options the same thing. Rather
  than ship a duplicate, the fifth option is the simplest consensus that
  actually differs from the other four by *learning* how to combine segments.
  `'rnn'` is still accepted as an alias. See §7.

### The generated dataset

There is no video dataset in this repository and no loader that would build
one: `datasets/image_datasets.py` covers MNIST, Fashion-MNIST, CIFAR-10 and
CIFAR-100 only, and UCF101, HMDB51 and LMTD are tens of gigabytes that have to
be downloaded and frame-extracted first. That, rather than the missing
wrapper, is what made this application unrunnable.

`video_configuration/synthetic.py` generates one: four shapes (disc, square,
ring, cross) on a noisy background, each following its own motion, written to
disk as numbered frame directories plus `train.txt` / `val.txt` list files —
**the same layout `TSNDataSet` reads for UCF101**, so the code path verified
here is the real one rather than a mock of it.

It is a smoke test, not a benchmark. Measured on an 8-core laptop with the
default design (resnet18, 3 segments, average consensus, lr 0.001) over 20
timesteps on 48 training and 16 held-out videos: validation accuracy rose
0.438 → 1.000 by timestep 7 and held 1.000 thereafter, training loss fell from
1.42 to below 0.2, and held-out accuracy finished at 1.000. It saturates, so
it demonstrates that a design trains — it will not discriminate between good
designs. Raise `synthetic_classes` (up to 6) and `noise` to make it harder.

One thing worth knowing before designing your own variant: it is tempting to
make motion *direction* the class signal, so only temporal information could
separate the classes. That dataset would report chance accuracy for every
design in the space, and not because of a defect — every consensus function
TSN offers (average, max, top-k, learned weighting) is **order-invariant**, so
a leftward and a rightward traversal of the same path aggregate identically.
That is a property of TSN, which trades temporal ordering for cheap long-range
coverage.

### Pointing it at a real dataset

Extract frames, write the list files (`<frame directory> <num frames> <class
index>` per line), then:

```python
app = VideoConfigurationApplication(
    'custom',
    train_list='data/ucf101/train.txt',
    val_list='data/ucf101/val.txt',
    num_classes=101,       # inferred from the list files if omitted
    image_tmpl='img_{:05d}.jpg',
    batch_size=16,
)
```

Lower θp when you do: 0.85 is reachable on the generated set but will not be
on a real benchmark at short-run budgets, and the reuse branch would never
fire.

### The model zoo

Base architectures 8 (BN-Inception) and 9 (InceptionV3) are defined in a
`tf_model_zoo` directory of 442 files and 51.8 MB that is **not** in this
bundle. Recover it with:

```bash
git checkout 20e8a896 -- applications/video_configuration/tf_model_zoo
```

Without it those two architectures are dropped from the design space and
reported in `get_run_statistics()['excluded_design_options']`; the five ResNets
and two VGGs need nothing extra.

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
  here) for the mechanism to be exercised. The same applies to clustering,
  where `CLUSTERING_CONFIG` already ships 0.65 for this reason (§3), and to
  video on any real dataset (§4).
- **Pretrained initialisation is unproven.** It hurt on MNIST; the controlled
  CIFAR-10 comparison was started but not finished.
- GPU and multi-GPU paths are written but untested — no CUDA device was
  available.

New to the two applications integrated here:

- **Video has no real benchmark result.** Everything reported for it was
  measured on the generated dataset (§4), which saturates at 1.000. The
  implementation is verified; its accuracy on UCF101 or HMDB51 is unmeasured.
- **Clustering is small by default.** 128 instances, because several
  components are superlinear in the instance count — spectral clustering
  builds a full affinity matrix. The thesis numbers are not reproducible at
  this scale; raise `num_instances` and expect the cost to grow faster than
  linearly.
- **PCA and t-SNE features are transductive.** Neither recovered extractor
  exposes a fitted transform, so held-out instances cannot be projected into a
  training-set embedding. Both sets are therefore feature-extracted together,
  which means the basis is fitted over the held-out *inputs* as well (never
  their labels). For t-SNE there is no alternative — it has no out-of-sample
  extension at all. Worth stating when reporting held-out clustering accuracy.
- **The clustering `distance_metric` gene is invisible to the search.**
  Training accuracy does not depend on it, so the design algorithm cannot
  optimise it, while held-out accuracy depends on it heavily (0.09–0.57 on
  MNIST for otherwise identical designs). §3 has the measurements and the
  `test_accuracy_euclidean` diagnostic. This is in the thesis's setup, not
  introduced here.
- **Clustering accuracy can be −1.** That is the components' own sentinel for
  a degenerate partition (nearly every instance in its own cluster), not an
  error. It propagates into the performance history as a legitimately bad
  score.
- **Very short runs starve the k-NN meta-learner.** The OffMAR variants split
  the timestep budget in two, so Phase 1 contributes only `timesteps/2`
  entries to the knowledge repository. Where that is fewer than `k`, `k` is
  clamped to what is available — OnMAR (both variants) and OffMAR-accuracy
  already did this; OffMAR-design did not and crashed with
  `Expected n_neighbors <= n_samples_fit`, which is fixed here (§7). The
  meta-learner is correspondingly weakly informed at such budgets, so keep
  `timesteps` comfortably above `2k` for any run whose numbers matter.

---

## 7. Repairs made during integration

Neither recovered application ran. Most of what follows is listed not because
it was hard to fix but because **several of these failed silently** — they
produced plausible numbers from code that was not doing what it claimed. Any
earlier clustering results, `clustering_composition/sample_runs/` included,
were produced under the first four of these and should not be compared with
current output.

### Clustering, silent failures

1. **Cluster identifiers collided about half the time.** Every component minted
   identifiers as `[str(time.time()).replace('.','') for ul in unique_labels]`.
   A list comprehension iterates faster than the clock ticks: measured here, 10
   labels produced 7 distinct identifiers and 50 produced 25. Clusters are
   matched to their members *by identifier*, so two clusters sharing one
   silently became a single merged cluster holding the union of their members —
   roughly half of all clusters were lost at every composition step. Replaced
   with a counter (`label_utils.new_identifiers`).
2. **PCA and t-SNE never ran.** `feature_extraction.py` used `PCA`, `TSNE`,
   `math` and `cv2` without importing any of them, and every extractor is
   wrapped in a bare `except` that returns the input unchanged — so the
   `NameError` was swallowed and those designs clustered raw pixels. Both also
   requested more components than the sample count allows, which would have
   failed the same way once imported. Imports added, component counts bounded
   by `min(n_samples, n_features)`, and the fallback now warns instead of
   passing silently.
3. **Both stopping criteria always reported "converged".**
   `check_for_changes_in_cluster_assignments` initialised its match vector to
   `[True] * n` and only ever assigned `True` to it, so its `False in checks`
   test could never fire. `check_for_changes_in_cluster_variance` compared
   `state[-2]` against `state[-2]` — the same checkpoint twice. Since a
   "converged" verdict *skips* the timestep's composition operator, both
   criteria froze the clustering from the second timestep onward.
4. **SIFT and BRIEF never ran either**, for a different reason: OpenCV needs
   8-bit images and was handed float32. Same silent fallback. Now cast.

### Clustering, loud failures

5. `from applications.clustering_composition.*` throughout — a package path
   that does not exist in this bundle. And `from dataset.utils import ...` for
   two functions whose module is not here at all; `cluster_creation.py` carried
   a private copy, every other component imported the missing one. Collected
   into `label_utils.py`.
6. `k-medoids` was handed the **data matrix** where `fasterpam` requires a
   square dissimilarity matrix — the line building it was commented out. The
   Rust extension aborted the process (`assertion left == right failed: 48 !=
   784`) rather than raising something catchable. Restored, which also makes
   that component honour the distance-metric gene.
7. **The pretrained CNN extractors rejected greyscale.** They widened grey to
   RGB only for 2-D `(H, W)` input; the project's loaders produce `(H, W, 1)`,
   so all five failed with *"The input must have 3 channels"*.
8. Markov clustering broke on `networkx` ≥ 3.0 (`to_scipy_sparse_matrix`
   removed) and then again on the replacement, which returns a sparse *array*
   — `markov_clustering` dispatches on `isspmatrix`, false for arrays, so it
   took its dense path and called `np.linalg.matrix_power` on a sparse object.
9. Optional dependencies were imported at module scope, so a missing `kneed`
   took out all seven cluster-creation options rather than the one that needs
   it. Now resolved per use, via `optional_deps.py`.

### Video

10. **The consensus module could not run at all.** `SegmentConsensus`
    subclassed `torch.autograd.Function` in the pre-0.4 style with instance
    methods; PyTorch removed support, so every forward pass raised *"Legacy
    autograd function with non-static forward method is deprecated"*.
    Rewritten as an `nn.Module` over differentiable ops, so autograd derives
    the backward pass instead of it being hand-written.
11. **Three of the five consensus functions were not implemented.** `forward`
    handled only `avg` and `identity` and returned `None` for the rest, while
    `ConsensusModule` quietly rewrote `rnn` to `identity` — so `max` and
    `topk` crashed and `rnn` was a silent duplicate of `identity`. `max` and
    `topk` are now real; the fifth option is a learnable per-segment weighting
    (`weighted`), since no recurrent consensus was ever implemented here and
    reproducing a duplicate seemed worse than documenting a substitute.
12. **VGG base architectures raised immediately.** TSN swaps a single named
    attribute for its own head and the code set `last_layer_name = 'fc'`;
    torchvision's VGG keeps its head in `.classifier`. Fixed, at the cost of
    VGG's pretrained fc6/fc7 blocks — only its convolutional features are
    reused.
13. **`run.py` was a zero-byte file** and `__init__.py` did `from run import
    run`, an absolute import of a sibling that resolved only when
    `video_configuration/` was itself the working directory. Same for
    `ops/__init__.py` and `models.py`.
14. **Keyframe extraction was not a design option.** `TSNDataSet` hard-coded
    three samplers and chose between them by training mode, not by design. The
    design space declares five strategies; they live in `keyframes.py` now and
    are selected by gene value. Strategy 1 is the old training sampler,
    strategy 2 the old evaluation one.
15. `torchvision.transforms.Scale` (removed in favour of `Resize`),
    `torch.ByteStorage.from_buffer`, `torch.nn.init.normal`/`constant`, and
    positional `pretrained=True` — all deprecated or removed API.
16. The learnable consensus's parameters tripped `get_optim_policies`, which
    raises on any leaf module carrying parameters it does not recognise.

### Framework

17. **OffMAR-design crashed on short runs.** Its k-NN meta-learner was
    constructed with the configured `k` unclamped, while the other three
    approaches all clamp it to `min(k, len(X))`. Phase 1 contributes only
    `timesteps/2` repository entries, so any run with `timesteps < 2k` died
    with `Expected n_neighbors <= n_samples_fit` — in every application, the
    CNN included. Clamped to match its three siblings.
18. `ImageDatasetLoader` gained `load_arrays`, because clustering needs a
    fixed sample as arrays rather than loaders — it clusters the same
    instances across every timestep — and needs them *un-normalised*, since
    the Keras feature extractors apply their own `preprocess_input` expecting
    raw 0–255 pixels. `load_dataset` is unchanged; the dataset construction it
    shares with `load_arrays` was factored into one helper.
19. **A misleading pruning message.** All three design-prediction variants
    reported *"nothing reached θp"* whenever the repository was relaxed, but
    the relaxation triggers when *fewer than three* entries pass, not when
    none do — so a run could print "nothing reached θp=0.85 (best was
    1.0000)". Reworded to "too few entries reached θp"; the logic was already
    correct.
20. A `.gitignore` was added. `data/` reaches ~435 MB after one clustering run
    and one video run, all of it regenerable, and there was nothing stopping
    it being committed.
