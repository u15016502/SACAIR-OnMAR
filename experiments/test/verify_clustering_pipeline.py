"""
Clustering Composition Pipeline Verification
============================================

Checks that the clustering composition application and all four meta-learning
variants actually work, at a budget small enough to run on a laptop in a few
minutes. Run this after changing the clustering components, the design space,
the GA, the application, or any of the four approaches.

What it checks, and why each check is here:

  1. Design space    - designs round trip through the fixed-length encoding,
                       and arbitrary vectors still decode to buildable designs
                       (the meta-learner's output is unconstrained regression)
  2. Every option    - every value of every one of the eleven genes runs for
                       two timesteps. This is the check that matters most
                       here: the design space is entirely categorical, so the
                       GA will eventually sample every option, and several of
                       the recovered components only ever ran under the one
                       design the original driver used
  3. Feature extractors - each extractor returns features of its own width,
                       rather than silently falling back to raw pixels. Four
                       of them did exactly that, because the module used PCA,
                       TSNE, math and cv2 without importing any of them and a
                       bare 'except' swallowed the NameError
  4. Cluster identifiers - clusters built in one step get distinct
                       identifiers. They are matched to their members by
                       identifier, and the recovered code minted them from
                       time.time() inside a list comprehension, which
                       collided about half the time and silently merged
                       clusters
  5. Persistent state - the clustering accumulates across timesteps, and a
                       change of feature extractor reprojects the clusters
                       rather than discarding them
  6. Meta-features   - the vector length is identical at every timestep
                       (ragged vectors cannot be fitted by a meta-learner)
  7. Approaches      - all four run end to end and report a test performance

Usage:
    python experiments/test/verify_clustering_pipeline.py
    python experiments/test/verify_clustering_pipeline.py --skip-approaches
    python experiments/test/verify_clustering_pipeline.py --instances 64
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time
import warnings

import numpy as np

from metalearner import load_approach
from applications.composition.clustering.clustering_application import (
    ClusteringCompositionApplication, CHROMOSOME_ORDER, THESIS_OPTIONS,
)
from clustering_composition import label_utils, optional_deps
from clustering_composition.utils import get_feature_extraction


# The operator genes, and the 'step' value that selects each one.
OPERATOR_GENES = [
    'cluster_creation',
    'cluster_addition',
    'cluster_removal',
    'cluster_merging',
    'cluster_splitting',
]


class CheckFailed(Exception):
    """A verification check failed."""


def check(condition, message):
    """Assert a condition with a readable failure."""
    if not condition:
        raise CheckFailed(message)


def make_app(instances, test_instances=64):
    """Build an application at the verification budget."""
    return ClusteringCompositionApplication(
        'mnist',
        random_seed=42,
        num_instances=instances,
        num_test_instances=test_instances,
    )


def verify_design_space(app):
    """Designs must survive the encode/decode round trip, and noise must decode."""
    print("\n[1/7] Design space")

    space = app.get_structured_design_space()
    rng = np.random.default_rng(0)
    print(f"  encoding length: {space.encoding_length}")
    print(f"  genes: {len(CHROMOSOME_ORDER)}")
    if app.excluded_options:
        print(f"  options dropped for missing dependencies: {app.excluded_options}")
        print(f"  missing: {optional_deps.unavailable_report()}")

    for _ in range(200):
        design = space.sample(rng)
        check(space.validate(design), f"sampled design invalid: {design}")

        encoded = space.encode(design)
        check(
            encoded.shape == (space.encoding_length,),
            f"encoding length {encoded.shape} != {space.encoding_length}",
        )

        decoded = space.decode(encoded)
        check(space.validate(decoded), f"decoded design invalid: {decoded}")
        check(
            decoded == design,
            f"round trip changed an all-categorical design:\n    {design}\n    {decoded}",
        )

    # The meta-learner's design prediction is unconstrained regression, so
    # arbitrary vectors must still decode to something runnable.
    for _ in range(200):
        noise = rng.normal(size=space.encoding_length).astype(np.float32)
        decoded = space.decode(noise)
        check(space.validate(decoded), f"noise decoded to an invalid design: {decoded}")

    chromosome = app.design_to_chromosome(app.get_default_design())
    check(
        len(chromosome) == 11,
        f"the clustering chromosome must have 11 genes, got {len(chromosome)}",
    )
    check(
        app.chromosome_to_design(chromosome) == app.get_default_design(),
        "design -> chromosome -> design is not the identity",
    )

    print("  design space: OK")
    return space


def verify_every_option(app):
    """Every value of every gene must run for two timesteps."""
    print("\n[2/7] Every design option")

    base = app.get_default_design()
    ran = 0
    excluded = []

    for gene in CHROMOSOME_ORDER:
        for value in THESIS_OPTIONS[gene]:
            if value in app.excluded_options.get(gene, []):
                excluded.append(f"{gene}={value}")
                continue

            design = dict(base)
            design[gene] = value
            # Point 'step' at the operator being exercised, or the operator
            # gene under test would have no effect on the timestep.
            if gene in OPERATOR_GENES:
                design['step'] = OPERATOR_GENES.index(gene)

            app.reset_run_state()
            try:
                app.exec_design(design, 0)
                app.exec_design(design, 1)
            except Exception as error:
                raise CheckFailed(
                    f"design option {gene}={value} failed: "
                    f"{type(error).__name__}: {error}"
                )
            ran += 1

    print(f"  ran {ran} options over two timesteps each")
    if excluded:
        print(f"  skipped (dependencies missing): {excluded}")
    print("  every option: OK")


def verify_feature_extractors(app):
    """Each extractor must produce its own features, not fall back to raw pixels."""
    print("\n[3/7] Feature extractors")

    # The layout run() produces: (N, W, H, C).
    images = app.train_images.reshape(
        (app.train_images.shape[0], app.train_images.shape[-1],
         app.train_images.shape[-2], app.train_images.shape[-3])
    )

    checked = 0
    for value in THESIS_OPTIONS['feature_extraction']:
        if value in app.excluded_options.get('feature_extraction', []):
            print(f"  fe={value}: skipped (dependency missing)")
            continue

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            features = np.asarray(get_feature_extraction(value, images))
            fell_back = [
                str(w.message) for w in caught
                if 'returning the input unchanged' in str(w.message)
            ]

        # Raised rather than passed to check(), because check()'s message is
        # built eagerly and fell_back is empty on the happy path.
        if fell_back:
            raise CheckFailed(
                f"feature extractor {value} fell back to raw pixels: "
                f"{fell_back[0]}"
            )
        check(
            features.shape[0] == images.shape[0],
            f"feature extractor {value} returned {features.shape[0]} rows "
            f"for {images.shape[0]} instances",
        )

        width = int(np.prod(features.shape[1:]))
        print(f"  fe={value}: width {width}")
        checked += 1

    check(checked > 0, "no feature extractor was available to check")
    print("  feature extractors: OK")


def verify_cluster_identifiers():
    """Identifiers minted in one burst must all be distinct."""
    print("\n[4/7] Cluster identifiers")

    label_utils.reset_identifiers()
    identifiers = label_utils.new_identifiers(500)
    check(
        len(set(identifiers)) == 500,
        f"only {len(set(identifiers))} of 500 identifiers were distinct; "
        f"clusters sharing an identifier are silently merged",
    )
    check(
        all(identifier.lstrip('-').isdigit() for identifier in identifiers),
        "identifiers must be digit strings: clustering_accuracy casts the "
        "assignment array with astype(np.int64)",
    )
    check(
        label_utils.new_identifier() not in identifiers,
        "new_identifier collided with an already-minted identifier",
    )

    print("  500 identifiers, all distinct and int64-castable: OK")


def verify_persistent_state(app):
    """The clustering must accumulate, and a feature-space change must reproject."""
    print("\n[5/7] Persistent clustering state")

    design = app.get_default_design()
    app.reset_run_state()

    performances = []
    for timestep in range(4):
        metrics = app.exec_design(design, timestep)
        performances.append(metrics['performance'])

    check(
        app.state is not None and len(app.state) == 4,
        f"expected 4 timesteps of state, got "
        f"{0 if app.state is None else len(app.state)}",
    )
    print(f"  accuracy over 4 timesteps: {[round(p, 3) for p in performances]}")

    clusters_before = app.state[-1]['clusters']
    members_before = sorted(
        sorted(int(i) for i in c.vector_indices) for c in clusters_before
    )
    width_before = int(np.asarray(clusters_before[0].centroid).ravel().size)

    # Switch to a different feature extractor, which changes the feature
    # space the clusters live in.
    available = [
        v for v in THESIS_OPTIONS['feature_extraction']
        if v not in app.excluded_options.get('feature_extraction', [])
        and v != design['feature_extraction']
    ]
    check(available, "no second feature extractor available to switch to")

    changed = dict(design)
    changed['feature_extraction'] = available[0]
    changed['step'] = 2          # removal, with cluster_removal=0 -> a no-op
    changed['cluster_removal'] = 0

    app.exec_design(changed, 4)

    clusters_after = app.state[-1]['clusters']
    members_after = sorted(
        sorted(int(i) for i in c.vector_indices) for c in clusters_after
    )
    width_after = int(np.asarray(clusters_after[0].centroid).ravel().size)

    check(
        members_before == members_after,
        "a change of feature extractor lost the clustering's membership; "
        "reprojection should keep which instances are grouped and recompute "
        "only the coordinates",
    )
    check(
        width_after != width_before,
        f"the centroid width did not change across a feature-space change "
        f"({width_before} -> {width_after}), so the clusters were not "
        f"actually reprojected",
    )
    check(
        app.num_reprojections > 0,
        "the application reported no reprojections",
    )

    print(f"  centroid width {width_before} -> {width_after}, "
          f"membership preserved across {app.num_reprojections} reprojections")
    print("  persistent state: OK")


def verify_meta_features(app):
    """The meta-feature vector must be the same length at every timestep."""
    print("\n[6/7] Meta-features")

    design = app.get_default_design()
    app.reset_run_state()

    names = app.get_meta_feature_names()
    lengths = set()

    for timestep in range(4):
        features = app.extract_meta_features(timestep=timestep)
        lengths.add(len(features))
        check(
            list(features.keys()) == names,
            f"meta-feature keys at timestep {timestep} do not match the "
            f"declared schema",
        )
        check(
            all(np.isfinite(v) for v in features.values()),
            f"meta-features at timestep {timestep} contain a non-finite value: "
            f"{ {k: v for k, v in features.items() if not np.isfinite(v)} }",
        )
        app.exec_design(design, timestep)

    check(
        len(lengths) == 1,
        f"meta-feature vector length varied across timesteps: {lengths}",
    )

    print(f"  {len(names)} features, identical length at every timestep: OK")


def verify_approaches(instances, timesteps):
    """All four variants must run end to end and report a test performance."""
    print("\n[7/7] All four approaches")

    ga_params = {'population_size': 4}
    summary = {}

    for key, kwargs, runner in [
        ('onmar-accuracy', {'theta_t': None, 'theta_p': 0.65}, 'run_onmar'),
        ('onmar-design', {'theta_t': None, 'theta_p': 0.65}, 'run_onmar'),
        ('offmar-accuracy', {'theta_p': 0.65}, 'run_full_offmar'),
        ('offmar-design', {'theta_p': 0.65}, 'run_full_offmar'),
    ]:
        module = load_approach(key)
        approach_class = next(
            getattr(module, name) for name in dir(module)
            if name.startswith(('OnMAR', 'OffMAR'))
            and name not in ('OnMARConfig', 'OffMARConfig')
        )

        print(f"\n  --- {key} ---")
        start = time.time()
        approach = approach_class(
            application=make_app(instances),
            meta_learner_type='knn',
        # k=2, not 3. The OffMAR variants split the timestep budget into two
        # phases, so Phase 1 contributes only timesteps/2 entries to the
        # knowledge repository, and scikit-learn's k-NN raises outright when
        # k exceeds the number of samples it was fitted on
        # ("Expected n_neighbors <= n_samples_fit"). k=2 keeps this script
        # runnable at the small budgets it is meant for.
            meta_learner_params={'k': 2},
            design_algorithm_params=ga_params,
            **kwargs,
        )
        results = getattr(approach, runner)(dataset_name='mnist', timesteps=timesteps)

        test_performance = results.get('test_performance')
        check(
            test_performance is not None,
            f"{key} reported no test performance",
        )
        # Chance on MNIST's ten classes is 0.1; anything at or below that
        # means the run produced no usable clustering at all.
        check(
            test_performance > 0.1,
            f"{key} reported a test performance at or below chance: "
            f"{test_performance}",
        )
        summary[key] = (test_performance, time.time() - start)
        print(f"  {key}: test={test_performance:.4f} ({summary[key][1]:.0f}s)")

    print("\n  summary:")
    for key, (performance, elapsed) in summary.items():
        print(f"    {key:18s} test={performance:.4f}  {elapsed:.0f}s")
    print("  all four approaches: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--instances', type=int, default=48,
        help='Instances to cluster (small keeps this quick)'
    )
    parser.add_argument(
        '--timesteps', type=int, default=6,
        help='Timesteps per approach run'
    )
    parser.add_argument('--skip-approaches', action='store_true',
                        help='Skip the four end-to-end approach runs')
    parser.add_argument('--skip-options', action='store_true',
                        help='Skip the per-option sweep, which is the slowest check')
    args = parser.parse_args()

    print("=" * 78)
    print("CLUSTERING COMPOSITION PIPELINE VERIFICATION")
    print("=" * 78)

    try:
        app = make_app(args.instances)
        app.load_data()

        verify_design_space(app)
        if args.skip_options:
            print("\n[2/7] Every design option: skipped (--skip-options)")
        else:
            verify_every_option(app)
        verify_feature_extractors(app)
        verify_cluster_identifiers()
        verify_persistent_state(app)
        verify_meta_features(app)
        if args.skip_approaches:
            print("\n[7/7] Approaches: skipped (--skip-approaches)")
        else:
            verify_approaches(args.instances, args.timesteps)
    except CheckFailed as error:
        print(f"\n{'=' * 78}")
        print(f"VERIFICATION FAILED: {error}")
        print("=" * 78)
        return 1

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
