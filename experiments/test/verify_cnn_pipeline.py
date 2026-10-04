"""
CNN Pipeline Verification
=========================

Checks that the CNN configuration application and all four meta-learning
variants actually work, at a budget small enough to run on a laptop in a few
minutes. Run this after changing the design space, the GA, the application, or
any of the four approaches.

What it checks, and why each check is here:

  1. Design space    - designs round trip through the fixed-length encoding,
                       and arbitrary vectors still decode to buildable designs
                       (the meta-learner's output is unconstrained regression)
  2. Model building  - every randomly sampled design builds and runs
  3. Genetic algorithm - fitness improves over generations
  4. Persistent state - accuracy accumulates across timesteps, and a design
                       change transfers weights instead of starting over
  5. Meta-features   - the vector length is identical at every timestep
                       (ragged vectors cannot be fitted by a meta-learner)
  6. Approaches      - all four run end to end and report a test performance

Usage:
    python experiments/test/verify_cnn_pipeline.py
    python experiments/test/verify_cnn_pipeline.py --skip-approaches   # faster
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import copy
import time

import numpy as np
import torch

from genetic_algorithm import DesignSpace, GeneticAlgorithm
from metalearner import load_approach
from applications.configuration.cnn.cnn_application import (
    CNNConfigurationApplication, DESIGN_SPACE_SPEC, DEFAULT_DESIGN,
)
from applications.configuration.cnn.cnn_model import ConfigurableCNN


class CheckFailed(Exception):
    """A verification check failed."""


def check(condition, message):
    """Assert a condition with a readable failure."""
    if not condition:
        raise CheckFailed(message)


def verify_design_space():
    """Designs must survive the encode/decode round trip, and noise must decode."""
    print("\n[1/6] Design space")
    space = DesignSpace(DESIGN_SPACE_SPEC)
    rng = np.random.default_rng(0)
    print(f"  encoding length: {space.encoding_length}")

    for _ in range(100):
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
            len(decoded['conv_layers']) == len(design['conv_layers']),
            "round trip changed the number of convolutional layers",
        )
        check(
            abs(decoded['learning_rate'] - design['learning_rate']) < 1e-6,
            "round trip changed the learning rate",
        )

    # Unconstrained vectors (what a regression meta-learner emits) must decode.
    for _ in range(100):
        noise = rng.normal(0.5, 0.8, size=space.encoding_length).astype(np.float32)
        check(space.validate(space.decode(noise)), "noise did not decode to a valid design")

    # Genetic operators must stay in-space.
    for _ in range(100):
        a, b = space.sample(rng), space.sample(rng)
        check(space.validate(space.mutate(a, rng)), "mutation left the design space")
        check(space.validate(space.crossover(a, b, rng)), "crossover left the design space")

    print("  round trip, noise decode, mutation and crossover: OK")
    return space


def verify_model_building(space):
    """Every sampled design must build and run a forward pass."""
    print("\n[2/6] Model building")
    rng = np.random.default_rng(1)

    for shape, classes, label in [((1, 28, 28), 10, 'MNIST'), ((3, 32, 32), 100, 'CIFAR-100')]:
        largest_params = 0
        largest_macs = 0
        for _ in range(40):
            design = space.sample(rng)
            model = ConfigurableCNN(shape, classes, design)
            output = model(torch.zeros(2, *shape))
            check(output.shape == (2, classes), f"wrong output shape {output.shape}")
            largest_params = max(largest_params, model.get_num_parameters())
            largest_macs = max(largest_macs, model.estimate_macs())
        print(f"  {label}: 40 designs built; largest {largest_params:,} params, "
              f"{largest_macs/1e6:.0f}M MACs/image")


def verify_genetic_algorithm(space):
    """The GA must improve on a known objective."""
    print("\n[3/6] Genetic algorithm")

    def fitness(design, timestep):
        # Rewards wide conv layers, batch norm, ReLU-family activations,
        # three conv layers, and a learning rate near 0.01.
        filters = np.mean([c['filters'] / 512.0 for c in design['conv_layers']])
        norm = np.mean([c['batch_norm'] for c in design['conv_layers']])
        activation = np.mean(
            [1.0 if c['activation'] in (2, 3, 8) else 0.0 for c in design['conv_layers']]
        )
        depth = 1.0 - abs(len(design['conv_layers']) - 3) / 5.0
        lr = 1.0 - abs(np.log10(design['learning_rate']) + 2) / 3.0
        return float(filters + norm + activation + depth + lr) / 5.0

    ga = GeneticAlgorithm(
        space, fitness, population_size=8, random_seed=1,
        mutation_rate=0.9, gene_mutation_rate=0.12, structure_mutation_rate=0.2,
    )

    history = []
    for t in range(15):
        ga.step(t)
        history.append(ga.best_individual.fitness)

    print(f"  best fitness: {history[0]:.3f} -> {history[-1]:.3f} "
          f"over 15 generations, {ga.num_fitness_evaluations} evaluations")
    check(history[-1] > history[0], f"GA did not improve: {history}")
    print("  GA improves: OK")


def verify_persistent_state(batches):
    """Training state must accumulate across timesteps and survive design changes."""
    print("\n[4/6] Persistent training state")
    app = CNNConfigurationApplication(
        'mnist', random_seed=42, num_workers=0,
        max_train_batches_per_timestep=batches,
        candidate_train_batches=12, candidate_val_batches=4,
    )
    app.load_data()

    accuracies = []
    lengths = set()
    for t in range(4):
        lengths.add(len(app.extract_meta_features(timestep=t)))
        metrics = app.exec_design(DEFAULT_DESIGN, t)
        accuracies.append(metrics['performance'])
        print(f"  t={t}: val_acc={metrics['performance']:.4f} "
              f"loss={metrics['train_loss']:.4f} kept={metrics['weights_transferred']:.2f}")

    check(len(lengths) == 1, f"meta-feature vector length varied across timesteps: {lengths}")
    print(f"\n[5/6] Meta-feature schema: {lengths.pop()} features at every timestep: OK")

    check(
        accuracies[-1] > accuracies[0],
        f"no learning across timesteps - state is not persisting: {accuracies}",
    )
    check(app.num_design_changes == 0, "design changed when it should not have")

    print("\n  design change mid-run:")
    changed = copy.deepcopy(DEFAULT_DESIGN)
    changed['conv_layers'][1]['filters'] = 128
    metrics = app.exec_design(changed, 4)
    print(f"    val_acc={metrics['performance']:.4f} "
          f"values_kept={metrics['weights_transferred']:.4f} "
          f"tensors_kept={metrics['tensors_transferred']:.2f}")
    check(
        metrics['tensors_transferred'] > 0.0,
        "no weights transferred across a design change",
    )
    check(
        metrics['performance'] > 0.3,
        f"collapsed to chance after a design change: {metrics['performance']}",
    )

    print("  a learning-rate-only change must keep everything:")
    lr_only = copy.deepcopy(changed)
    lr_only['learning_rate'] = 0.0005
    metrics = app.exec_design(lr_only, 5)
    print(f"    values_kept={metrics['weights_transferred']:.2f}")
    check(metrics['weights_transferred'] == 1.0, "a learning-rate change discarded weights")

    results = app.evaluate()
    print(f"  test evaluation: {results['test_performance']:.4f}")
    check('test_performance' in results, "evaluate() must report 'test_performance'")
    print("  persistent state: OK")


def verify_approaches(batches, timesteps):
    """All four variants must run end to end and report a test performance."""
    print("\n[6/6] All four approaches")

    def make_app():
        return CNNConfigurationApplication(
            'mnist', random_seed=42, num_workers=0,
            max_train_batches_per_timestep=batches,
            candidate_train_batches=10, candidate_val_batches=4,
        )

    ga_params = {'population_size': 4}
    summary = {}

    for key, kwargs, runner in [
        ('onmar-accuracy', {'theta_t': None, 'theta_p': 0.85}, 'run_onmar'),
        ('onmar-design', {'theta_t': None, 'theta_p': 0.85}, 'run_onmar'),
        ('offmar-accuracy', {'theta_p': 0.85}, 'run_full_offmar'),
        ('offmar-design', {'theta_p': 0.85}, 'run_full_offmar'),
    ]:
        module = load_approach(key)
        approach_class = next(
            getattr(module, name) for name in dir(module)
            if name.startswith(('OnMAR', 'OffMAR')) and name not in ('OnMARConfig', 'OffMARConfig')
        )

        print(f"\n  --- {key} ---")
        start = time.time()
        approach = approach_class(
            application=make_app(),
            meta_learner_type='knn',
            meta_learner_params={'k': 3},
            design_algorithm_params=ga_params,
            **kwargs,
        )
        results = getattr(approach, runner)(dataset_name='mnist', timesteps=timesteps)

        test_performance = results.get('test_performance')
        check(
            test_performance is not None and test_performance > 0.2,
            f"{key} reported no usable test performance: {test_performance}",
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
        '--batches', type=int, default=20,
        help='Training batches per timestep (small keeps this quick)'
    )
    parser.add_argument(
        '--timesteps', type=int, default=6,
        help='Timesteps per approach run'
    )
    parser.add_argument('--skip-approaches', action='store_true',
                        help='Skip the four end-to-end approach runs')
    args = parser.parse_args()

    print("=" * 78)
    print("CNN PIPELINE VERIFICATION")
    print("=" * 78)

    try:
        space = verify_design_space()
        verify_model_building(space)
        verify_genetic_algorithm(space)
        verify_persistent_state(args.batches)
        if args.skip_approaches:
            print("\n[6/6] Approaches: skipped (--skip-approaches)")
        else:
            verify_approaches(args.batches, args.timesteps)
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
