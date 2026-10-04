"""
OnMAR on CIFAR-10 (CNN configuration)
=====================================

Runs OnMAR accuracy-prediction over the CNN configuration application on
CIFAR-10 and saves the full result set as JSON.

CIFAR-10 is roughly four times the pixels of MNIST and has three channels, so
a timestep costs several times more. The defaults below therefore cap the
training batches per timestep rather than running a full epoch; pass
``--batches 0`` for full epochs when running on a GPU or a cluster.

Usage:
    python experiments/test/run_cifar10_onmar.py
    python experiments/test/run_cifar10_onmar.py --timesteps 80 --batches 0
    python experiments/test/run_cifar10_onmar.py --variant design --meta-learner rf
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import json
import time

import numpy as np

from metalearner import load_approach
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication


def json_safe(obj):
    """Convert numpy scalars and arrays so the results can be serialised."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=40,
                        help='Number of timesteps (thesis uses 80)')
    parser.add_argument('--batches', type=int, default=150,
                        help='Training batches per timestep; 0 means a full epoch')
    parser.add_argument('--variant', choices=['accuracy', 'design'], default='accuracy',
                        help='OnMAR variant to run')
    parser.add_argument('--meta-learner', choices=['knn', 'rf', 'xgboost'], default='knn',
                        help='Meta-learner (Table 8.2 favours kNN for OnMAR)')
    parser.add_argument('--theta-p', type=float, default=0.85)
    parser.add_argument('--population', type=int, default=8,
                        help='GA population size')
    parser.add_argument('--workers', type=int, default=2,
                        help='Data loading workers')
    parser.add_argument('--warm-start', dest='warm_start', action='store_true',
                        default=False,
                        help='Warm-start GA candidates from the live model. Off by '
                             'default: measured worse than cold probes on MNIST '
                             '(0.9795 vs 0.9843 test) and 38%% slower.')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        default=True,
                        help='Disable pretrained (ImageNet) initialisation of the '
                             'first design, for a from-scratch comparison')
    parser.add_argument('--donor', default='resnet18',
                        help='Pretrained donor architecture')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None,
                        help='Where to write results (default: the approach results dir)')
    args = parser.parse_args()

    key = f'onmar-{args.variant}'
    approach_class = (
        load_approach(key).OnMARAccuracyPrediction if args.variant == 'accuracy'
        else load_approach(key).OnMARDesignPrediction
    )

    print("=" * 78)
    print(f"OnMAR {args.variant}-prediction on CIFAR-10")
    print(f"  meta-learner : {args.meta_learner}")
    print(f"  timesteps    : {args.timesteps}")
    print(f"  batches/step : {'full epoch' if args.batches == 0 else args.batches}")
    print(f"  GA population: {args.population}")
    print(f"  warm start   : {args.warm_start}")
    print(f"  pretrained   : {args.pretrained} ({args.donor})")
    print("=" * 78)

    app = CNNConfigurationApplication(
        dataset_name='cifar-10',
        random_seed=args.seed,
        num_workers=args.workers,
        max_train_batches_per_timestep=None if args.batches == 0 else args.batches,
        warm_start_candidates=args.warm_start,
        pretrained_init=args.pretrained,
        pretrained_donor=args.donor,
    )

    onmar = approach_class(
        application=app,
        meta_learner_type=args.meta_learner,
        theta_t=None,                     # N/2, per Algorithm 9
        theta_p=args.theta_p,
        meta_learner_params={'k': 5} if args.meta_learner == 'knn' else {},
        design_algorithm_params={'population_size': args.population},
    )

    start = time.time()
    results = onmar.run_onmar(
        dataset_name='cifar-10',
        timesteps=args.timesteps,
        initial_design=None,
    )
    elapsed = time.time() - start

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent.parent
        / 'OnMAR' / f'{args.variant}-prediction' / 'results'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'cnn_cifar-10_{args.meta_learner}.json'

    payload = dict(results)
    payload['wall_clock_seconds'] = elapsed
    payload['configuration'] = vars(args)
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2, default=json_safe)

    history = results['performance_history']
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  best validation accuracy : {max(history):.4f}")
    print(f"  final validation accuracy: {history[-1]:.4f}")
    print(f"  test accuracy            : {results['test_performance']:.4f}")
    print(f"  wall clock               : {elapsed/60:.1f} min "
          f"({elapsed/args.timesteps:.1f}s per timestep)")
    print(f"  design algorithm         : {results['design_algorithm']}")
    print(f"  application              : {results['application_statistics']}")
    print(f"  trace                    : {[round(p, 4) for p in history]}")
    print(f"\n  results written to {output_path}")


if __name__ == "__main__":
    main()
