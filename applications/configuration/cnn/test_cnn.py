"""
Test script for CNN Configuration Application

Checks the pieces the meta-learning approaches rely on: the design space, the
fixed meta-feature schema, timestep execution against persistent training
state, candidate scoring, and test-set evaluation.

Run with a small --batches value for a quick check:

    python applications/configuration/cnn/test_cnn.py --batches 20 --timesteps 4
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import copy

from applications.configuration.cnn.cnn_application import (
    CNNConfigurationApplication, DEFAULT_DESIGN,
)


def _describe_layer(layer, size_key):
    """Render one layer as e.g. ``32f/a3/p2/bn/do0.25``."""
    parts = [f"{layer[size_key]}{'f' if size_key == 'filters' else 'n'}",
             f"a{layer['activation']}"]
    if 'max_pool' in layer:
        parts.append(f"p{layer['max_pool']}")
    if layer['batch_norm']:
        parts.append("bn")
    if layer['dropout'] > 0:
        parts.append(f"do{layer['dropout']:.2f}")
    return "/".join(parts)


def describe_design(design):
    """Render a design compactly enough to print."""
    conv = ", ".join(_describe_layer(c, 'filters') for c in design['conv_layers'])
    dense = ", ".join(_describe_layer(d, 'nodes') for d in design['dense_layers'])
    return (
        f"conv[{conv}] dense[{dense}] "
        f"opt={design['optimizer']} lr={design['learning_rate']:.5f}"
    )


def test_cnn_application(dataset='mnist', timesteps=4, batches=25):
    """Exercise the CNN application end to end."""
    print("=" * 80)
    print("Testing CNN Configuration Application")
    print("=" * 80)

    print(f"\n1. Initializing with {dataset}...")
    app = CNNConfigurationApplication(
        dataset_name=dataset,
        random_seed=42,
        num_workers=0,
        max_train_batches_per_timestep=batches,
    )

    print("\n2. Loading data...")
    app.load_data()

    print("\n3. Design space:")
    for name, option in app.get_design_space().items():
        if option['type'] == 'block_list':
            print(f"   {name}: {option['min_blocks']}-{option['max_blocks']} blocks of "
                  f"{list(option['genes'].keys())}")
        elif option['type'] == 'continuous':
            print(f"   {name}: {option['min']} to {option['max']}"
                  f"{' (log scale)' if option.get('log') else ''}")
        else:
            print(f"   {name}: {option['options']}")
    print(f"   -> encodes to a fixed-length vector of {app.get_design_encoding_length()}")

    print("\n4. Meta-features:")
    print(f"   fixed schema of {len(app.get_meta_feature_names())} features")
    for key, value in app.get_meta_features().items():
        print(f"   {key}: {value}")

    print("\n5. Random designs from the space:")
    for i in range(3):
        design = app.sample_design()
        assert app.validate_design(design)
        print(f"   {describe_design(design)}")

    print(f"\n6. Running {timesteps} timesteps with the default design...")
    print(f"   {describe_design(DEFAULT_DESIGN)}")
    lengths = set()
    for t in range(timesteps):
        lengths.add(len(app.extract_meta_features(timestep=t)))
        metrics = app.exec_design(DEFAULT_DESIGN, t)
        print(f"   t={t}: val_acc={metrics['performance']:.4f} "
              f"loss={metrics['train_loss']:.4f} "
              f"transferred={metrics['weights_transferred']:.2f}")
    assert len(lengths) == 1, f"meta-feature vector length varied: {lengths}"
    print(f"   meta-feature vector length was {lengths.pop()} at every timestep")

    print("\n7. Changing the design mid-run (training state must carry over):")
    changed = copy.deepcopy(DEFAULT_DESIGN)
    changed['conv_layers'][0]['filters'] = 64
    metrics = app.exec_design(changed, timesteps)
    print(f"   val_acc={metrics['performance']:.4f} "
          f"values_kept={metrics['weights_transferred']:.3f} "
          f"tensors_kept={metrics['tensors_transferred']:.2f}")

    print("\n8. Scoring candidate designs (the GA's fitness function):")
    for label, design in [('default', DEFAULT_DESIGN), ('random', app.sample_design())]:
        print(f"   {label}: {app.evaluate_candidate(design, timesteps):.4f}")

    print("\n9. Test-set evaluation:")
    for key, value in app.evaluate().items():
        print(f"   {key}: {value:.4f}")

    print("\n10. Application metadata:")
    print(f"   type: {app.get_application_type()}")
    print(f"   dynamic designs: {app.supports_dynamic_designs()}")
    print(f"   built-in design algorithm: {app.has_builtin_design_algorithm()}")
    print(f"   default timesteps: {app.get_num_timesteps()}")
    print(f"   run statistics: {app.get_run_statistics()}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--timesteps', type=int, default=4)
    parser.add_argument(
        '--batches', type=int, default=25,
        help='Training batches per timestep (fewer is faster; omit for a full epoch)'
    )
    args = parser.parse_args()

    test_cnn_application(args.dataset, args.timesteps, args.batches)
