"""
Setup Verification Script
=========================

Quick script to verify all dependencies and imports are working correctly
before running the full experiment suite.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# The four approach directories contain a hyphen, so they cannot be imported
# by name and are resolved by path instead.
from metalearner import load_approach, load_approach_config

def verify_imports():
    """Verify all required imports work."""
    print("Verifying imports...")
    errors = []

    try:
        load_approach('onmar-accuracy').OnMARAccuracyPrediction
        print("  ✓ OnMAR Accuracy Prediction")
    except Exception as e:
        errors.append(f"OnMAR Accuracy Prediction: {e}")
        print(f"  ✗ OnMAR Accuracy Prediction: {e}")

    try:
        load_approach('onmar-design').OnMARDesignPrediction
        print("  ✓ OnMAR Design Prediction")
    except Exception as e:
        errors.append(f"OnMAR Design Prediction: {e}")
        print(f"  ✗ OnMAR Design Prediction: {e}")

    try:
        load_approach('offmar-accuracy').OffMARAccuracyPrediction
        print("  ✓ OffMAR Accuracy Prediction")
    except Exception as e:
        errors.append(f"OffMAR Accuracy Prediction: {e}")
        print(f"  ✗ OffMAR Accuracy Prediction: {e}")

    try:
        load_approach('offmar-design').OffMARDesignPrediction
        print("  ✓ OffMAR Design Prediction")
    except Exception as e:
        errors.append(f"OffMAR Design Prediction: {e}")
        print(f"  ✗ OffMAR Design Prediction: {e}")

    try:
        from sota.autosklearn_wrapper import AutoSklearnWrapper
        print("  ✓ AutoSklearn Wrapper")
    except Exception as e:
        errors.append(f"AutoSklearn Wrapper: {e}")
        print(f"  ✗ AutoSklearn Wrapper: {e}")

    try:
        from applications.configuration.cnn.cnn_application import CNNConfigurationApplication
        print("  ✓ CNN Application")
    except Exception as e:
        errors.append(f"CNN Application: {e}")
        print(f"  ✗ CNN Application: {e}")

    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import scipy
        print("  ✓ Analysis libraries (numpy, pandas, matplotlib, scipy)")
    except Exception as e:
        errors.append(f"Analysis libraries: {e}")
        print(f"  ✗ Analysis libraries: {e}")

    return errors


def verify_configs():
    """Verify configuration files are accessible."""
    print("\nVerifying configurations...")
    errors = []

    try:
        CNN_CONFIG = load_approach_config('onmar-accuracy').CNN_CONFIG
        print(f"  ✓ OnMAR Accuracy Prediction configs ({len(CNN_CONFIG)} meta-learners)")
    except Exception as e:
        errors.append(f"OnMAR Accuracy configs: {e}")
        print(f"  ✗ OnMAR Accuracy configs: {e}")

    try:
        CNN_CONFIG = load_approach_config('onmar-design').CNN_CONFIG
        print(f"  ✓ OnMAR Design Prediction configs ({len(CNN_CONFIG)} meta-learners)")
    except Exception as e:
        errors.append(f"OnMAR Design configs: {e}")
        print(f"  ✗ OnMAR Design configs: {e}")

    try:
        CNN_CONFIG = load_approach_config('offmar-accuracy').CNN_CONFIG
        print(f"  ✓ OffMAR Accuracy Prediction configs ({len(CNN_CONFIG)} meta-learners)")
    except Exception as e:
        errors.append(f"OffMAR Accuracy configs: {e}")
        print(f"  ✗ OffMAR Accuracy configs: {e}")

    try:
        CNN_CONFIG = load_approach_config('offmar-design').CNN_CONFIG
        print(f"  ✓ OffMAR Design Prediction configs ({len(CNN_CONFIG)} meta-learners)")
    except Exception as e:
        errors.append(f"OffMAR Design configs: {e}")
        print(f"  ✗ OffMAR Design configs: {e}")

    return errors


def verify_datasets():
    """Verify datasets can be loaded."""
    print("\nVerifying dataset access...")
    errors = []

    try:
        from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

        datasets = ['mnist', 'fashion-mnist', 'cifar-10', 'cifar-100']

        for dataset in datasets:
            try:
                app = CNNConfigurationApplication(dataset_name=dataset, random_seed=42)
                app.load_data()
                print(f"  ✓ {dataset.upper()} dataset loaded")
            except Exception as e:
                errors.append(f"{dataset} dataset: {e}")
                print(f"  ✗ {dataset.upper()} dataset: {e}")

    except Exception as e:
        errors.append(f"Dataset loading: {e}")
        print(f"  ✗ Dataset loading: {e}")

    return errors


def test_quick_run():
    """Test a very quick run of OnMAR."""
    print("\nTesting quick OnMAR run (this may take 1-2 minutes)...")

    try:
        OnMARAccuracyPrediction = load_approach('onmar-accuracy').OnMARAccuracyPrediction
        CNN_CONFIG = load_approach_config('onmar-accuracy').CNN_CONFIG
        from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

        # Initialize with MNIST
        app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)
        config = CNN_CONFIG['knn']

        onmar = OnMARAccuracyPrediction(
            application=app,
            meta_learner_type=config.meta_learner_type,
            theta_t=config.theta_t,
            theta_p=config.theta_p,
            meta_learner_params=config.meta_learner_params
        )

        # Run for just 5 timesteps
        results = onmar.run_onmar(
            dataset_name='mnist',
            timesteps=5,
            initial_design=None
        )

        print(f"  ✓ Quick test successful!")
        print(f"    - Best performance: {results['best_performance']:.4f}")
        print(f"    - Runtime: {results['total_time']:.2f}s")

        return []

    except Exception as e:
        print(f"  ✗ Quick test failed: {e}")
        return [f"Quick test: {e}"]


def main():
    """Run all verification checks."""
    print("="*80)
    print("SETUP VERIFICATION")
    print("="*80)

    all_errors = []

    # Check imports
    errors = verify_imports()
    all_errors.extend(errors)

    # Check configs
    errors = verify_configs()
    all_errors.extend(errors)

    # Check datasets (this may download data)
    print("\nNote: Dataset verification may download data if not already present...")
    response = input("Proceed with dataset verification? (y/n): ")
    if response.lower() == 'y':
        errors = verify_datasets()
        all_errors.extend(errors)

    # Quick functionality test
    print("\nNote: Quick test will run 5 epochs of OnMAR (1-2 minutes)...")
    response = input("Proceed with quick functionality test? (y/n): ")
    if response.lower() == 'y':
        errors = test_quick_run()
        all_errors.extend(errors)

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    if not all_errors:
        print("✓ All checks passed! Setup is ready.")
        print("\nYou can now run:")
        print("  python runner.py --quick-test")
        print("or")
        print("  python runner.py --all-datasets --timesteps 50 --runs 30")
    else:
        print(f"✗ {len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"  - {error}")
        print("\nPlease fix these errors before running experiments.")

    print("="*80)


if __name__ == "__main__":
    main()
