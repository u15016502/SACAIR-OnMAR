"""
Comprehensive Test Runner for All Applications
==============================================

This script runs OnMAR (accuracy-prediction), OnMAR (design-prediction),
OffMAR (accuracy-prediction), OffMAR (design-prediction), and AutoSklearn
on all datasets for the specified application.

Supports three applications:
- CNN (Configuration): mnist, fashion-mnist, cifar-10, cifar-100
- Segmentation (Composition): bsd500, covid, pascal
- FuzzyART (Generation): chatgpt, enron, imdb

Results are logged in the same format as reported in the thesis (Chapter 8).

Usage:
    # CNN application
    python runner.py --application cnn --datasets mnist cifar-10 --timesteps 50 --runs 30
    python runner.py --application cnn --all-datasets --timesteps 50 --runs 30

    # Segmentation application
    python runner.py --application segmentation --datasets bsd500 --timesteps 60 --runs 30

    # FuzzyART application
    python runner.py --application fuzzyart --datasets chatgpt --timesteps 60 --runs 30

    # Quick test (any application)
    python runner.py --application segmentation --quick-test
"""

import sys
from pathlib import Path
import argparse
import json
import csv
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import using importlib for directories with hyphens
def import_from_path(module_name, file_path):
    """Import module from file path (handles directories with hyphens)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import OnMAR accuracy-prediction
onmar_acc_module = import_from_path(
    'onmar_accuracy_prediction',
    PROJECT_ROOT / 'OnMAR' / 'accuracy-prediction' / 'onmar.py'
)
onmar_acc_config = import_from_path(
    'onmar_accuracy_prediction_config',
    PROJECT_ROOT / 'OnMAR' / 'accuracy-prediction' / 'config.py'
)
OnMARAccuracyPrediction = onmar_acc_module.OnMARAccuracyPrediction
CNN_CONFIG_ONMAR_ACC = onmar_acc_config.CNN_CONFIG

# Import OnMAR design-prediction
onmar_des_module = import_from_path(
    'onmar_design_prediction',
    PROJECT_ROOT / 'OnMAR' / 'design-prediction' / 'onmar.py'
)
onmar_des_config = import_from_path(
    'onmar_design_prediction_config',
    PROJECT_ROOT / 'OnMAR' / 'design-prediction' / 'config.py'
)
OnMARDesignPrediction = onmar_des_module.OnMARDesignPrediction
CNN_CONFIG_ONMAR_DES = onmar_des_config.CNN_CONFIG

# Import OffMAR accuracy-prediction
offmar_acc_module = import_from_path(
    'offmar_accuracy_prediction',
    PROJECT_ROOT / 'OffMAR' / 'accuracy-prediction' / 'offmar.py'
)
offmar_acc_config = import_from_path(
    'offmar_accuracy_prediction_config',
    PROJECT_ROOT / 'OffMAR' / 'accuracy-prediction' / 'config.py'
)
OffMARAccuracyPrediction = offmar_acc_module.OffMARAccuracyPrediction
CNN_CONFIG_OFFMAR_ACC = offmar_acc_config.CNN_CONFIG

# Import OffMAR design-prediction
offmar_des_module = import_from_path(
    'offmar_design_prediction',
    PROJECT_ROOT / 'OffMAR' / 'design-prediction' / 'offmar.py'
)
offmar_des_config = import_from_path(
    'offmar_design_prediction_config',
    PROJECT_ROOT / 'OffMAR' / 'design-prediction' / 'config.py'
)
OffMARDesignPrediction = offmar_des_module.OffMARDesignPrediction
CNN_CONFIG_OFFMAR_DES = offmar_des_config.CNN_CONFIG

from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# The AutoSklearn baseline is optional. auto-sklearn pins an old scikit-learn
# and cannot share an environment with the rest of the project, so importing it
# unconditionally made the whole runner unusable - including for CNN runs that
# do not touch the baseline at all.
try:
    from sota.autosklearn_wrapper import AutoSklearnWrapper
    AUTOSKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AutoSklearn baseline not available: {e}")
    AutoSklearnWrapper = None
    AUTOSKLEARN_AVAILABLE = False

# Optional imports - applications with many dependencies
try:
    from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication
    SEGMENTATION_AVAILABLE = True
    SEGMENTATION_CONFIG_ONMAR_ACC = onmar_acc_config.SEGMENTATION_CONFIG
    SEGMENTATION_CONFIG_ONMAR_DES = onmar_des_config.SEGMENTATION_CONFIG
    SEGMENTATION_CONFIG_OFFMAR_ACC = offmar_acc_config.SEGMENTATION_CONFIG
    SEGMENTATION_CONFIG_OFFMAR_DES = offmar_des_config.SEGMENTATION_CONFIG
except ImportError as e:
    print(f"Warning: Segmentation application not available: {e}")
    SegmentationCompositionApplication = None
    SEGMENTATION_AVAILABLE = False
    SEGMENTATION_CONFIG_ONMAR_ACC = None
    SEGMENTATION_CONFIG_ONMAR_DES = None
    SEGMENTATION_CONFIG_OFFMAR_ACC = None
    SEGMENTATION_CONFIG_OFFMAR_DES = None

try:
    from applications.generation.fuzzyart.fuzzyart_application import FuzzyARTGenerationApplication
    FUZZYART_AVAILABLE = True
    FUZZYART_CONFIG_ONMAR_ACC = onmar_acc_config.FUZZYART_CONFIG
    FUZZYART_CONFIG_ONMAR_DES = onmar_des_config.FUZZYART_CONFIG
    FUZZYART_CONFIG_OFFMAR_ACC = offmar_acc_config.FUZZYART_CONFIG
    FUZZYART_CONFIG_OFFMAR_DES = offmar_des_config.FUZZYART_CONFIG
except ImportError as e:
    print(f"Warning: FuzzyART application not available: {e}")
    FuzzyARTGenerationApplication = None
    FUZZYART_AVAILABLE = False
    FUZZYART_CONFIG_ONMAR_ACC = None
    FUZZYART_CONFIG_ONMAR_DES = None
    FUZZYART_CONFIG_OFFMAR_ACC = None
    FUZZYART_CONFIG_OFFMAR_DES = None


class ExperimentRunner:
    """Main experiment runner for all approaches and datasets."""

    # Application configurations (datasets mapped to application classes)
    # Build dynamically based on what's available
    APPLICATION_INFO = {}

    # CNN is always available
    APPLICATION_INFO['cnn'] = {
        'class': CNNConfigurationApplication,
        'datasets': ['mnist', 'fashion-mnist', 'cifar-10', 'cifar-100'],
        'configs': {
            'onmar_acc': CNN_CONFIG_ONMAR_ACC,
            'onmar_des': CNN_CONFIG_ONMAR_DES,
            'offmar_acc': CNN_CONFIG_OFFMAR_ACC,
            'offmar_des': CNN_CONFIG_OFFMAR_DES
        }
    }

    # Add segmentation if available
    if SEGMENTATION_AVAILABLE:
        APPLICATION_INFO['segmentation'] = {
            'class': SegmentationCompositionApplication,
            'datasets': ['bsd500', 'covid', 'pascal'],
            'configs': {
                'onmar_acc': SEGMENTATION_CONFIG_ONMAR_ACC,
                'onmar_des': SEGMENTATION_CONFIG_ONMAR_DES,
                'offmar_acc': SEGMENTATION_CONFIG_OFFMAR_ACC,
                'offmar_des': SEGMENTATION_CONFIG_OFFMAR_DES
            }
        }

    # Add fuzzyart if available
    if FUZZYART_AVAILABLE:
        APPLICATION_INFO['fuzzyart'] = {
            'class': FuzzyARTGenerationApplication,
            'datasets': ['chatgpt', 'enron', 'imdb'],
            'configs': {
                'onmar_acc': FUZZYART_CONFIG_ONMAR_ACC,
                'onmar_des': FUZZYART_CONFIG_ONMAR_DES,
                'offmar_acc': FUZZYART_CONFIG_OFFMAR_ACC,
                'offmar_des': FUZZYART_CONFIG_OFFMAR_DES
            }
        }

    # All meta-learners from thesis Chapter 8
    META_LEARNERS = ['knn', 'rf', 'xgboost']

    def __init__(self, output_dir: str = "experiments/test/results", application: str = 'cnn'):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save results
            application: Application type ('cnn', 'segmentation', 'fuzzyart')
        """
        if application not in self.APPLICATION_INFO:
            raise ValueError(f"Unknown application: {application}. Choose from {list(self.APPLICATION_INFO.keys())}")

        self.application_name = application
        self.application_class = self.APPLICATION_INFO[application]['class']
        self.application_datasets = self.APPLICATION_INFO[application]['datasets']
        self.application_configs = self.APPLICATION_INFO[application]['configs']

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this experimental run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Results storage
        self.results = defaultdict(lambda: defaultdict(list))

    def run_onmar_accuracy_prediction(
        self,
        dataset_name: str,
        meta_learner: str,
        timesteps: int,
        run_number: int
    ) -> Dict[str, Any]:
        """
        Run OnMAR with accuracy prediction.

        Args:
            dataset_name: Dataset to use
            meta_learner: Meta-learner type ('knn', 'rf', 'xgboost')
            timesteps: Number of training epochs
            run_number: Current run number (for seeding)

        Returns:
            Results dictionary
        """
        print(f"\n  [Run {run_number}] OnMAR-AccuracyPrediction with {meta_learner.upper()}...")

        # Initialize application with unique seed per run
        app = self.application_class(
            dataset_name=dataset_name,
            random_seed=42 + run_number
        )

        # Get config
        config = self.application_configs['onmar_acc'][meta_learner]

        # Initialize OnMAR
        onmar = OnMARAccuracyPrediction(
            application=app,
            meta_learner_type=config.meta_learner_type,
            theta_t=config.theta_t,
            theta_p=config.theta_p,
            meta_learner_params=config.meta_learner_params
        )

        # Run
        results = onmar.run_onmar(
            dataset_name=dataset_name,
            timesteps=timesteps,
            initial_design=None
        )

        # Add identifying metadata
        results['approach'] = 'OnMAR-AccuracyPrediction'
        results['meta_learner'] = meta_learner
        results['dataset'] = dataset_name
        results['run_number'] = run_number

        return results

    def run_onmar_design_prediction(
        self,
        dataset_name: str,
        meta_learner: str,
        timesteps: int,
        run_number: int
    ) -> Dict[str, Any]:
        """
        Run OnMAR with design prediction.

        Args:
            dataset_name: Dataset to use
            meta_learner: Meta-learner type
            timesteps: Number of training epochs
            run_number: Current run number

        Returns:
            Results dictionary
        """
        print(f"\n  [Run {run_number}] OnMAR-DesignPrediction with {meta_learner.upper()}...")

        app = self.application_class(
            dataset_name=dataset_name,
            random_seed=42 + run_number
        )

        config = self.application_configs['onmar_des'][meta_learner]

        onmar = OnMARDesignPrediction(
            application=app,
            meta_learner_type=config.meta_learner_type,
            theta_t=config.theta_t,
            theta_p=config.theta_p,
            meta_learner_params=config.meta_learner_params
        )

        results = onmar.run_onmar(
            dataset_name=dataset_name,
            timesteps=timesteps,
            initial_design=None
        )

        results['approach'] = 'OnMAR-DesignPrediction'
        results['meta_learner'] = meta_learner
        results['dataset'] = dataset_name
        results['run_number'] = run_number

        return results

    def run_offmar_accuracy_prediction(
        self,
        dataset_name: str,
        meta_learner: str,
        timesteps: int,
        run_number: int
    ) -> Dict[str, Any]:
        """
        Run OffMAR with accuracy prediction.

        Args:
            dataset_name: Dataset to use
            meta_learner: Meta-learner type
            timesteps: Number of training epochs
            run_number: Current run number

        Returns:
            Results dictionary
        """
        print(f"\n  [Run {run_number}] OffMAR-AccuracyPrediction with {meta_learner.upper()}...")

        app = self.application_class(
            dataset_name=dataset_name,
            random_seed=42 + run_number
        )

        config = self.application_configs['offmar_acc'][meta_learner]

        offmar = OffMARAccuracyPrediction(
            application=app,
            meta_learner_type=config.meta_learner_type,
            theta_p=config.theta_p,
            meta_learner_params=config.meta_learner_params
        )

        results = offmar.run_full_offmar(
            dataset_name=dataset_name,
            timesteps=timesteps
        )

        results['approach'] = 'OffMAR-AccuracyPrediction'
        results['meta_learner'] = meta_learner
        results['dataset'] = dataset_name
        results['run_number'] = run_number

        return results

    def run_offmar_design_prediction(
        self,
        dataset_name: str,
        meta_learner: str,
        timesteps: int,
        run_number: int
    ) -> Dict[str, Any]:
        """
        Run OffMAR with design prediction.

        Args:
            dataset_name: Dataset to use
            meta_learner: Meta-learner type
            timesteps: Number of training epochs
            run_number: Current run number

        Returns:
            Results dictionary
        """
        print(f"\n  [Run {run_number}] OffMAR-DesignPrediction with {meta_learner.upper()}...")

        app = self.application_class(
            dataset_name=dataset_name,
            random_seed=42 + run_number
        )

        config = self.application_configs['offmar_des'][meta_learner]

        offmar = OffMARDesignPrediction(
            application=app,
            meta_learner_type=config.meta_learner_type,
            theta_p=config.theta_p,
            meta_learner_params=config.meta_learner_params
        )

        results = offmar.run_full_offmar(
            dataset_name=dataset_name,
            timesteps=timesteps
        )

        results['approach'] = 'OffMAR-DesignPrediction'
        results['meta_learner'] = meta_learner
        results['dataset'] = dataset_name
        results['run_number'] = run_number

        return results

    def run_autosklearn(
        self,
        dataset_name: str,
        time_budget: int,
        run_number: int
    ) -> Dict[str, Any]:
        """
        Run AutoSklearn.

        Args:
            dataset_name: Dataset to use
            time_budget: Total time budget in seconds
            run_number: Current run number

        Returns:
            Results dictionary
        """
        print(f"\n  [Run {run_number}] AutoSklearn...")

        if not AUTOSKLEARN_AVAILABLE:
            raise RuntimeError(
                "The AutoSklearn baseline is not installed in this environment. "
                "auto-sklearn pins scikit-learn<0.25 and needs its own Python "
                "3.10 environment (see requirements.txt); run the other "
                "techniques here and the baseline separately."
            )

        app = self.application_class(
            dataset_name=dataset_name,
            random_seed=42 + run_number
        )

        autosklearn = AutoSklearnWrapper(
            application=app,
            time_budget=time_budget,
            per_run_time_limit=300,
            n_jobs=1,
            random_seed=42 + run_number
        )

        results = autosklearn.run_optimization(
            dataset_name=dataset_name,
            max_evaluations=50
        )

        results['approach'] = 'AutoSklearn'
        results['meta_learner'] = 'N/A'
        results['dataset'] = dataset_name
        results['run_number'] = run_number

        return results

    def run_full_experiment(
        self,
        datasets: List[str],
        timesteps: int = 50,
        num_runs: int = 30,
        time_budget_autosklearn: int = 3600,
        techniques: List[str] = None,
        meta_learners: List[str] = None
    ):
        """
        Run full experimental suite across all approaches and datasets.

        Args:
            datasets: List of datasets to test on
            timesteps: Number of timesteps for MAR approaches
            num_runs: Number of independent runs per configuration
            time_budget_autosklearn: Time budget for AutoSklearn (seconds)
            techniques: List of techniques to run (None = all)
            meta_learners: List of meta-learners to run (None = all)
        """
        # Set defaults if not specified
        if techniques is None:
            techniques = ['onmar-accuracy', 'onmar-design', 'offmar-accuracy', 'offmar-design', 'autosklearn']
        if meta_learners is None:
            meta_learners = self.META_LEARNERS

        print("="*80)
        print(f"COMPREHENSIVE {self.application_name.upper()} EXPERIMENT RUNNER")
        print("="*80)
        print(f"Application: {self.application_name}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Techniques: {', '.join(techniques)}")
        print(f"Meta-learners: {', '.join(meta_learners)}")
        print(f"Timesteps (MAR): {timesteps}")
        print(f"Time budget (AutoSklearn): {time_budget_autosklearn}s")
        print(f"Number of runs per config: {num_runs}")
        print(f"Results will be saved to: {self.output_dir}")
        print("="*80)

        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*80}")

            # Run all meta-learners for each approach
            for meta_learner in meta_learners:
                print(f"\n--- Meta-Learner: {meta_learner.upper()} ---")

                for run in range(1, num_runs + 1):
                    # OnMAR Accuracy Prediction
                    if 'onmar-accuracy' in techniques:
                        try:
                            result = self.run_onmar_accuracy_prediction(
                                dataset, meta_learner, timesteps, run
                            )
                            self.results[dataset][f'OnMAR-Acc-{meta_learner}'].append(result)
                            self._save_intermediate_result(result)
                        except Exception as e:
                            print(f"    ERROR in OnMAR-Acc-{meta_learner}: {e}")

                    # OnMAR Design Prediction
                    if 'onmar-design' in techniques:
                        try:
                            result = self.run_onmar_design_prediction(
                                dataset, meta_learner, timesteps, run
                            )
                            self.results[dataset][f'OnMAR-Des-{meta_learner}'].append(result)
                            self._save_intermediate_result(result)
                        except Exception as e:
                            print(f"    ERROR in OnMAR-Des-{meta_learner}: {e}")

                    # OffMAR Accuracy Prediction
                    if 'offmar-accuracy' in techniques:
                        try:
                            result = self.run_offmar_accuracy_prediction(
                                dataset, meta_learner, timesteps, run
                            )
                            self.results[dataset][f'OffMAR-Acc-{meta_learner}'].append(result)
                            self._save_intermediate_result(result)
                        except Exception as e:
                            print(f"    ERROR in OffMAR-Acc-{meta_learner}: {e}")

                    # OffMAR Design Prediction
                    if 'offmar-design' in techniques:
                        try:
                            result = self.run_offmar_design_prediction(
                                dataset, meta_learner, timesteps, run
                            )
                            self.results[dataset][f'OffMAR-Des-{meta_learner}'].append(result)
                            self._save_intermediate_result(result)
                        except Exception as e:
                            print(f"    ERROR in OffMAR-Des-{meta_learner}: {e}")

            # Run AutoSklearn (no meta-learner variants)
            if 'autosklearn' in techniques:
                print(f"\n--- AutoSklearn ---")
                for run in range(1, num_runs + 1):
                    try:
                        result = self.run_autosklearn(
                            dataset, time_budget_autosklearn, run
                        )
                        self.results[dataset]['AutoSklearn'].append(result)
                        self._save_intermediate_result(result)
                    except Exception as e:
                        print(f"    ERROR in AutoSklearn: {e}")

        # Generate final reports
        self._generate_summary_report()
        self._generate_thesis_format_tables()
        self._save_all_results()

        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

    def _save_intermediate_result(self, result: Dict[str, Any]):
        """Save individual result immediately to prevent data loss."""
        filename = (f"{result['dataset']}_{result['approach']}_"
                   f"{result['meta_learner']}_run{result['run_number']}.json")
        filepath = self.output_dir / "individual_runs" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def _generate_summary_report(self):
        """Generate summary statistics for all experiments."""
        summary_file = self.output_dir / f"summary_{self.timestamp}.csv"

        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header matching thesis Table 8.2 format
            writer.writerow([
                'Dataset', 'Approach', 'Meta-Learner',
                'Mean Best Performance', 'Std Best Performance',
                'Mean Final Performance', 'Std Final Performance',
                'Mean Test Performance', 'Std Test Performance',
                'Mean Reuse %', 'Std Reuse %',
                'Mean Runtime (min)', 'Std Runtime (min)',
                'Mean Design Algorithm Calls', 'Mean Design Reuses',
                'Number of Runs'
            ])

            # Process each dataset
            for dataset, approaches in sorted(self.results.items()):
                for approach_key, runs in sorted(approaches.items()):
                    if not runs:
                        continue

                    # Extract metrics
                    best_perfs = [r['best_performance'] for r in runs]
                    final_perfs = [r['final_performance'] for r in runs]
                    test_perfs = [r.get('test_results', {}).get('test_performance', 0) for r in runs]
                    runtimes = [r['total_time'] / 60.0 for r in runs]  # Convert to minutes

                    # Reuse percentage (only for MAR approaches)
                    if 'design_reuse_percentage' in runs[0]:
                        reuse_pcts = [r['design_reuse_percentage'] for r in runs]
                        design_calls = [r['num_design_algorithm_calls'] for r in runs]
                        design_reuses = [r['num_design_reuses'] for r in runs]
                    else:
                        reuse_pcts = [0]
                        design_calls = [0]
                        design_reuses = [0]

                    writer.writerow([
                        dataset,
                        runs[0]['approach'],
                        runs[0]['meta_learner'],
                        f"{np.mean(best_perfs):.4f}",
                        f"{np.std(best_perfs):.4f}",
                        f"{np.mean(final_perfs):.4f}",
                        f"{np.std(final_perfs):.4f}",
                        f"{np.mean(test_perfs):.4f}",
                        f"{np.std(test_perfs):.4f}",
                        f"{np.mean(reuse_pcts):.2f}",
                        f"{np.std(reuse_pcts):.2f}",
                        f"{np.mean(runtimes):.2f}",
                        f"{np.std(runtimes):.2f}",
                        f"{np.mean(design_calls):.1f}",
                        f"{np.mean(design_reuses):.1f}",
                        len(runs)
                    ])

        print(f"\nSummary report saved to: {summary_file}")

    def _generate_thesis_format_tables(self):
        """
        Generate results in the exact format used in thesis Chapter 8 tables.
        Includes:
        - Table 8.2 style ranking table
        - Box plot data for Figure 8.7 style plots
        """
        # Table 8.2 format: Rankings per approach
        rankings_file = self.output_dir / f"thesis_rankings_{self.timestamp}.txt"

        with open(rankings_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESULTS IN THESIS CHAPTER 8 TABLE 8.2 FORMAT\n")
            f.write("="*80 + "\n\n")

            for dataset in sorted(self.results.keys()):
                f.write(f"\nDataset: {dataset.upper()}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Approach':<30} {'Meta-Learner':<15} {'Mean Acc':<12} {'Std':<12}\n")
                f.write("-"*80 + "\n")

                for approach_key, runs in sorted(self.results[dataset].items()):
                    if not runs:
                        continue

                    test_perfs = [r.get('test_results', {}).get('test_performance', 0) for r in runs]

                    f.write(f"{runs[0]['approach']:<30} "
                           f"{runs[0]['meta_learner']:<15} "
                           f"{np.mean(test_perfs):<12.4f} "
                           f"{np.std(test_perfs):<12.4f}\n")

                f.write("\n")

        print(f"Thesis-format rankings saved to: {rankings_file}")

        # Box plot data (for creating Figure 8.7 style plots)
        boxplot_file = self.output_dir / f"boxplot_data_{self.timestamp}.json"
        boxplot_data = {}

        for dataset, approaches in self.results.items():
            boxplot_data[dataset] = {}
            for approach_key, runs in approaches.items():
                if not runs:
                    continue
                test_perfs = [r.get('test_results', {}).get('test_performance', 0) for r in runs]
                boxplot_data[dataset][approach_key] = test_perfs

        with open(boxplot_file, 'w') as f:
            json.dump(boxplot_data, f, indent=2)

        print(f"Box plot data saved to: {boxplot_file}")

    def _save_all_results(self):
        """Save complete results dictionary."""
        results_file = self.output_dir / f"all_results_{self.timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(dict(self.results), f, indent=2, default=str)

        print(f"Complete results saved to: {results_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for all meta-learning approaches across different applications"
    )

    parser.add_argument(
        '--application',
        type=str,
        default='cnn',
        choices=['cnn', 'segmentation', 'fuzzyart'],
        help='Application to test (default: cnn). Options: cnn, segmentation, fuzzyart'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Datasets to test. If not specified, uses first dataset for the application.'
    )

    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Run on all available datasets for the chosen application'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=50,
        help='Number of timesteps/epochs for MAR approaches (default: 50)'
    )

    parser.add_argument(
        '--runs',
        type=int,
        default=30,
        help='Number of independent runs per configuration (default: 30)'
    )

    parser.add_argument(
        '--autosklearn-time',
        type=int,
        default=3600,
        help='Time budget for AutoSklearn in seconds (default: 3600)'
    )

    parser.add_argument(
        '--techniques',
        nargs='+',
        default=None,
        choices=['onmar-accuracy', 'onmar-design', 'offmar-accuracy', 'offmar-design', 'autosklearn'],
        help='Specific techniques to run (default: all). Options: onmar-accuracy, onmar-design, offmar-accuracy, offmar-design, autosklearn'
    )

    parser.add_argument(
        '--meta-learners',
        nargs='+',
        default=None,
        choices=['knn', 'rf', 'xgboost'],
        help='Specific meta-learners to run (default: all). Options: knn, rf, xgboost. Note: Not applicable for AutoSklearn.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/test/results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: 1 dataset, 1 run, 10 timesteps'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize runner with specified application
    runner = ExperimentRunner(output_dir=args.output_dir, application=args.application)

    # Determine datasets to use
    if args.all_datasets:
        datasets = runner.application_datasets
    elif args.quick_test:
        datasets = [runner.application_datasets[0]]  # First dataset for quick test
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = [runner.application_datasets[0]]  # Default to first dataset

    # Determine run parameters
    if args.quick_test:
        timesteps = 10
        num_runs = 1
        autosklearn_time = 150  # 2.5 minutes
    else:
        timesteps = args.timesteps
        num_runs = args.runs
        autosklearn_time = args.autosklearn_time

    print(f"\n{'='*80}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Application: {args.application}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Techniques: {', '.join(args.techniques) if args.techniques else 'all'}")
    print(f"Meta-learners: {', '.join(args.meta_learners) if args.meta_learners else 'all'}")
    print(f"Timesteps: {timesteps}")
    print(f"Runs per config: {num_runs}")
    print(f"AutoSklearn time budget: {autosklearn_time}s")
    print(f"{'='*80}\n")

    # Run experiments
    runner.run_full_experiment(
        datasets=datasets,
        timesteps=timesteps,
        num_runs=num_runs,
        time_budget_autosklearn=autosklearn_time,
        techniques=args.techniques,
        meta_learners=args.meta_learners
    )


if __name__ == "__main__":
    main()
