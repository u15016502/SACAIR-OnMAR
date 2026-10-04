"""
OffMAR (Offline Meta-learning for AutoML in Real-time) - Accuracy Prediction

This implementation is a variant of OffMAR where the meta-learner predicts accuracy
(instead of designs) in an offline fashion.

Unlike OnMAR which trains online, this approach trains the meta-learner once in Phase 1,
then uses it to make reuse decisions in Phase 2 based on predicted accuracy.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from metalearner.mar_support import (
    DesignAlgorithm, MetaFeatureEncoder, prune_repository,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


class OffMARAccuracyPrediction:
    """
    OffMAR approach with accuracy prediction for real-time AutoML.

    The meta-learner is trained offline (Phase 1) and predicts accuracy to decide
    whether to reuse the current design or create a new one (Phase 2).
    """

    def __init__(
        self,
        application,
        meta_learner_type: str = 'knn',
        theta_p: float = 0.85,
        meta_learner_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = None,
        design_algorithm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OffMAR accuracy prediction.

        Args:
            application: Application instance (CNN, Segmentation, or FuzzyART)
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_p: Performance threshold for reusing design (default: 0.85)
            meta_learner_params: Optional parameters for meta-learner
            n_jobs: Number of parallel jobs for Phase 1 data collection (default: CPU count - 1)
        """
        self.application = application
        self.meta_learner_type = meta_learner_type.lower()
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}
        self.design_algorithm_params = design_algorithm_params or {}

        # Built per run.
        self.design_algorithm: Optional[DesignAlgorithm] = None
        self.meta_feature_encoder: Optional[MetaFeatureEncoder] = None
        self._current_timestep = 0
        self.prune_info: Dict[str, Any] = {}

        # Pre-encoded repository, kept index-aligned with knowledge_repository.
        self.knowledge_repository_flat_X: List[np.ndarray] = []
        self.knowledge_repository_flat_y: List[float] = []

        # Knowledge repository (built in Phase 1)
        self.knowledge_repository: List[Dict[str, Any]] = []

        # Meta-learner (trained once in Phase 1)
        self.meta_learner = None

        # Design space information
        self.design_space = None
        self.design_params = []
        self.param_types = {}
        self.param_values = {}

    def phase_1_collect_data(
        self,
        dataset_name: str,
        timesteps: int,
        initial_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Phase 1: Collect data by running design algorithm for all timesteps,
        build knowledge repository, and train meta-learner.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps for Phase 1
            initial_design: Optional initial design

        Returns:
            Phase 1 results dictionary
        """
        print(f"\n=== OffMAR Accuracy Prediction - Phase 1 ===")
        print(f"Dataset: {dataset_name}")
        print(f"Timesteps: {timesteps}")
        print(f"Meta-learner: {self.meta_learner_type.upper()}")
        print(f"θp (performance threshold): {self.theta_p}")
        print(f"{'='*50}\n")

        start_time = time.time()

        # Clear state from any previous run, then load the dataset.
        self.application.reset_run_state()
        self.application.load_data()

        # Get design space for encoding
        self.design_space = self.application.get_design_space()
        self._initialize_design_encoding()

        # Build the design algorithm and pin the meta-feature schema, so every
        # repository entry encodes to the same length.
        self.design_algorithm = DesignAlgorithm(
            self.application,
            random_seed=getattr(self.application, 'random_seed', 42),
            config_overrides=self.design_algorithm_params,
        )
        self.meta_feature_encoder = MetaFeatureEncoder(
            self.application.get_meta_feature_names()
        )

        # Reset knowledge repository
        self.knowledge_repository = []
        self.knowledge_repository_flat_X = []
        self.knowledge_repository_flat_y = []

        # Track performance
        performance_history = []
        designs_history = []
        best_performance = 0.0
        best_design = None

        # Collect data for all timesteps
        current_design = initial_design

        for t in range(timesteps):
            print(f"\nPhase 1 - Timestep {t+1}/{timesteps}")
            self._current_timestep = t

            # Extract meta-features
            meta_features = self.application.extract_meta_features(timestep=t)

            # Run design algorithm
            design = self._run_design_algorithm(current_design)

            # Evaluate design
            performance = self._evaluate_design(design, t)
            print(f"  Performance: {performance:.4f}")

            # Add to knowledge repository (pruning happens once, below,
            # immediately before the meta-learner is trained)
            self.knowledge_repository.append({
                'meta_features': meta_features,
                'design': design,
                'performance': performance
            })
            self.knowledge_repository_flat_X.append(
                np.concatenate([
                    self._flatten_meta_features(meta_features),
                    self._encode_design(design),
                ])
            )
            self.knowledge_repository_flat_y.append(performance)

            # Track history
            performance_history.append(performance)
            designs_history.append(design)

            # Track best
            if performance > best_performance:
                best_performance = performance
                best_design = design

            # Update current design for next iteration
            current_design = design

        # Prune the knowledge repository, once, immediately before the
        # meta-learner is trained.
        samples_before = len(self.knowledge_repository)
        self._prune_knowledge_repository()
        samples_after = len(self.knowledge_repository)
        if self.prune_info.get('relaxed'):
            print(
                f"\n  Note: too few entries reached θp={self.theta_p} to "
                f"fit on (best was "
                f"{self.prune_info['best_performance']:.4f}); kept the "
                f"{self.prune_info['kept']} best entries so the meta-learner "
                f"still has something to learn from"
            )

        print(f"\n{'='*50}")
        print(f"Phase 1 Complete - Knowledge Repository")
        print(f"  Samples collected: {samples_before}")
        print(f"  Samples after pruning (performance ≥ {self.theta_p}): {samples_after}")
        print(f"  Pruned: {samples_before - samples_after}")
        print(f"{'='*50}\n")

        # Train meta-learner on pruned repository
        self._train_meta_learner()

        phase_1_time = time.time() - start_time

        return {
            'phase': 'Phase 1 - Data Collection',
            'samples_collected': samples_before,
            'samples_after_pruning': samples_after,
            'performance_history': performance_history,
            'designs_history': designs_history,
            'best_performance': best_performance,
            'best_design': best_design,
            'phase_1_time': phase_1_time
        }

    def phase_2_predict_and_reuse(
        self,
        dataset_name: str,
        timesteps: int
    ) -> Dict[str, Any]:
        """
        Phase 2: Use trained meta-learner to predict accuracy and make reuse decisions.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps for Phase 2

        Returns:
            Phase 2 results dictionary
        """
        print(f"\n=== OffMAR Accuracy Prediction - Phase 2 ===")
        print(f"Using trained meta-learner for reuse decisions")
        print(f"Timesteps: {timesteps}")
        print(f"{'='*50}\n")

        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained. Run phase_1_collect_data first.")

        start_time = time.time()

        # Phase 2 is a fresh deployment run, not a continuation of Phase 1:
        # clear the training state so its trace starts from scratch.
        self.application.reset_run_state()
        self.application.load_data()

        # Track performance and decisions
        performance_history = []
        designs_history = []
        best_performance = 0.0
        best_design = None
        num_reuses = 0
        num_new_designs = 0

        # Start with a random design
        current_design = self._run_design_algorithm(None)

        for t in range(timesteps):
            print(f"\nPhase 2 - Timestep {t+1}/{timesteps}")
            self._current_timestep = t

            # Extract meta-features
            meta_features = self.application.extract_meta_features(timestep=t)

            # Predict accuracy of current design with current meta-features
            predicted_performance = self._predict_performance(meta_features, current_design)
            print(f"  Predicted performance: {predicted_performance:.4f}")

            # Decision: reuse or create new
            if predicted_performance >= self.theta_p:
                print(f"  Predicted performance ≥ θp ({self.theta_p:.2f}): Reusing design")
                design = current_design
                num_reuses += 1
            else:
                print(f"  Predicted performance < θp ({self.theta_p:.2f}): Creating new design")
                design = self._run_design_algorithm(current_design)
                num_new_designs += 1

            # Evaluate design
            performance = self._evaluate_design(design, t)
            print(f"  Actual performance: {performance:.4f}")

            # Track history
            performance_history.append(performance)
            designs_history.append(design)

            # Track best
            if performance > best_performance:
                best_performance = performance
                best_design = design

            # Update current design
            current_design = design

        phase_2_time = time.time() - start_time

        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("Final Evaluation on Test Set")
        # Evaluate the network the run produced rather than retraining the
        # best-scoring design from scratch.
        test_results = self.application.evaluate(current_design)
        test_performance = test_results.get(
            'test_performance', test_results.get('test_accuracy', 0.0)
        )
        print(f"Test Performance: {test_performance:.4f}")
        print(f"Design algorithm: {self.design_algorithm.get_statistics()}")
        print(f"{'='*50}\n")

        # Calculate statistics
        final_performance = performance_history[-1] if performance_history else 0.0
        reuse_percentage = (num_reuses / timesteps) * 100
        new_design_percentage = (num_new_designs / timesteps) * 100

        return {
            'phase': 'Phase 2 - Prediction and Reuse',
            'performance_history': performance_history,
            'designs_history': designs_history,
            'best_performance': best_performance,
            'best_design': best_design,
            'final_performance': final_performance,
            'test_results': test_results,
            'num_reuses': num_reuses,
            'num_new_designs': num_new_designs,
            'reuse_percentage': reuse_percentage,
            'new_design_percentage': new_design_percentage,
            'phase_2_time': phase_2_time
        }

    def run_full_offmar(
        self,
        dataset_name: str,
        timesteps: int,
        initial_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run both phases of OffMAR sequentially.

        Args:
            dataset_name: Name of dataset
            timesteps: Number of timesteps (split equally between phases)
            initial_design: Optional initial design for Phase 1

        Returns:
            Combined results from both phases
        """
        # Split timesteps between phases
        phase_1_timesteps = timesteps // 2
        phase_2_timesteps = timesteps - phase_1_timesteps

        # Run Phase 1
        phase_1_results = self.phase_1_collect_data(
            dataset_name=dataset_name,
            timesteps=phase_1_timesteps,
            initial_design=initial_design
        )

        # Run Phase 2
        phase_2_results = self.phase_2_predict_and_reuse(
            dataset_name=dataset_name,
            timesteps=phase_2_timesteps
        )

        total_time = phase_1_results['phase_1_time'] + phase_2_results['phase_2_time']

        return {
            'approach': 'OffMAR-AccuracyPrediction',
            'meta_learner': self.meta_learner_type,
            'theta_p': self.theta_p,
            'phase_1': phase_1_results,
            'phase_2': phase_2_results,
            'total_time': total_time,
            'best_performance': max(
                phase_1_results['best_performance'],
                phase_2_results['best_performance']
            ),
            'final_performance': phase_2_results['final_performance'],
            # Surfaced at the top level as well, because the experiment runner
            # reads results['test_results']['test_performance'] uniformly
            # across all four approaches.
            'test_results': phase_2_results['test_results'],
            'test_performance': phase_2_results['test_results'].get(
                'test_performance', phase_2_results['test_results'].get('test_accuracy', 0.0)
            ),
            'performance_history': phase_2_results['performance_history'],
            'design_algorithm': self.design_algorithm.get_statistics(),
            'prune_info': self.prune_info
        }

    def _initialize_design_encoding(self):
        """Record the design space.

        Encoding and decoding are delegated to the application, which owns a
        structured design space and can encode a variable-length design to a
        fixed-length vector. The per-option tables below are kept only for the
        legacy flat spaces some applications still declare; reading the
        structured spec positionally (as this used to) breaks on it.
        """
        space = self.application.get_structured_design_space()
        self.design_params = list(space.option_names)
        self.param_types = {}
        self.param_values = {}

        for name in self.design_params:
            option = space.spec[name]
            option_type = option.get('type', 'categorical')
            if option_type == 'continuous':
                self.param_types[name] = 'continuous'
                self.param_values[name] = (option['min'], option['max'])
            elif option_type == 'categorical':
                self.param_types[name] = 'categorical'
                self.param_values[name] = option['options']
            else:
                self.param_types[name] = option_type
                self.param_values[name] = None

    def _run_design_algorithm(self, initial_design: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the design algorithm (a GA) to create a new design.

        One generation per call, with the population persisting between calls,
        so the search accumulates across Phase 1.
        """
        return self.design_algorithm.propose(
            timestep=self._current_timestep, incumbent=initial_design
        )

    def _evaluate_design(self, design: Dict[str, Any], timestep: int) -> float:
        """Apply a design for one timestep and measure it.

        aa.exec(c, dataset, t): advances the application's training state by
        one timestep rather than starting a new training run.
        """
        self.last_timestep_metrics = self.application.exec_design(design, timestep)
        return self.application.extract_performance(self.last_timestep_metrics)

    def _prune_knowledge_repository(self):
        """Remove designs with performance below theta_p.

        Called once, at the end of Phase 1, immediately before the
        meta-learner is trained. If nothing clears theta_p, the best entries
        are kept instead of leaving the repository empty, which would leave
        the meta-learner untrained and Phase 2 running on random designs.
        """
        (
            self.knowledge_repository,
            self.knowledge_repository_flat_X,
            self.knowledge_repository_flat_y,
            self.prune_info,
        ) = prune_repository(
            self.knowledge_repository,
            self.knowledge_repository_flat_X,
            self.knowledge_repository_flat_y,
            theta_p=self.theta_p,
        )

    def _train_meta_learner(self):
        """
        Train meta-learner on knowledge repository.
        Uses vectorized processing for faster training (4x speedup).
        Meta-learner learns to map (meta-features + design) -> accuracy.
        """
        if len(self.knowledge_repository) == 0:
            raise ValueError("Knowledge repository is empty after pruning. Lower theta_p threshold.")

        # Use the vectors encoded as the repository was built; they are
        # index-aligned with it and already on the pinned schema.
        print(f"Preparing training data from {len(self.knowledge_repository)} samples...")
        if len(self.knowledge_repository_flat_X) == len(self.knowledge_repository):
            X = np.array(self.knowledge_repository_flat_X)
            y = np.array(self.knowledge_repository_flat_y)
        else:
            X = np.array([
                np.concatenate([
                    self._flatten_meta_features(entry['meta_features']),
                    self._encode_design(entry['design'])
                ])
                for entry in self.knowledge_repository
            ])
            y = np.array([entry['performance'] for entry in self.knowledge_repository])

        # Create and train meta-learner based on type
        if self.meta_learner_type == 'knn':
            self.meta_learner = self._create_knn_meta_learner(X, y)
        elif self.meta_learner_type == 'rf':
            self.meta_learner = self._create_rf_meta_learner(X, y)
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = self._create_xgboost_meta_learner(X, y)
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

        print(f"Meta-learner trained on {len(X)} samples")

    def _predict_performance(self, meta_features: Dict[str, Any], design: Dict[str, Any]) -> float:
        """
        Use meta-learner to predict performance (accuracy) of a design.

        Args:
            meta_features: Extracted meta-features
            design: Design to evaluate

        Returns:
            Predicted performance (scalar)
        """
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained yet")

        # Combine meta-features and design into single input vector
        meta_features_flat = self._flatten_meta_features(meta_features)
        design_flat = self._encode_design(design)
        combined = np.concatenate([meta_features_flat, design_flat])
        X = combined.reshape(1, -1)

        # Predict performance
        predicted_performance = self.meta_learner.predict(X)[0]

        return predicted_performance

    def _flatten_meta_features(self, meta_features: Dict[str, Any]) -> np.ndarray:
        """Encode meta-features to a fixed-length vector (see MetaFeatureEncoder)."""
        if self.meta_feature_encoder is None:
            self.meta_feature_encoder = MetaFeatureEncoder(
                self.application.get_meta_feature_names()
            )
        return self.meta_feature_encoder.encode(meta_features)

    def _encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """Encode a design to a fixed-length vector (owned by the application)."""
        return self.application.encode_design(design)

    def _create_knn_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train kNN meta-learner for regression."""
        from sklearn.neighbors import KNeighborsRegressor

        k = self.meta_learner_params.get('k', 5)
        meta_learner = KNeighborsRegressor(n_neighbors=min(k, len(X)))
        meta_learner.fit(X, y)

        return meta_learner

    def _create_rf_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train Random Forest meta-learner for regression."""
        from sklearn.ensemble import RandomForestRegressor

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', None)

        meta_learner = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.application.random_seed
        )
        meta_learner.fit(X, y)

        return meta_learner

    def _create_xgboost_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train XGBoost meta-learner for regression."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', 6)
        learning_rate = self.meta_learner_params.get('learning_rate', 0.1)

        meta_learner = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.application.random_seed
        )
        meta_learner.fit(X, y)

        return meta_learner

    def save_model(self, filepath: str):
        """Save trained meta-learner and knowledge repository."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'meta_learner': self.meta_learner,
            'meta_learner_type': self.meta_learner_type,
            'knowledge_repository': self.knowledge_repository,
            'theta_p': self.theta_p,
            'design_space': self.design_space,
            'design_params': self.design_params,
            'param_types': self.param_types,
            'param_values': self.param_values
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained meta-learner and knowledge repository."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.meta_learner = model_data['meta_learner']
        self.meta_learner_type = model_data['meta_learner_type']
        self.knowledge_repository = model_data['knowledge_repository']
        self.theta_p = model_data['theta_p']
        self.design_space = model_data['design_space']
        self.design_params = model_data['design_params']
        self.param_types = model_data['param_types']
        self.param_values = model_data['param_values']

        print(f"Model loaded from {filepath}")
