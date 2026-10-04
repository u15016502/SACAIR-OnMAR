"""
OnMAR (Online Meta-learning for AutoML in Real-time) - Accuracy Prediction

This implementation follows Chapter 8 of the thesis, where OnMAR uses online meta-learning
to predict the accuracy of designs in real-time.

The meta-learner predicts accuracy (not design), and decides whether to reuse the previous
design or create a new one based on predicted performance.
"""

import sys
from typing import Dict, List, Any, Optional
import numpy as np
import time
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from metalearner.mar_support import DesignAlgorithm, MetaFeatureEncoder


class OnMARAccuracyPrediction:
    """
    OnMAR approach for real-time AutoML using online meta-learning.

    The meta-learner is trained online (during execution) and predicts the accuracy
    of the current design. If predicted accuracy is above threshold, reuse design.
    Otherwise, run the design algorithm to create a new design.
    """

    def __init__(
        self,
        application,
        meta_learner_type: str = 'knn',
        theta_t: Optional[int] = None,
        theta_p: float = 0.85,
        meta_learner_params: Optional[Dict[str, Any]] = None,
        design_algorithm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OnMAR accuracy prediction.

        Args:
            application: Application instance (CNN, Segmentation, or FuzzyART)
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_t: Timestep threshold to start using meta-learner (default: N/2)
            theta_p: Performance threshold for reusing design (default: 0.85)
            meta_learner_params: Optional parameters for meta-learner
            design_algorithm_params: Optional GA settings, overriding those the
                application suggests
        """
        self.application = application
        self.meta_learner_type = meta_learner_type.lower()
        self.theta_t = theta_t  # Will be set to N/2 if None
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}
        self.design_algorithm_params = design_algorithm_params or {}

        # The design algorithm (a GA) and the meta-feature encoder are built
        # per run, in run_onmar.
        self.design_algorithm: Optional[DesignAlgorithm] = None
        self.meta_feature_encoder: Optional[MetaFeatureEncoder] = None

        # Knowledge repository (continuously updated online)
        self.knowledge_repository: List[Dict[str, Any]] = []

        # Pre-computed flattened repository for faster meta-learner training
        self.knowledge_repository_flat_X: List[np.ndarray] = []  # Pre-flattened X vectors
        self.knowledge_repository_flat_y: List[float] = []  # Performance values

        # Meta-learner (trained and updated online)
        self.meta_learner = None

        # Current design (tracked for reuse decision)
        self.current_design = None
        self.current_performance = 0.0
        self._current_timestep = 0

        # Statistics
        self.num_design_algorithm_calls = 0
        self.num_design_reuses = 0

    def run_onmar(
        self,
        dataset_name: str,
        timesteps: int,
        initial_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run OnMAR for all timesteps with online meta-learning.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps (epochs/generations)
            initial_design: Optional initial design

        Returns:
            Dictionary with results
        """
        print(f"\n=== OnMAR Accuracy Prediction ===")
        print(f"Dataset: {dataset_name}")
        print(f"Timesteps: {timesteps}")
        print(f"Meta-learner: {self.meta_learner_type.upper()}")
        print(f"θt (start using meta-learner): {self.theta_t if self.theta_t else 'N/2'}")
        print(f"θp (performance threshold): {self.theta_p}")
        print(f"{'='*50}\n")

        start_time = time.time()

        # Set theta_t to N/2 if not specified (from thesis)
        if self.theta_t is None:
            self.theta_t = timesteps // 2

        # Clear any state left by a previous run, then load the dataset.
        self.application.reset_run_state()
        self.application.load_data()

        # Build the design algorithm and pin the meta-feature schema for this
        # run. The schema must be fixed up front: vectors of differing length
        # cannot be stored in one repository or fitted by a meta-learner.
        self.design_algorithm = DesignAlgorithm(
            self.application,
            random_seed=getattr(self.application, 'random_seed', 42),
            config_overrides=self.design_algorithm_params,
        )
        self.meta_feature_encoder = MetaFeatureEncoder(
            self.application.get_meta_feature_names()
        )

        # Reset statistics
        self.num_design_algorithm_calls = 0
        self.num_design_reuses = 0
        self.knowledge_repository = []
        self.knowledge_repository_flat_X = []
        self.knowledge_repository_flat_y = []

        # Track designs and performances over time
        designs_history = []
        performance_history = []

        # Main OnMAR loop
        for t in range(timesteps):
            print(f"\nTimestep {t+1}/{timesteps}")
            self._current_timestep = t

            # Extract meta-features for current timestep
            meta_features = self.application.extract_meta_features(timestep=t)

            # Decide whether to use meta-learner or design algorithm
            if t < self.theta_t:
                # Phase 1: Always run design algorithm to build knowledge repository
                print(f"  Phase 1: Running design algorithm (t < θt)")
                design = self._run_design_algorithm(initial_design if t == 0 else self.current_design)
                self.num_design_algorithm_calls += 1

            else:
                # Phase 2: Use meta-learner to predict performance
                print(f"  Phase 2: Using meta-learner")

                # Predict performance of current design with current meta-features
                predicted_performance = self._predict_performance(meta_features, self.current_design)
                print(f"  Predicted performance: {predicted_performance:.4f}")

                if predicted_performance >= self.theta_p:
                    # Reuse current design
                    print(f"  Predicted performance ≥ θp ({self.theta_p:.2f}): Reusing design")
                    design = self.current_design
                    self.num_design_reuses += 1
                else:
                    # Create new design
                    print(f"  Predicted performance < θp ({self.theta_p:.2f}): Creating new design")
                    design = self._run_design_algorithm(self.current_design)
                    self.num_design_algorithm_calls += 1

            # Apply design and measure actual performance
            performance = self._evaluate_design(design, t)
            print(f"  Actual performance: {performance:.4f}")

            # Update knowledge repository
            self._update_knowledge_repository(meta_features, design, performance)

            # Re-train meta-learner with updated knowledge
            if t >= self.theta_t - 1:  # Start training at θt - 1
                self._train_meta_learner()

            # Update current design and performance
            self.current_design = design
            self.current_performance = performance

            # Track history
            designs_history.append(design)
            performance_history.append(performance)

        total_time = time.time() - start_time

        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("Evaluating final design on test set...")
        # Evaluates the network the run actually produced; it does not retrain
        # from scratch, so the number reported is the one the run earned.
        test_results = self.application.evaluate(self.current_design)
        test_performance = test_results.get(
            'test_performance', test_results.get('test_accuracy', 0.0)
        )
        design_algorithm_stats = self.design_algorithm.get_statistics()

        # Compute statistics
        design_algorithm_percentage = (self.num_design_algorithm_calls / timesteps) * 100
        design_reuse_percentage = (self.num_design_reuses / timesteps) * 100

        print(f"\n{'='*50}")
        print("OnMAR Results Summary:")
        print(f"{'='*50}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Design algorithm calls: {self.num_design_algorithm_calls} ({design_algorithm_percentage:.1f}%)")
        print(f"Design reuses: {self.num_design_reuses} ({design_reuse_percentage:.1f}%)")
        print(f"Best performance: {max(performance_history):.4f}")
        print(f"Final performance: {performance_history[-1]:.4f}")
        print(f"Test performance: {test_performance:.4f}")
        print(f"Knowledge repository size: {len(self.knowledge_repository)}")
        print(f"Design algorithm: {design_algorithm_stats}")
        print(f"{'='*50}\n")

        return {
            'approach': 'OnMAR-AccuracyPrediction',
            'meta_learner': self.meta_learner_type,
            'timesteps': timesteps,
            'theta_t': self.theta_t,
            'theta_p': self.theta_p,
            'designs_history': designs_history,
            'performance_history': performance_history,
            'best_performance': max(performance_history),
            'final_performance': performance_history[-1],
            'test_results': test_results,
            'test_performance': test_performance,
            'design_algorithm': design_algorithm_stats,
            'application_statistics': (
                self.application.get_run_statistics()
                if hasattr(self.application, 'get_run_statistics') else {}
            ),
            'num_design_algorithm_calls': self.num_design_algorithm_calls,
            'num_design_reuses': self.num_design_reuses,
            'design_algorithm_percentage': design_algorithm_percentage,
            'design_reuse_percentage': design_reuse_percentage,
            'knowledge_repository_size': len(self.knowledge_repository),
            'total_time': total_time
        }

    def _run_design_algorithm(self, initial_design: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the design algorithm to create a new design.

        This is ``c <- design_algorithm(dataset, t)`` of Algorithm 9. One
        generation of the GA runs per call, with the population carried over
        between calls, so the search accumulates across the run.

        Args:
            initial_design: Design currently in use, seeded into the GA's
                initial population

        Returns:
            New design
        """
        return self.design_algorithm.propose(
            timestep=self._current_timestep, incumbent=initial_design
        )

    def _generate_random_design(self) -> Dict[str, Any]:
        """
        Generate a random design from the design space.

        Used as a fallback before the design algorithm has produced anything.

        Returns:
            Random design
        """
        return self.application.sample_design()

    def _evaluate_design(self, design: Dict[str, Any], timestep: int) -> float:
        """
        Evaluate design and return performance.

        Args:
            design: Design to evaluate
            timestep: Current timestep

        Returns:
            Performance value (accuracy, IoU, fitness, etc.)
        """
        # This is aa.exec(c, dataset, t): advance the application by exactly
        # one timestep under this design. For the CNN that means one more
        # epoch of the network already being trained, so the performance trace
        # over a run is one training trajectory under a changing design.
        self.last_timestep_metrics = self.application.exec_design(design, timestep)
        return self.application.extract_performance(self.last_timestep_metrics)

    def _update_knowledge_repository(
        self,
        meta_features: Dict[str, Any],
        design: Dict[str, Any],
        performance: float
    ):
        """
        Add new entry to knowledge repository.
        Also pre-computes and stores flattened version for faster meta-learner training.

        Args:
            meta_features: Extracted meta-features
            design: Design used
            performance: Measured performance
        """
        # Store original entry
        self.knowledge_repository.append({
            'meta_features': meta_features,
            'design': design,
            'performance': performance
        })

        # Pre-compute and store flattened version for faster training
        meta_features_flat = self._flatten_meta_features(meta_features)
        design_flat = self._encode_design(design)
        x = np.concatenate([meta_features_flat, design_flat])

        self.knowledge_repository_flat_X.append(x)
        self.knowledge_repository_flat_y.append(performance)

    def _train_meta_learner(self):
        """
        Train (or re-train) meta-learner on current knowledge repository.
        Uses pre-flattened vectors for faster training (4x speedup).

        For OnMAR, the meta-learner predicts performance (accuracy) given
        meta-features and design.
        """
        if len(self.knowledge_repository_flat_X) == 0:
            return

        # Use pre-computed flattened vectors (4x speedup - no need to loop and flatten)
        X = np.array(self.knowledge_repository_flat_X)
        y = np.array(self.knowledge_repository_flat_y)

        # Create and train meta-learner based on type
        if self.meta_learner_type == 'knn':
            self.meta_learner = self._create_knn_meta_learner(X, y)
        elif self.meta_learner_type == 'rf':
            self.meta_learner = self._create_rf_meta_learner(X, y)
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = self._create_xgboost_meta_learner(X, y)
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

    def _predict_performance(
        self,
        meta_features: Dict[str, Any],
        design: Dict[str, Any]
    ) -> float:
        """
        Use meta-learner to predict performance of design with given meta-features.

        Args:
            meta_features: Current meta-features
            design: Design to evaluate

        Returns:
            Predicted performance
        """
        if self.meta_learner is None:
            return 0.0  # No prediction available yet

        # Prepare input
        meta_features_flat = self._flatten_meta_features(meta_features)
        design_flat = self._encode_design(design)

        # Combine into single input vector
        x = np.concatenate([meta_features_flat, design_flat]).reshape(1, -1)

        # Predict performance
        predicted_performance = self.meta_learner.predict(x)[0]

        return float(predicted_performance)

    def _create_knn_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train k-Nearest Neighbors meta-learner for regression."""
        from sklearn.neighbors import KNeighborsRegressor

        k = self.meta_learner_params.get('k', 5)
        knn = KNeighborsRegressor(n_neighbors=min(k, len(X)), weights='distance')
        knn.fit(X, y)
        return knn

    def _create_rf_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train Random Forest meta-learner for regression."""
        from sklearn.ensemble import RandomForestRegressor

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', None)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        rf.fit(X, y)
        return rf

    def _create_xgboost_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train XGBoost meta-learner for regression."""
        import xgboost as xgb

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', 6)
        learning_rate = self.meta_learner_params.get('learning_rate', 0.1)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X, y)
        return model

    def _flatten_meta_features(self, meta_features: Dict[str, Any]) -> np.ndarray:
        """
        Convert meta-features dictionary to flat numpy array.

        Args:
            meta_features: Dictionary of meta-features

        Returns:
            Flattened feature vector
        """
        if self.meta_feature_encoder is None:
            self.meta_feature_encoder = MetaFeatureEncoder(
                self.application.get_meta_feature_names()
            )
        return self.meta_feature_encoder.encode(meta_features)

    def _encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """
        Encode design dictionary as numpy array.

        Args:
            design: Design dictionary

        Returns:
            Encoded design vector
        """
        # The application owns its encoding, so a variable-length design (a
        # CNN with a varying number of layers) still encodes to a fixed-length
        # vector.
        return self.application.encode_design(design)

    def save_model(self, filepath: str):
        """Save trained meta-learner to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'meta_learner': self.meta_learner,
                'meta_learner_type': self.meta_learner_type,
                'theta_t': self.theta_t,
                'theta_p': self.theta_p,
                'knowledge_repository': self.knowledge_repository,
                'current_design': self.current_design,
                'current_performance': self.current_performance
            }, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained meta-learner from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.meta_learner = data['meta_learner']
        self.meta_learner_type = data['meta_learner_type']
        self.theta_t = data['theta_t']
        self.theta_p = data['theta_p']
        self.knowledge_repository = data['knowledge_repository']
        self.current_design = data['current_design']
        self.current_performance = data['current_performance']

        print(f"Model loaded from {filepath}")
