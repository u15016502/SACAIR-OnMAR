"""
OffMAR (Offline Meta-learning for AutoML in Real-time) - Design Prediction

This implementation follows Chapter 8 of the thesis, where OffMAR consists of two phases:
Phase 1: Run design algorithm for all timesteps, collect meta-features, designs, and performance
Phase 2: Train meta-learner on collected data, then use it to predict designs

The meta-learner predicts the design itself (not accuracy).
"""

import sys
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod
import time
import pickle
from pathlib import Path
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent.parent.parent))

from metalearner.mar_support import (
    DesignAlgorithm, MetaFeatureEncoder, prune_repository,
)


class OffMARDesignPrediction:
    """
    OffMAR approach for real-time AutoML using offline meta-learning.

    The meta-learner is trained offline in Phase 1, then used in Phase 2 to predict
    designs directly from meta-features, replacing the design algorithm.
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
        Initialize OffMAR design prediction.

        Args:
            application: Application instance (CNN, Segmentation, or FuzzyART)
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_p: Performance threshold for pruning (default: 0.85)
            meta_learner_params: Optional parameters for meta-learner
            n_jobs: Retained for API compatibility; Phase 1 now runs one
                timestep at a time against persistent training state, which
                is inherently sequential
            design_algorithm_params: Optional GA settings, overriding those
                the application suggests
        """
        self.application = application
        self.meta_learner_type = meta_learner_type.lower()
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.design_algorithm_params = design_algorithm_params or {}

        # Built per run.
        self.design_algorithm: Optional[DesignAlgorithm] = None
        self.meta_feature_encoder: Optional[MetaFeatureEncoder] = None
        self._current_timestep = 0
        self.prune_info: Dict[str, Any] = {}
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}

        # Knowledge repository (populated in Phase 1)
        self.knowledge_repository: List[Dict[str, Any]] = []

        # Pre-computed flattened repository for faster meta-learner training
        self.knowledge_repository_flat_X: List[np.ndarray] = []  # Pre-flattened X vectors
        self.knowledge_repository_flat_y: List[np.ndarray] = []  # Design vectors

        # Meta-learner (trained in Phase 1, used in Phase 2)
        self.meta_learner = None

        # Phase tracking
        self.phase_1_complete = False

    def phase_1_collect_data(
        self,
        dataset_name: str,
        timesteps: int,
        initial_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Phase 1: Run design algorithm for all timesteps and collect data.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps (epochs/generations)
            initial_design: Optional initial design (if None, uses random/default)

        Returns:
            Dictionary with Phase 1 results
        """
        print(f"\n=== OffMAR Phase 1: Data Collection ===")
        print(f"Dataset: {dataset_name}")
        print(f"Timesteps: {timesteps}")

        start_time = time.time()

        # Clear knowledge repository
        self.knowledge_repository = []
        self.knowledge_repository_flat_X = []
        self.knowledge_repository_flat_y = []

        # Clear state from any previous run, then load the dataset.
        self.application.reset_run_state()
        self.application.load_data()

        # Build the design algorithm and pin the meta-feature schema.
        self.design_algorithm = DesignAlgorithm(
            self.application,
            random_seed=getattr(self.application, 'random_seed', 42),
            config_overrides=self.design_algorithm_params,
        )
        self.meta_feature_encoder = MetaFeatureEncoder(
            self.application.get_meta_feature_names()
        )

        # One timestep at a time: choose a design, apply it, record what the
        # state looked like when the choice was made.
        #
        # The meta-features must be captured *per timestep*, before that
        # timestep is executed. Running the whole training first and then
        # extracting meta-features for each t afterwards - as this phase used
        # to - reads the same final model state every time, so every entry in
        # the repository ends up a near-duplicate and the meta-learner has
        # nothing to distinguish timesteps by.
        performance_history = []
        designs_history = []
        best_performance = 0.0
        best_design = None
        current_design = initial_design

        print(f"Running design algorithm for {timesteps} timesteps...")

        for t in range(timesteps):
            print(f"\nPhase 1 - Timestep {t+1}/{timesteps}")
            self._current_timestep = t

            meta_features = self.application.extract_meta_features(timestep=t)

            design = self.design_algorithm.propose(timestep=t, incumbent=current_design)

            metrics = self.application.exec_design(design, t)
            performance = self.application.extract_performance(metrics)
            print(f"  Performance: {performance:.4f}")

            self.knowledge_repository.append({
                'timestep': t,
                'meta_features': meta_features,
                'design': design,
                'performance': performance,
            })
            self.knowledge_repository_flat_X.append(
                self._flatten_meta_features(meta_features)
            )
            self.knowledge_repository_flat_y.append(self._encode_design(design))

            performance_history.append(performance)
            designs_history.append(design)
            if performance > best_performance:
                best_performance = performance
                best_design = design

            current_design = design

        phase_1_time = time.time() - start_time

        print(f"Phase 1 complete. Collected {len(self.knowledge_repository)} samples.")
        print(f"Phase 1 runtime: {phase_1_time:.2f} seconds")

        # Prune poorly performing designs
        original_size = len(self.knowledge_repository)
        self._prune_knowledge_repository()
        pruned_size = len(self.knowledge_repository)

        print(f"Pruned {original_size - pruned_size} poorly performing designs (threshold: {self.theta_p})")
        print(f"Knowledge repository size: {pruned_size}")
        if self.prune_info.get('relaxed'):
            print(
                f"  Note: nothing reached θp={self.theta_p} (best was "
                f"{self.prune_info['best_performance']:.4f}); kept the "
                f"{self.prune_info['kept']} best entries so the meta-learner "
                f"still has something to learn from"
            )

        # Train meta-learner
        print(f"\nTraining {self.meta_learner_type.upper()} meta-learner...")
        self._train_meta_learner()

        self.phase_1_complete = True

        return {
            'phase': 1,
            'timesteps': timesteps,
            'samples_collected': original_size,
            'samples_after_pruning': pruned_size,
            'best_performance': best_performance,
            'best_design': best_design,
            'performance_history': performance_history,
            'designs_history': designs_history,
            'phase_1_time': phase_1_time,
            'design_algorithm': self.design_algorithm.get_statistics(),
            'prune_info': self.prune_info
        }

    def phase_2_predict_designs(
        self,
        dataset_name: str,
        timesteps: int
    ) -> Dict[str, Any]:
        """
        Phase 2: Use trained meta-learner to predict designs.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps

        Returns:
            Dictionary with Phase 2 results
        """
        if not self.phase_1_complete:
            raise RuntimeError("Phase 1 must be completed before Phase 2")

        print(f"\n=== OffMAR Phase 2: Design Prediction ===")
        print(f"Dataset: {dataset_name}")
        print(f"Timesteps: {timesteps}")

        start_time = time.time()

        # Phase 2 is a fresh deployment run: clear the training state left by
        # Phase 1 so its trace starts from scratch.
        self.application.reset_run_state()
        self.application.load_data()

        predicted_designs = []
        performances = []
        best_performance = 0.0
        best_design = None

        # For each timestep, predict a design and *apply* it. Predicting
        # without applying would leave the phase with no performance to
        # report, which is what this phase used to do.
        for t in range(timesteps):
            self._current_timestep = t

            # Extract meta-features for current timestep
            meta_features = self.application.extract_meta_features(timestep=t)

            # Predict design using meta-learner
            predicted_design = self._predict_design(meta_features)
            predicted_designs.append(predicted_design)

            # Apply the predicted design for this timestep and measure it
            metrics = self.application.exec_design(predicted_design, t)
            performance = self.application.extract_performance(metrics)
            performances.append(performance)

            if performance > best_performance:
                best_performance = performance
                best_design = predicted_design

            print(f"Phase 2 - Timestep {t+1}/{timesteps}: "
                  f"predicted design, performance {performance:.4f}")

        # Evaluate the network the run produced on the test set
        final_design = predicted_designs[-1]
        test_results = self.application.evaluate()
        test_performance = test_results.get(
            'test_performance', test_results.get('test_accuracy', 0.0)
        )

        phase_2_time = time.time() - start_time

        print(f"\nPhase 2 complete.")
        print(f"Phase 2 runtime: {phase_2_time:.2f} seconds")
        print(f"Best performance: {best_performance:.4f}")
        print(f"Final test performance: {test_performance:.4f}")

        return {
            'phase': 2,
            'timesteps': timesteps,
            'predicted_designs': predicted_designs,
            'final_design': final_design,
            'performance_history': performances,
            'best_performance': best_performance,
            'best_design': best_design,
            'final_performance': performances[-1] if performances else 0.0,
            'test_results': test_results,
            'test_performance': test_performance,
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
            timesteps: Total number of timesteps, split equally between the
                two phases (as in the accuracy-prediction variant, so the two
                are comparable at the same budget)
            initial_design: Optional initial design

        Returns:
            Combined results from both phases
        """
        # Split timesteps between phases
        phase_1_timesteps = timesteps // 2
        phase_2_timesteps = timesteps - phase_1_timesteps

        phase_1_results = self.phase_1_collect_data(
            dataset_name=dataset_name,
            timesteps=phase_1_timesteps,
            initial_design=initial_design
        )

        phase_2_results = self.phase_2_predict_designs(
            dataset_name=dataset_name,
            timesteps=phase_2_timesteps
        )

        total_time = phase_1_results['phase_1_time'] + phase_2_results['phase_2_time']

        return {
            'approach': 'OffMAR-DesignPrediction',
            'meta_learner': self.meta_learner_type,
            'phase_1': phase_1_results,
            'phase_2': phase_2_results,
            'total_time': total_time,
            # Surfaced at the top level as well, because the experiment runner
            # reads results['test_results']['test_performance'] uniformly
            # across all four approaches.
            'test_results': phase_2_results['test_results'],
            'test_performance': phase_2_results['test_performance'],
            'performance_history': phase_2_results['performance_history'],
            'best_performance': max(
                phase_1_results['best_performance'] or 0.0,
                phase_2_results['best_performance']
            ),
            'final_performance': phase_2_results['final_performance'],
            'design_algorithm': phase_1_results['design_algorithm'],
            'prune_info': phase_1_results['prune_info']
        }

    def _prune_knowledge_repository(self):
        """
        Remove designs with performance below theta_p threshold.
        This prevents meta-learner from learning poorly performing designs.

        Called once, at the end of Phase 1, immediately before the
        meta-learner is trained. If nothing clears theta_p the best entries
        are kept, rather than leaving the repository empty and the
        meta-learner untrained.
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
        Meta-learner learns to map meta-features -> designs.
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
            X = np.array([self._flatten_meta_features(entry['meta_features'])
                          for entry in self.knowledge_repository])
            y = np.array([self._encode_design(entry['design'])
                          for entry in self.knowledge_repository])

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

    def _predict_design(self, meta_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use meta-learner to predict design from meta-features.

        Args:
            meta_features: Extracted meta-features

        Returns:
            Predicted design
        """
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained yet")

        # Flatten meta-features to vector
        X = self._flatten_meta_features(meta_features).reshape(1, -1)

        # Predict design encoding
        design_encoding = self.meta_learner.predict(X)[0]

        # Decode back to design dictionary
        design = self._decode_design(design_encoding)

        return design

    def _create_knn_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train k-Nearest Neighbors meta-learner."""
        from sklearn.neighbors import KNeighborsRegressor

        k = self.meta_learner_params.get('k', 5)
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(X, y)
        return knn

    def _create_rf_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train Random Forest meta-learner."""
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
        """Create and train XGBoost meta-learner."""
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
        """Encode meta-features to a fixed-length vector.

        The schema is pinned for the run (see MetaFeatureEncoder): which
        features exist varies by timestep, and vectors of differing length
        cannot be fitted by a meta-learner.
        """
        if self.meta_feature_encoder is None:
            self.meta_feature_encoder = MetaFeatureEncoder(
                self.application.get_meta_feature_names()
            )
        return self.meta_feature_encoder.encode(meta_features)

    def _encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """Encode a design to a fixed-length vector (owned by the application)."""
        return self.application.encode_design(design)

    def _decode_design(self, design_encoding: np.ndarray) -> Dict[str, Any]:
        """Decode a predicted vector back into a buildable design.

        The application clamps every field into its design space, so
        unconstrained regression output still yields a valid design.
        """
        return self.application.decode_design(design_encoding)

    def save_model(self, filepath: str):
        """Save trained meta-learner to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'meta_learner': self.meta_learner,
                'meta_learner_type': self.meta_learner_type,
                'theta_p': self.theta_p,
                'knowledge_repository': self.knowledge_repository,
                'phase_1_complete': self.phase_1_complete
            }, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained meta-learner from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.meta_learner = data['meta_learner']
        self.meta_learner_type = data['meta_learner_type']
        self.theta_p = data['theta_p']
        self.knowledge_repository = data['knowledge_repository']
        self.phase_1_complete = data['phase_1_complete']

        print(f"Model loaded from {filepath}")
