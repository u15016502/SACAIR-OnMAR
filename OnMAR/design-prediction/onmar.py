"""
OnMAR (Online Meta-learning for AutoML in Real-time) - Design Prediction

This implementation is a variant of OnMAR where the meta-learner predicts designs
(instead of accuracy) in an online fashion.

Unlike OffMAR which trains offline, this approach continuously updates the meta-learner
online and predicts new designs at each timestep.
"""

import sys
from typing import Dict, List, Any, Optional
import numpy as np
import time
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from metalearner.mar_support import (
    DesignAlgorithm, MetaFeatureEncoder, prune_repository,
)


class OnMARDesignPrediction:
    """
    OnMAR approach with design prediction for real-time AutoML.

    The meta-learner is trained online (during execution) and directly predicts
    designs from meta-features. The knowledge repository is continuously updated
    throughout execution.
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
        Initialize OnMAR design prediction.

        Args:
            application: Application instance (CNN, Segmentation, or FuzzyART)
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_t: Timestep threshold to start using meta-learner (default: N/2)
            theta_p: Performance threshold for pruning knowledge repository (default: 0.85)
            meta_learner_params: Optional parameters for meta-learner
            design_algorithm_params: Optional GA settings, overriding those
                the application suggests
        """
        self.application = application
        self.meta_learner_type = meta_learner_type.lower()
        self.theta_t = theta_t  # Will be set to N/2 if None
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}
        self.design_algorithm_params = design_algorithm_params or {}

        # Built per run, in run_onmar.
        self.design_algorithm: Optional[DesignAlgorithm] = None
        self.meta_feature_encoder: Optional[MetaFeatureEncoder] = None
        self._current_timestep = 0

        # theta_p pruning happens once, immediately before the meta-learner is
        # first trained (half way through the run), not on every update.
        self.has_pruned = False
        self.prune_info: Dict[str, Any] = {}

        # Knowledge repository (continuously updated online)
        self.knowledge_repository: List[Dict[str, Any]] = []

        # Pre-computed flattened repository for faster meta-learner training
        self.knowledge_repository_flat_X: List[np.ndarray] = []  # Pre-flattened X vectors
        self.knowledge_repository_flat_y: List[np.ndarray] = []  # Design vectors

        # Meta-learner (trained and updated online)
        self.meta_learner = None

        # Current design (tracked for continuity)
        self.current_design = None
        self.current_performance = 0.0

        # Design space information (for encoding/decoding)
        self.design_space = None
        self.design_params = []
        self.param_types = {}  # 'categorical' or 'continuous'
        self.param_values = {}  # Possible values for categorical, (min, max) for continuous

        # Statistics
        self.num_design_algorithm_calls = 0
        self.num_design_predictions = 0

    def run_onmar(
        self,
        dataset_name: str,
        timesteps: int,
        initial_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run OnMAR for all timesteps with online meta-learning and design prediction.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps (epochs/generations)
            initial_design: Optional initial design

        Returns:
            Dictionary with results
        """
        print(f"\n=== OnMAR Design Prediction ===")
        print(f"Dataset: {dataset_name}")
        print(f"Timesteps: {timesteps}")
        print(f"Meta-learner: {self.meta_learner_type.upper()}")
        print(f"θt (start using meta-learner): {self.theta_t if self.theta_t else 'N/2'}")
        print(f"θp (pruning threshold): {self.theta_p}")
        print(f"{'='*50}\n")

        start_time = time.time()

        # Set theta_t to N/2 if not specified
        if self.theta_t is None:
            self.theta_t = timesteps // 2

        # Clear any state left by a previous run, then load the dataset.
        self.application.reset_run_state()
        self.application.load_data()

        # Get design space for encoding/decoding
        self.design_space = self.application.get_design_space()
        self._initialize_design_encoding()

        # Build the design algorithm and pin the meta-feature schema for this
        # run, so every repository entry has the same vector length.
        self.design_algorithm = DesignAlgorithm(
            self.application,
            random_seed=getattr(self.application, 'random_seed', 42),
            config_overrides=self.design_algorithm_params,
        )
        self.meta_feature_encoder = MetaFeatureEncoder(
            self.application.get_meta_feature_names()
        )
        self.has_pruned = False
        self.prune_info = {}

        # Reset statistics
        self.num_design_algorithm_calls = 0
        self.num_design_predictions = 0
        self.knowledge_repository = []
        self.knowledge_repository_flat_X = []
        self.knowledge_repository_flat_y = []

        # Track designs and performances over time
        designs_history = []
        performance_history = []
        best_performance = 0.0
        best_design = None

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
                # Phase 2: Use meta-learner to predict design
                print(f"  Phase 2: Using meta-learner to predict design")
                design = self._predict_design(meta_features)
                self.num_design_predictions += 1

            # Apply design and measure actual performance
            performance = self._evaluate_design(design, t)
            print(f"  Performance: {performance:.4f}")

            # Update knowledge repository with pruning
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

            # Track best
            if performance > best_performance:
                best_performance = performance
                best_design = design

        total_time = time.time() - start_time

        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("Final Evaluation on Test Set")
        # Evaluate the network the run actually produced, rather than
        # retraining the best-scoring design from scratch.
        test_results = self.application.evaluate(self.current_design)
        test_performance = test_results.get(
            'test_performance', test_results.get('test_accuracy', 0.0)
        )
        design_algorithm_stats = self.design_algorithm.get_statistics()
        print(f"Test Performance: {test_performance:.4f}")
        print(f"Design algorithm: {design_algorithm_stats}")

        # Calculate statistics
        final_performance = performance_history[-1] if performance_history else 0.0
        design_algorithm_percentage = (self.num_design_algorithm_calls / timesteps) * 100
        design_prediction_percentage = (self.num_design_predictions / timesteps) * 100

        results = {
            'approach': 'OnMAR-DesignPrediction',
            'meta_learner': self.meta_learner_type,
            'theta_t': self.theta_t,
            'theta_p': self.theta_p,
            'timesteps': timesteps,
            'designs_history': designs_history,
            'performance_history': performance_history,
            'best_performance': best_performance,
            'best_design': best_design,
            'final_performance': final_performance,
            'final_design': designs_history[-1] if designs_history else None,
            'test_results': test_results,
            'test_performance': test_performance,
            'design_algorithm': design_algorithm_stats,
            'prune_info': self.prune_info,
            'application_statistics': (
                self.application.get_run_statistics()
                if hasattr(self.application, 'get_run_statistics') else {}
            ),
            'num_design_algorithm_calls': self.num_design_algorithm_calls,
            'num_design_predictions': self.num_design_predictions,
            'design_algorithm_percentage': design_algorithm_percentage,
            'design_prediction_percentage': design_prediction_percentage,
            'knowledge_repository_size': len(self.knowledge_repository),
            'total_time': total_time
        }

        print(f"\n{'='*50}")
        print("OnMAR Design Prediction Complete")
        print(f"Total runtime: {total_time:.2f}s")
        print(f"Best performance: {best_performance:.4f}")
        print(f"Final performance: {final_performance:.4f}")
        print(f"Design algorithm calls: {self.num_design_algorithm_calls} ({design_algorithm_percentage:.1f}%)")
        print(f"Design predictions: {self.num_design_predictions} ({design_prediction_percentage:.1f}%)")
        print(f"Knowledge repository size: {len(self.knowledge_repository)}")
        print(f"{'='*50}\n")

        return results

    def _initialize_design_encoding(self):
        """Record the design space.

        Encoding and decoding are delegated to the application, which owns a
        structured design space and can therefore encode a variable-length
        design to a fixed-length vector. The per-parameter tables below are
        kept only for the legacy flat spaces some applications still declare.
        """
        space = self.application.get_structured_design_space()
        self.design_params = list(space.option_names)
        self.param_types = {}
        self.param_values = {}
        for name in self.design_params:
            option = space.spec[name]
            if option.get('type') == 'continuous':
                self.param_types[name] = 'continuous'
                self.param_values[name] = (option['min'], option['max'])
            elif option.get('type') == 'categorical':
                self.param_types[name] = 'categorical'
                self.param_values[name] = option['options']
            else:
                self.param_types[name] = option.get('type', 'categorical')
                self.param_values[name] = None
        return

    def _run_design_algorithm(self, initial_design: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the design algorithm to create a new design.

        Args:
            initial_design: Optional starting design

        Returns:
            New design
        """
        # c <- design_algorithm(dataset, t) of Algorithm 9. One GA generation
        # per call, with the population persisting between calls so the search
        # accumulates over the run.
        return self.design_algorithm.propose(
            timestep=self._current_timestep, incumbent=initial_design
        )

    def _evaluate_design(self, design: Dict[str, Any], timestep: int) -> float:
        """
        Evaluate a design by training and measuring performance.

        Args:
            design: Design to evaluate
            timestep: Current timestep

        Returns:
            Performance metric (e.g., accuracy)
        """
        # aa.exec(c, dataset, t): advance the application by one timestep
        # under this design, carrying its training state forward.
        self.last_timestep_metrics = self.application.exec_design(design, timestep)
        return self.application.extract_performance(self.last_timestep_metrics)

    def _update_knowledge_repository(
        self,
        meta_features: Dict[str, Any],
        design: Dict[str, Any],
        performance: float
    ):
        """
        Update knowledge repository with new sample and apply online pruning.
        Also maintains pre-flattened vectors for faster training.

        Args:
            meta_features: Extracted meta-features
            design: Design used
            performance: Measured performance
        """
        # Add to repository
        self.knowledge_repository.append({
            'meta_features': meta_features,
            'design': design,
            'performance': performance
        })

        # Pre-compute flattened version
        x = self._flatten_meta_features(meta_features)
        y = self._encode_design(design)
        self.knowledge_repository_flat_X.append(x)
        self.knowledge_repository_flat_y.append(y)

        # Pruning is deliberately *not* done here. It happens once, in
        # _train_meta_learner, immediately before the meta-learner is first
        # trained. Pruning on every update instead would discard every entry
        # below theta_p as soon as it arrived, so while performance sits below
        # the threshold - which it does for most of a run - the repository
        # would be emptied continuously and the meta-learner would never be
        # trained on anything.

    def _train_meta_learner(self):
        """
        Train meta-learner on current knowledge repository.
        Uses pre-flattened vectors for faster training (4x speedup).
        Meta-learner learns to map meta-features -> designs.
        """
        # theta_p pruning: once, right before the first training. The
        # meta-learner here is trained to reproduce designs, so it should only
        # be shown designs worth reproducing.
        if not self.has_pruned:
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
            self.has_pruned = True

            if self.prune_info.get('relaxed'):
                print(
                    f"  Pruning at θp={self.theta_p}: nothing reached the "
                    f"threshold (best was "
                    f"{self.prune_info['best_performance']:.4f}); kept the "
                    f"{self.prune_info['kept']} best entries instead"
                )
            else:
                print(
                    f"  Pruning at θp={self.theta_p}: dropped "
                    f"{self.prune_info['pruned']}, kept "
                    f"{self.prune_info['kept']}"
                )

        if len(self.knowledge_repository_flat_X) == 0:
            print("  Warning: Knowledge repository empty. Skipping training.")
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

        print(f"  Meta-learner re-trained on {len(X)} samples")

    def _predict_design(self, meta_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use meta-learner to predict design from meta-features.

        Args:
            meta_features: Extracted meta-features

        Returns:
            Predicted design
        """
        if self.meta_learner is None:
            print("  Warning: Meta-learner not trained yet, using random design")
            return self._run_design_algorithm()

        # Flatten meta-features to vector
        X = self._flatten_meta_features(meta_features).reshape(1, -1)

        # Predict design encoding
        design_encoding = self.meta_learner.predict(X)[0]

        # Decode back to design
        design = self._decode_design(design_encoding)

        return design

    def _flatten_meta_features(self, meta_features: Dict[str, Any]) -> np.ndarray:
        """
        Flatten nested meta-features dictionary to 1D numpy array.

        Args:
            meta_features: Nested dictionary of meta-features

        Returns:
            Flattened 1D array
        """
        if self.meta_feature_encoder is None:
            self.meta_feature_encoder = MetaFeatureEncoder(
                self.application.get_meta_feature_names()
            )
        return self.meta_feature_encoder.encode(meta_features)

    def _encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """
        Encode design dictionary to numerical vector.

        Args:
            design: Design dictionary

        Returns:
            Encoded design as numpy array
        """
        # Delegated to the application, which encodes even a
        # variable-length design to a fixed-length vector.
        return self.application.encode_design(design)

    def _decode_design(self, design_encoding: np.ndarray) -> Dict[str, Any]:
        """
        Decode numerical vector back to design dictionary.

        Args:
            design_encoding: Encoded design

        Returns:
            Design dictionary
        """
        # Delegated to the application, which clamps every field back into
        # the design space, so unconstrained regression output still decodes
        # to a design that can actually be built.
        return self.application.decode_design(design_encoding)

    def _create_knn_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train kNN meta-learner."""
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.multioutput import MultiOutputRegressor

        k = self.meta_learner_params.get('k', 5)
        knn = KNeighborsRegressor(n_neighbors=min(k, len(X)))
        meta_learner = MultiOutputRegressor(knn)
        meta_learner.fit(X, y)

        return meta_learner

    def _create_rf_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train Random Forest meta-learner."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', None)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.application.random_seed
        )
        meta_learner = MultiOutputRegressor(rf)
        meta_learner.fit(X, y)

        return meta_learner

    def _create_xgboost_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train XGBoost meta-learner."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        from sklearn.multioutput import MultiOutputRegressor

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', 6)
        learning_rate = self.meta_learner_params.get('learning_rate', 0.1)

        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.application.random_seed
        )
        meta_learner = MultiOutputRegressor(xgb_model)
        meta_learner.fit(X, y)

        return meta_learner

    def save_model(self, filepath: str):
        """
        Save trained meta-learner and knowledge repository.

        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'meta_learner': self.meta_learner,
            'meta_learner_type': self.meta_learner_type,
            'knowledge_repository': self.knowledge_repository,
            'theta_t': self.theta_t,
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
        """
        Load trained meta-learner and knowledge repository.

        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.meta_learner = model_data['meta_learner']
        self.meta_learner_type = model_data['meta_learner_type']
        self.knowledge_repository = model_data['knowledge_repository']
        self.theta_t = model_data['theta_t']
        self.theta_p = model_data['theta_p']
        self.design_space = model_data['design_space']
        self.design_params = model_data['design_params']
        self.param_types = model_data['param_types']
        self.param_values = model_data['param_values']

        print(f"Model loaded from {filepath}")
