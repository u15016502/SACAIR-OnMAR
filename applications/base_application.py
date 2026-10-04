"""
Base Application Interface

This module defines the abstract base class that all applications must inherit from.
It provides a standard interface for OnMAR, OffMAR, and AutoSklearn to interact with.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from genetic_algorithm.design_space import DesignSpace, space_from_legacy


class BaseApplication(ABC):
    """
    Abstract base class for all AutoML applications.

    This provides a plug-and-play interface for meta-learning approaches
    (OnMAR, OffMAR, AutoSklearn) to interact with different application types
    (configuration, composition, generation).
    """

    def __init__(self, dataset_name: str, random_seed: int = 42):
        """
        Initialize the application.

        Args:
            dataset_name: Name of the dataset to use
            random_seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.is_trained = False
        self._structured_design_space: Optional[DesignSpace] = None

    @abstractmethod
    def load_data(self) -> None:
        """
        Load and prepare the dataset.

        This should handle downloading if necessary and splitting into
        train/validation/test sets.
        """
        pass

    @abstractmethod
    def get_design_space(self) -> Dict[str, List[Any]]:
        """
        Get the design space for this application.

        Returns:
            Dictionary mapping design option names to their possible values.
            For continuous parameters, return [min_value, max_value, 'continuous'].
            For categorical parameters, return list of possible values.

        Example:
            {
                'learning_rate': [0.0001, 0.1, 'continuous'],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'num_layers': [1, 2, 3, 4, 5]
            }
        """
        pass

    @abstractmethod
    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train the application with the given design.

        Args:
            design: Dictionary mapping design option names to chosen values
            timesteps: Optional number of timesteps/epochs for dynamic designs

        Returns:
            Dictionary of performance metrics

        Example:
            {
                'accuracy': 0.95,
                'loss': 0.123,
                'training_time': 45.6
            }
        """
        pass

    @abstractmethod
    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate the current or given design on the test set.

        Args:
            design: Optional design to evaluate. If None, evaluate the last trained design.

        Returns:
            Dictionary of performance metrics on test set
        """
        pass

    # ------------------------------------------------------------------
    # Design space: sampling, validation, encoding
    #
    # All of these route through a single DesignSpace, so the GA, the
    # meta-learners and the application cannot drift apart in their idea of
    # what a design is.
    # ------------------------------------------------------------------

    def get_structured_design_space(self) -> DesignSpace:
        """
        Get this application's design space as a DesignSpace.

        Applications declaring a structured space return it directly; those
        still using the flat ``{name: [values]}`` format are adapted
        automatically.

        Returns:
            A DesignSpace for sampling, mutation, encoding and decoding.
        """
        if self._structured_design_space is None:
            self._structured_design_space = space_from_legacy(self.get_design_space())
        return self._structured_design_space

    def sample_design(self, rng=None) -> Dict[str, Any]:
        """
        Draw a uniformly random valid design.

        Args:
            rng: Optional numpy Generator.

        Returns:
            A valid design.
        """
        return self.get_structured_design_space().sample(rng)

    def validate_design(self, design: Dict[str, Any]) -> bool:
        """
        Validate that a design is within the design space.

        Args:
            design: Design to validate

        Returns:
            True if design is valid, False otherwise
        """
        if design is None:
            return False
        return self.get_structured_design_space().validate(design)

    def encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """
        Encode a design as a fixed-length numeric vector.

        Fixed length matters: this vector is part of the meta-learner's input
        (accuracy prediction) or its target (design prediction), and neither
        can be fitted on ragged data.

        Args:
            design: Design to encode

        Returns:
            Encoded design
        """
        return self.get_structured_design_space().encode(design)

    def decode_design(self, encoding: np.ndarray) -> Dict[str, Any]:
        """
        Decode a vector back into a valid design.

        Values are clamped into the design space, so unconstrained
        meta-learner output still yields a buildable design.

        Args:
            encoding: Encoded design

        Returns:
            Decoded design
        """
        return self.get_structured_design_space().decode(encoding)

    def get_design_encoding_length(self) -> int:
        """
        Get the length of this application's design encoding.

        Returns:
            Number of dimensions in an encoded design
        """
        return self.get_structured_design_space().encoding_length

    # ------------------------------------------------------------------
    # Timestep execution
    # ------------------------------------------------------------------

    def exec_design(self, design: Dict[str, Any], timestep: int) -> Dict[str, float]:
        """
        Apply a design for one timestep and measure it.

        This is ``aa.exec(c, dataset, t)`` of Algorithm 9: it advances the
        application by a single timestep rather than running a whole training
        job. Applications that carry state across timesteps (such as a
        partially trained network) should override this to advance that state
        instead of starting over.

        Args:
            design: Design to apply
            timestep: Current timestep

        Returns:
            Metrics for this timestep, always including 'performance'
        """
        metrics = dict(self.train(design, timesteps=1))
        metrics.setdefault('performance', self.extract_performance(metrics))
        return metrics

    def evaluate_candidate(self, design: Dict[str, Any], timestep: int) -> float:
        """
        Score a candidate design for the design algorithm.

        Used as the GA's fitness function. The default runs a full timestep,
        which is correct but expensive; applications where a timestep is
        costly should override this with a cheaper proxy, since the GA
        evaluates a whole population per generation.

        Args:
            design: Candidate design
            timestep: Current timestep

        Returns:
            Fitness (higher is better)
        """
        return float(self.exec_design(design, timestep).get('performance', 0.0))

    def evaluate_population(
        self,
        designs: List[Dict[str, Any]],
        timestep: int,
    ) -> List[float]:
        """
        Score a whole population of candidate designs.

        The design algorithm evaluates a population per generation, and the
        candidates are independent of one another, so an application that can
        spread that work over CPUs or GPUs should override this. The default
        scores them one at a time.

        Args:
            designs: Candidate designs
            timestep: Current timestep

        Returns:
            One fitness per design, in the same order
        """
        return [self.evaluate_candidate(design, timestep) for design in designs]

    def extract_performance(self, metrics: Dict[str, float]) -> float:
        """
        Pull the primary performance number out of a metrics dictionary.

        Args:
            metrics: Metrics from training or execution

        Returns:
            The primary performance value, or 0.0 if none is recognised
        """
        for key in (
            'performance', 'val_accuracy', 'accuracy', 'best_iou', 'final_iou',
            'best_fitness', 'final_fitness', 'dice_coefficient',
        ):
            if key in metrics:
                return float(metrics[key])
        return 0.0

    def has_builtin_design_algorithm(self) -> bool:
        """
        Whether this application's own ``train`` already searches for designs.

        Applications whose ``train`` embeds a search (and therefore returns a
        'best_design') report True, and the meta-learning layer uses that
        search. Applications that merely apply the design they are given
        report False, and the meta-learning layer runs the GA for them.

        Returns:
            True if ``train`` performs its own design search
        """
        return False

    def get_default_design(self) -> Optional[Dict[str, Any]]:
        """
        A reasonable hand-picked design to start a search from.

        The design algorithm gets only theta_t calls in a run, which is far
        too few generations to discover a competent design from an entirely
        random population. Seeding one known-good individual gives the search
        a floor to improve on. Return None if the application has no such
        design.

        Returns:
            A valid design, or None
        """
        return None

    def get_design_algorithm_config(self) -> Dict[str, Any]:
        """
        Suggested GA hyperparameters for this application.

        Returns:
            Keyword arguments for GeneticAlgorithm; empty to accept defaults
        """
        return {}

    def reset_run_state(self) -> None:
        """
        Clear per-run state so a fresh run starts from a clean slate.

        Called at the start of each meta-learning run. Applications holding
        state across timesteps (model weights, optimiser state) must clear it
        here, or runs leak into one another.
        """
        self.reset()

    def get_meta_feature_names(self) -> Optional[List[str]]:
        """
        Get the fixed meta-feature schema, if this application declares one.

        Returns:
            Ordered feature names, or None if the schema is not fixed
        """
        return None

    def get_meta_features(self) -> Dict[str, float]:
        """
        Extract meta-features from the dataset for meta-learning.

        Returns:
            Dictionary of meta-features

        Example:
            {
                'num_samples': 60000,
                'num_features': 784,
                'num_classes': 10,
                'class_balance': 0.95
            }
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def reset(self) -> None:
        """
        Reset the application to initial state.
        Useful for running multiple experiments.
        """
        self.is_trained = False

    @abstractmethod
    def get_application_type(self) -> str:
        """
        Get the type of application.

        Returns:
            One of: 'configuration', 'composition', 'generation'
        """
        pass

    def supports_dynamic_designs(self) -> bool:
        """
        Check if this application supports dynamic designs (designs that change over time).

        Returns:
            True if dynamic designs are supported, False otherwise
        """
        return False

    def get_num_timesteps(self) -> int:
        """
        Get the number of timesteps for dynamic designs.

        Returns:
            Number of timesteps (e.g., epochs for training)
        """
        return 1
