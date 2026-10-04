"""
Configuration for OnMAR Design Prediction.

This module contains configurable hyperparameters for the OnMAR design prediction approach.
"""

from typing import Dict, Any


class OnMARConfig:
    """Configuration class for OnMAR design prediction hyperparameters."""

    def __init__(
        self,
        meta_learner_type: str = 'knn',
        theta_t: int = None,
        theta_p: float = 0.85,
        meta_learner_params: Dict[str, Any] = None
    ):
        """
        Initialize OnMAR design prediction configuration.

        Args:
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_t: Timestep threshold to start using meta-learner (default: N/2)
            theta_p: Performance threshold for pruning knowledge repository (default: 0.85)
            meta_learner_params: Optional parameters for specific meta-learner
        """
        self.meta_learner_type = meta_learner_type
        self.theta_t = theta_t
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OnMARConfig':
        """Create configuration from dictionary."""
        return cls(
            meta_learner_type=config_dict.get('meta_learner_type', 'knn'),
            theta_t=config_dict.get('theta_t', None),
            theta_p=config_dict.get('theta_p', 0.85),
            meta_learner_params=config_dict.get('meta_learner_params', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'meta_learner_type': self.meta_learner_type,
            'theta_t': self.theta_t,
            'theta_p': self.theta_p,
            'meta_learner_params': self.meta_learner_params
        }


# Predefined configurations for different meta-learners

KNN_CONFIG = OnMARConfig(
    meta_learner_type='knn',
    theta_t=None,  # Will be set to N/2
    theta_p=0.85,
    meta_learner_params={
        'k': 5  # Number of neighbors
    }
)

RF_CONFIG = OnMARConfig(
    meta_learner_type='rf',
    theta_t=None,  # Will be set to N/2
    theta_p=0.85,
    meta_learner_params={
        'n_estimators': 100,  # Number of trees
        'max_depth': None     # No depth limit
    }
)

XGBOOST_CONFIG = OnMARConfig(
    meta_learner_type='xgboost',
    theta_t=None,  # Will be set to N/2
    theta_p=0.85,
    meta_learner_params={
        'n_estimators': 100,   # Number of boosting rounds
        'max_depth': 6,        # Maximum tree depth
        'learning_rate': 0.1   # Learning rate (eta)
    }
)

# Preset configurations based on different strategies

CONSERVATIVE_CONFIG = OnMARConfig(
    meta_learner_type='xgboost',
    theta_t=None,  # Will be set to N/2
    theta_p=0.90,  # Higher threshold = stricter pruning
    meta_learner_params={
        'n_estimators': 150,
        'max_depth': 8,
        'learning_rate': 0.05  # Slower learning
    }
)

AGGRESSIVE_CONFIG = OnMARConfig(
    meta_learner_type='knn',
    theta_t=None,  # Will be set to N/2
    theta_p=0.75,  # Lower threshold = more designs kept
    meta_learner_params={
        'k': 3  # Fewer neighbors = more responsive
    }
)

BALANCED_CONFIG = OnMARConfig(
    meta_learner_type='rf',
    theta_t=None,  # Will be set to N/2
    theta_p=0.85,  # Standard threshold
    meta_learner_params={
        'n_estimators': 100,
        'max_depth': None
    }
)

# Application-specific configurations (can be customized based on empirical results)

CNN_CONFIG = {
    'knn': OnMARConfig(
        meta_learner_type='knn',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OnMARConfig(
        meta_learner_type='rf',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OnMARConfig(
        meta_learner_type='xgboost',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}

SEGMENTATION_CONFIG = {
    'knn': OnMARConfig(
        meta_learner_type='knn',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OnMARConfig(
        meta_learner_type='rf',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OnMARConfig(
        meta_learner_type='xgboost',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}

FUZZYART_CONFIG = {
    'knn': OnMARConfig(
        meta_learner_type='knn',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OnMARConfig(
        meta_learner_type='rf',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OnMARConfig(
        meta_learner_type='xgboost',
        theta_t=None,
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}
