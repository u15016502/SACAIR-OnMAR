"""
Configuration for OffMAR Accuracy Prediction.

This module contains configurable hyperparameters for the OffMAR accuracy prediction approach.
"""

from typing import Dict, Any


class OffMARConfig:
    """Configuration class for OffMAR accuracy prediction hyperparameters."""

    def __init__(
        self,
        meta_learner_type: str = 'knn',
        theta_p: float = 0.85,
        meta_learner_params: Dict[str, Any] = None
    ):
        """
        Initialize OffMAR accuracy prediction configuration.

        Args:
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_p: Performance threshold for reuse decisions (default: 0.85)
            meta_learner_params: Optional parameters for specific meta-learner
        """
        self.meta_learner_type = meta_learner_type
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OffMARConfig':
        """Create configuration from dictionary."""
        return cls(
            meta_learner_type=config_dict.get('meta_learner_type', 'knn'),
            theta_p=config_dict.get('theta_p', 0.85),
            meta_learner_params=config_dict.get('meta_learner_params', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'meta_learner_type': self.meta_learner_type,
            'theta_p': self.theta_p,
            'meta_learner_params': self.meta_learner_params
        }


# Predefined configurations for different meta-learners

KNN_CONFIG = OffMARConfig(
    meta_learner_type='knn',
    theta_p=0.85,
    meta_learner_params={
        'k': 5  # Number of neighbors
    }
)

RF_CONFIG = OffMARConfig(
    meta_learner_type='rf',
    theta_p=0.85,
    meta_learner_params={
        'n_estimators': 100,  # Number of trees
        'max_depth': None     # No depth limit
    }
)

XGBOOST_CONFIG = OffMARConfig(
    meta_learner_type='xgboost',
    theta_p=0.85,
    meta_learner_params={
        'n_estimators': 100,   # Number of boosting rounds
        'max_depth': 6,        # Maximum tree depth
        'learning_rate': 0.1   # Learning rate (eta)
    }
)

# Preset configurations based on different strategies

CONSERVATIVE_CONFIG = OffMARConfig(
    meta_learner_type='xgboost',
    theta_p=0.90,  # Higher threshold = more stringent reuse criteria
    meta_learner_params={
        'n_estimators': 150,
        'max_depth': 8,
        'learning_rate': 0.05
    }
)

AGGRESSIVE_CONFIG = OffMARConfig(
    meta_learner_type='knn',
    theta_p=0.75,  # Lower threshold = more reuse
    meta_learner_params={
        'k': 3  # Fewer neighbors
    }
)

BALANCED_CONFIG = OffMARConfig(
    meta_learner_type='rf',
    theta_p=0.85,  # Standard threshold
    meta_learner_params={
        'n_estimators': 100,
        'max_depth': None
    }
)

# Application-specific configurations (can be customized based on empirical results)

CNN_CONFIG = {
    'knn': OffMARConfig(
        meta_learner_type='knn',
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OffMARConfig(
        meta_learner_type='rf',
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OffMARConfig(
        meta_learner_type='xgboost',
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}

SEGMENTATION_CONFIG = {
    'knn': OffMARConfig(
        meta_learner_type='knn',
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OffMARConfig(
        meta_learner_type='rf',
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OffMARConfig(
        meta_learner_type='xgboost',
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}

FUZZYART_CONFIG = {
    'knn': OffMARConfig(
        meta_learner_type='knn',
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OffMARConfig(
        meta_learner_type='rf',
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OffMARConfig(
        meta_learner_type='xgboost',
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}
