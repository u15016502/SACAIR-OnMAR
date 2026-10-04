"""
Configuration for OnMAR Accuracy Prediction.

This module contains configurable hyperparameters for the OnMAR approach.
"""

from typing import Dict, Any, Optional


class OnMARConfig:
    """Configuration class for OnMAR hyperparameters."""

    def __init__(
        self,
        meta_learner_type: str = 'knn',
        theta_t: Optional[int] = None,
        theta_p: float = 0.85,
        meta_learner_params: Dict[str, Any] = None
    ):
        """
        Initialize OnMAR configuration.

        Args:
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_t: Timestep threshold to start using meta-learner (None = N/2 from thesis)
            theta_p: Performance threshold for reusing design (default: 0.85 from thesis)
            meta_learner_params: Optional parameters for specific meta-learner
        """
        self.meta_learner_type = meta_learner_type
        self.theta_t = theta_t  # None means N/2 (from Algorithm 9 in thesis)
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OnMARConfig':
        """Create configuration from dictionary."""
        return cls(
            meta_learner_type=config_dict.get('meta_learner_type', 'knn'),
            theta_t=config_dict.get('theta_t'),  # None is valid
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
# Based on Algorithm 9 from Chapter 8

KNN_CONFIG = OnMARConfig(
    meta_learner_type='knn',
    theta_t=None,  # N/2 (set dynamically)
    theta_p=0.85,
    meta_learner_params={
        'k': 5  # Number of neighbors
    }
)

RF_CONFIG = OnMARConfig(
    meta_learner_type='rf',
    theta_t=None,  # N/2 (set dynamically)
    theta_p=0.85,
    meta_learner_params={
        'n_estimators': 100,  # Number of trees
        'max_depth': None     # No depth limit
    }
)

XGBOOST_CONFIG = OnMARConfig(
    meta_learner_type='xgboost',
    theta_t=None,  # N/2 (set dynamically)
    theta_p=0.85,
    meta_learner_params={
        'n_estimators': 100,   # Number of boosting rounds
        'max_depth': 6,        # Maximum tree depth
        'learning_rate': 0.1   # Learning rate (eta)
    }
)

# Application-specific configurations
# These can be customized based on empirical results from Table 8.2

CNN_CONFIG = {
    'knn': OnMARConfig(
        meta_learner_type='knn',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OnMARConfig(
        meta_learner_type='rf',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OnMARConfig(
        meta_learner_type='xgboost',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}

SEGMENTATION_CONFIG = {
    'knn': OnMARConfig(
        meta_learner_type='knn',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OnMARConfig(
        meta_learner_type='rf',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OnMARConfig(
        meta_learner_type='xgboost',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}

FUZZYART_CONFIG = {
    'knn': OnMARConfig(
        meta_learner_type='knn',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'k': 5}
    ),
    'rf': OnMARConfig(
        meta_learner_type='rf',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OnMARConfig(
        meta_learner_type='xgboost',
        theta_t=None,  # N/2
        theta_p=0.85,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    )
}


# Preset configurations for different scenarios

# Conservative: Higher threshold, later start
CONSERVATIVE_CONFIG = OnMARConfig(
    meta_learner_type='xgboost',
    theta_t=None,  # N/2 (could be increased to 2*N/3 for more training data)
    theta_p=0.90,  # Higher threshold = less reuse, more design algorithm calls
    meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
)

# Aggressive: Lower threshold, earlier start
AGGRESSIVE_CONFIG = OnMARConfig(
    meta_learner_type='knn',
    theta_t=None,  # Could be decreased to N/3 for earlier meta-learner use
    theta_p=0.75,  # Lower threshold = more reuse, fewer design algorithm calls
    meta_learner_params={'k': 3}
)

# Balanced: Default settings from thesis
BALANCED_CONFIG = OnMARConfig(
    meta_learner_type='rf',
    theta_t=None,  # N/2 from thesis
    theta_p=0.85,  # Default from thesis
    meta_learner_params={'n_estimators': 100, 'max_depth': None}
)
