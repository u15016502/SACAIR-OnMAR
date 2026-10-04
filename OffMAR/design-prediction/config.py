"""
Configuration for OffMAR Design Prediction.

This module contains configurable hyperparameters for the OffMAR approach.
"""

from typing import Dict, Any


class OffMARConfig:
    """Configuration class for OffMAR hyperparameters."""

    def __init__(
        self,
        meta_learner_type: str = 'knn',
        theta_p: float = 0.85,
        meta_learner_params: Dict[str, Any] = None
    ):
        """
        Initialize OffMAR configuration.

        Args:
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_p: Performance threshold for pruning (default: 0.85 from thesis)
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


# Clustering composition.
#
# theta_p is 0.65 rather than the thesis's 0.85, and that is a deliberate
# departure. theta_p is the predicted-accuracy threshold above which OnMAR
# reuses the current design instead of calling the design algorithm, so it has
# to sit inside the range of accuracies the application actually reaches or the
# mechanism never fires. Clustering accuracy measured here on MNIST, over 128
# instances with the thesis design space, runs to about 0.67 - so at 0.85
# OnMAR would reuse a design exactly never and degenerate into plain search,
# which is the same failure the README records for CIFAR-10 in the CNN
# application. 0.65 is attainable and leaves the reuse-versus-recreate
# decision live. Raise it towards 0.85 for a larger instance sample, where
# accuracies are higher.
CLUSTERING_CONFIG = {
    'knn': OffMARConfig(
        meta_learner_type='knn',
        theta_p=0.65,
        meta_learner_params={'k': 5}
    ),
    'rf': OffMARConfig(
        meta_learner_type='rf',
        theta_p=0.65,
        meta_learner_params={'n_estimators': 100, 'max_depth': None}
    ),
    'xgboost': OffMARConfig(
        meta_learner_type='xgboost',
        theta_p=0.65,
        meta_learner_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    ),
}

# Video classification configuration.
#
# theta_p keeps the thesis's 0.85. On the generated dataset that ships with
# the application (four shapes, so an easy problem) accuracies reach it, so
# the reuse decision is exercised. On a real benchmark - UCF101 or HMDB51 at
# the budgets a short run allows - 0.85 will not be reached, and theta_p
# should be lowered to something inside the attainable range, exactly as for
# clustering above and for CIFAR-10 in the CNN application.
VIDEO_CONFIG = {
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
    ),
}
