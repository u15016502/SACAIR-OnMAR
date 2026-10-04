# OffMAR: Offline Meta-learning for AutoML in Real-time - Design Prediction

This implementation follows Chapter 8 of the thesis, specifically the OffMAR approach where the meta-learner predicts designs directly.

## Overview

**OffMAR (Offline Meta-learning for AutoML in Real-time)** is a two-phase meta-learning approach:

### Phase 1: Data Collection
- Run the design algorithm (GA/GE) for all timesteps
- Collect meta-features, designs, and performance metrics
- Build knowledge repository
- Prune poorly performing designs (below θp threshold)
- Train meta-learner on remaining high-quality designs

### Phase 2: Design Prediction
- Use trained meta-learner to predict designs from meta-features
- Meta-learner replaces the design algorithm
- Significantly reduced computational cost

## Key Differences from OnMAR

| Aspect | OffMAR | OnMAR |
|--------|---------|-------|
| **Training** | Offline (before deployment) | Online (during execution) |
| **Phases** | Two phases | Single phase |
| **Meta-learner predicts** | **Design** | Accuracy |
| **Decision** | Direct design prediction | Whether to reuse design |
| **Computational cost** | High in Phase 1, low in Phase 2 | Moderate throughout |

## Directory Structure

```
OffMAR/design-prediction/
├── offmar.py              # Main OffMAR implementation
├── config.py              # Configuration and hyperparameters
├── README.md              # This file
├── examples/              # Example scripts
│   ├── run_cnn.py        # CNN configuration example
│   ├── run_segmentation.py  # Segmentation composition example
│   └── run_fuzzyart.py   # Fuzzy ART generation example
└── models/               # Saved trained meta-learners
```

## Supported Applications

1. **CNN Configuration** (`applications/configuration/cnn`)
   - Automated hyperparameter optimization
   - Meta-learner predicts CNN configurations

2. **Image Segmentation Composition** (`applications/composition/segmentation`)
   - Automated algorithm composition
   - Meta-learner predicts component sequences

3. **Fuzzy ART Choice Function Generation** (`applications/generation/fuzzyart`)
   - Automated function generation
   - Meta-learner predicts GE chromosomes

## Meta-Learners

Three meta-learners are implemented (as tested in Chapter 8):

1. **kNN (k-Nearest Neighbors)**
   - Simple, instance-based learning
   - Good for small datasets
   - Default k=5

2. **RF (Random Forest)**
   - Ensemble of decision trees
   - Robust to overfitting
   - Default: 100 trees

3. **XGBoost (Extreme Gradient Boosting)**
   - Advanced gradient boosting
   - Often best performance
   - Default: 100 estimators, depth=6, lr=0.1

## Usage

### Basic Usage

```python
from metalearner import load_approach
OffMARDesignPrediction = load_approach('offmar-design').OffMARDesignPrediction
from metalearner import load_approach_config
KNN_CONFIG = load_approach_config('offmar-design').KNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize application
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)
app.load_data()

# Initialize OffMAR
offmar = OffMARDesignPrediction(
    application=app,
    meta_learner_type='knn',
    theta_p=0.85,
    meta_learner_params={'k': 5}
)

# Run both phases
results = offmar.run_full_offmar(
    dataset_name='mnist',
    timesteps=50,
    initial_design=None
)
```

### Run Phase-by-Phase

```python
# Phase 1: Collect data and train meta-learner
phase_1_results = offmar.phase_1_collect_data(
    dataset_name='mnist',
    timesteps=50,
    initial_design=None
)

# Phase 2: Use meta-learner to predict designs
phase_2_results = offmar.phase_2_predict_designs(
    dataset_name='mnist',
    timesteps=50
)
```

### Using Configuration Presets

```python
from metalearner import load_approach_config
CNN_CONFIG = load_approach_config('offmar-design').CNN_CONFIG

config = CNN_CONFIG['xgboost']  # Get XGBoost config for CNN

offmar = OffMARDesignPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)
```

### Save and Load Models

```python
# Save trained meta-learner
offmar.save_model('models/cnn_mnist_knn.pkl')

# Load trained meta-learner
offmar.load_model('models/cnn_mnist_knn.pkl')

# Now can directly run Phase 2
results = offmar.phase_2_predict_designs(dataset_name='mnist', timesteps=50)
```

## Configuration

### Hyperparameters

**θp (Performance Threshold)**
- Default: 0.85 (from thesis)
- Used to prune poorly performing designs from knowledge repository
- Range: [0.0, 1.0]
- Lower values: Keep more designs (more data, potentially noisier)
- Higher values: Keep only best designs (less data, higher quality)

**Meta-learner Parameters**

*kNN:*
- `k`: Number of neighbors (default: 5)

*Random Forest:*
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)

*XGBoost:*
- `n_estimators`: Number of boosting rounds (default: 100)
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Learning rate/eta (default: 0.1)

### Custom Configuration

```python
from metalearner import load_approach_config
OffMARConfig = load_approach_config('offmar-design').OffMARConfig

custom_config = OffMARConfig(
    meta_learner_type='rf',
    theta_p=0.90,  # Higher threshold
    meta_learner_params={
        'n_estimators': 200,  # More trees
        'max_depth': 10       # Limit depth
    }
)

offmar = OffMARDesignPrediction(
    application=app,
    meta_learner_type=custom_config.meta_learner_type,
    theta_p=custom_config.theta_p,
    meta_learner_params=custom_config.meta_learner_params
)
```

## Meta-Features

OffMAR extracts comprehensive meta-features for each application:

### Application-Agnostic Features
- Dataset properties (size, classes, dimensions)
- Class imbalance metrics
- ELA (Exploratory Landscape Analysis) features:
  - Y-distribution (skewness, kurtosis, peaks)
  - Meta-model (linear/quadratic fit)
  - Dispersion (clustering of good solutions)
  - Information content (landscape complexity)
  - NBC (nearest-better clustering)

### Application-Specific Features

**CNN Configuration:**
- Confusion matrix metrics (TP, TN, FP, FN)
- Performance metrics (accuracy, sensitivity, specificity, F1, etc.)
- Weight distances (cosine, Euclidean, Manhattan, etc.)
- Loss metrics (KL divergence, R2, Pearson correlation)

**Segmentation Composition:**
- Local/Global Consistency Error (LCE/GCE)
- ROC curve and AUC
- Dice coefficient, Cohen-Kappa
- Distance metrics between predicted and ground truth
- Distribution features (Bernoulli, Laplacian, Zipf, etc.)

**Fuzzy ART Generation:**
- Similar to CNN (both are classification tasks)
- L2 layer weight analysis
- Clustering quality metrics

## Examples

### Example 1: CNN Configuration with kNN

```bash
python OffMAR/design-prediction/examples/run_cnn.py
```

### Example 2: Compare All Meta-Learners

```python
from examples.run_cnn import run_offmar_cnn_example

for meta_learner in ['knn', 'rf', 'xgboost']:
    results = run_offmar_cnn_example(
        dataset_name='mnist',
        meta_learner_type=meta_learner,
        timesteps=50
    )
    print(f"{meta_learner}: {results['final_performance']:.4f}")
```

## Performance Expectations

Based on Chapter 8 results:

- **OffMAR often outperforms DDOV** for configuration and generation applications
- **Meta-learner choice matters**:
  - kNN: Good for simple datasets
  - RF: Robust across applications
  - XGBoost: Often best performance but higher computational cost
- **Phase 1 is expensive** (runs full design algorithm)
- **Phase 2 is fast** (meta-learner inference only)

## Troubleshooting

**Empty knowledge repository after pruning:**
- Lower θp threshold
- Increase timesteps in Phase 1 to collect more diverse designs

**Poor Phase 2 performance:**
- Insufficient training data in Phase 1
- θp too high (pruned too many designs)
- Meta-learner underfitting/overfitting

**Memory issues:**
- Reduce number of timesteps
- Use kNN (lower memory) instead of RF/XGBoost
- Implement batched meta-feature extraction

## References

- Chapter 8: Meta-learning for Real-time Design (Thesis)
- Section 8.2.2: OffMAR approach description
- Table 8.2: Performance rankings across applications

## Future Improvements

1. **Incremental learning**: Update meta-learner with new data without full retraining
2. **Transfer learning**: Use meta-learner trained on one dataset for another
3. **Ensemble meta-learners**: Combine predictions from multiple meta-learners
4. **Active learning**: Strategically select which designs to evaluate
5. **Meta-feature selection**: Identify most important features per application
