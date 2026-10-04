# OffMAR Accuracy Prediction

**Offline Meta-learning for AutoML in Real-time** with accuracy prediction for reuse decisions.

This is a variant of OffMAR where the meta-learner **predicts accuracy** (like OnMAR accuracy-prediction) but learns **offline** (like OffMAR design-prediction). This combines reproducible offline training with flexible reuse decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Key Differences](#key-differences)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Meta-Learners](#meta-learners)
9. [Hyperparameters](#hyperparameters)
10. [Performance Expectations](#performance-expectations)
11. [Troubleshooting](#troubleshooting)

---

## Overview

OffMAR Accuracy Prediction is a **hybrid approach** that combines:

- **Offline Learning** (from OffMAR-DesignPrediction): Meta-learner trained once in Phase 1, fixed in Phase 2
- **Accuracy Prediction** (from OnMAR-AccuracyPrediction): Meta-learner predicts accuracy to make reuse decisions

### When to Use This Approach

✅ **Use OffMAR Accuracy Prediction when:**
- You need reproducible results (same meta-learner → same decisions)
- You want flexible reuse decisions (not forced to use predicted designs)
- You can afford expensive Phase 1 training
- You want to reuse the meta-learner across multiple Phase 2 runs

❌ **Don't use this approach when:**
- Data distribution changes over time (use OnMAR)
- You prefer direct design prediction (use OffMAR-DesignPrediction)
- You need maximum performance (OnMAR-AccuracyPrediction is typically better)

---

## How It Works

### Two-Phase Algorithm

```
PHASE 1: Data Collection and Training (Offline)
For each timestep t in Phase 1:
    1. Extract meta-features from current state
    2. Run design algorithm → get design
    3. Evaluate design → get performance
    4. Add (meta-features, design, performance) to repository

After all timesteps:
    5. Prune repository (keep only performance ≥ θp)
    6. Train meta-learner: (meta-features + design) → accuracy

PHASE 2: Accuracy Prediction and Reuse (Online Deployment)
For each timestep t in Phase 2:
    1. Extract meta-features from current state
    2. Use meta-learner to predict accuracy of current design
    3. If predicted accuracy ≥ θp:
          Reuse current design
       Else:
          Create new design with design algorithm
    4. Evaluate chosen design
    5. Continue to next timestep
```

### Phase 1: Offline Data Collection

- **Goal**: Build knowledge repository and train meta-learner
- **Process**: Run design algorithm for all timesteps
- **Output**: Trained meta-learner (fixed for Phase 2)
- **Cost**: Expensive (100% design algorithm calls)

### Phase 2: Online Deployment with Reuse Decisions

- **Goal**: Use meta-learner to minimize design algorithm calls
- **Process**: Predict accuracy, reuse or create based on threshold
- **Output**: Final design and performance
- **Cost**: Variable (depends on how often predicted accuracy ≥ θp)

---

## Key Differences

### vs. OnMAR Accuracy-Prediction

| Aspect | OffMAR Accuracy | OnMAR Accuracy |
|--------|----------------|----------------|
| **Learning type** | Offline (once) | Online (continuous) |
| **Meta-learner updates** | Fixed after Phase 1 | Every timestep |
| **Reproducibility** | ⭐⭐⭐⭐⭐ (same results) | ⭐⭐ (may vary) |
| **Adaptability** | ⭐⭐ (cannot adapt) | ⭐⭐⭐⭐⭐ (adapts online) |
| **Reusability** | ⭐⭐⭐⭐⭐ (use many times) | ⭐⭐ (one-time use) |

### vs. OffMAR Design-Prediction

| Aspect | OffMAR Accuracy | OffMAR Design |
|--------|----------------|---------------|
| **Meta-learner output** | Scalar accuracy | Design vector |
| **Decision in Phase 2** | Reuse OR create (conditional) | Always use predicted design |
| **Flexibility** | ⭐⭐⭐⭐ (binary decision) | ⭐⭐ (must use prediction) |
| **Design algorithm calls** | Variable in Phase 2 | Zero in Phase 2 |

### vs. OnMAR Design-Prediction

| Aspect | OffMAR Accuracy | OnMAR Design |
|--------|----------------|--------------|
| **Learning type** | Offline | Online |
| **Prediction target** | Accuracy | Design |
| **Phase 1 cost** | 100% calls | 50% calls (only until θt) |
| **Phase 2 adaptability** | None (fixed) | High (continuous learning) |

---

## Installation

### Prerequisites

```bash
pip install numpy scikit-learn xgboost matplotlib
```

### Optional (for specific applications)
```bash
# For CNN application
pip install torch torchvision

# For segmentation application
pip install opencv-python scikit-image

# For Fuzzy ART application
pip install nltk
```

---

## Quick Start

### Basic Usage

```python
from metalearner import load_approach
OffMARAccuracyPrediction = load_approach('offmar-accuracy').OffMARAccuracyPrediction
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize application
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Initialize OffMAR Accuracy Prediction
offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type='rf',  # or 'knn', 'xgboost'
    theta_p=0.85,  # Reuse threshold
    meta_learner_params={'n_estimators': 100}
)

# Run both phases
results = offmar.run_full_offmar(
    dataset_name='mnist',
    timesteps=50  # Split equally: 25 for Phase 1, 25 for Phase 2
)

# Access results
print(f"Phase 1 best: {results['phase_1']['best_performance']}")
print(f"Phase 2 reuse %: {results['phase_2']['reuse_percentage']}")
print(f"Test performance: {results['test_performance']}")
```

### Separate Phase Execution

```python
# Phase 1: Train meta-learner offline
phase_1_results = offmar.phase_1_collect_data(
    dataset_name='mnist',
    timesteps=30
)

# Save trained meta-learner
offmar.save_model('models/offmar_acc_mnist.pkl')

# ... Later, in production ...

# Load trained meta-learner
offmar_new = OffMARAccuracyPrediction(app, meta_learner_type='rf')
offmar_new.load_model('models/offmar_acc_mnist.pkl')

# Phase 2: Use for reuse decisions (can run multiple times!)
phase_2_results = offmar_new.phase_2_predict_and_reuse(
    dataset_name='mnist',
    timesteps=50
)
```

---

## Configuration

### Using Predefined Configurations

```python
from metalearner import load_approach_config
CNN_CONFIG = load_approach_config('offmar-accuracy').CNN_CONFIG

# Get configuration for specific meta-learner
config = CNN_CONFIG['rf']  # or 'knn', 'xgboost'

offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)
```

### Preset Strategies

```python
from metalearner import load_approach_config
config = load_approach_config('offmar-accuracy')
CONSERVATIVE_CONFIG = config.CONSERVATIVE_CONFIG  # Higher θp (0.90)
AGGRESSIVE_CONFIG = config.AGGRESSIVE_CONFIG      # Lower θp (0.75)
BALANCED_CONFIG = config.BALANCED_CONFIG          # Standard θp (0.85)

offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type=BALANCED_CONFIG.meta_learner_type,
    theta_p=BALANCED_CONFIG.theta_p,
    meta_learner_params=BALANCED_CONFIG.meta_learner_params
)
```

### Custom Configuration

```python
offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type='xgboost',
    theta_p=0.88,  # Custom reuse threshold
    meta_learner_params={
        'n_estimators': 150,
        'max_depth': 8,
        'learning_rate': 0.05
    }
)
```

---

## Usage Examples

### Example 1: CNN Configuration

```python
from metalearner import load_approach
OffMARAccuracyPrediction = load_approach('offmar-accuracy').OffMARAccuracyPrediction
from metalearner import load_approach_config
CNN_CONFIG = load_approach_config('offmar-accuracy').CNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)
config = CNN_CONFIG['rf']

offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)

# Run full OffMAR
results = offmar.run_full_offmar(dataset_name='mnist', timesteps=50)

# Save model for reuse
offmar.save_model('models/offmar_acc_cnn_mnist.pkl')
```

### Example 2: Production Deployment Pattern

```python
# === OFFLINE TRAINING (run once) ===
offmar_train = OffMARAccuracyPrediction(app, meta_learner_type='rf', theta_p=0.85)

# Phase 1: Collect data and train
phase_1 = offmar_train.phase_1_collect_data(
    dataset_name='mnist',
    timesteps=50  # More data = better meta-learner
)

# Save trained meta-learner
offmar_train.save_model('production/meta_learner_v1.pkl')

# === ONLINE DEPLOYMENT (run many times) ===
for deployment_run in range(10):  # Reuse 10 times!
    offmar_deploy = OffMARAccuracyPrediction(app, meta_learner_type='rf')
    offmar_deploy.load_model('production/meta_learner_v1.pkl')

    # Phase 2: Fast inference with reuse decisions
    phase_2 = offmar_deploy.phase_2_predict_and_reuse(
        dataset_name='mnist',
        timesteps=30
    )

    print(f"Run {deployment_run}: Reuse {phase_2['reuse_percentage']:.1f}%")
```

### Example 3: Segmentation Composition

```python
from metalearner import load_approach
OffMARAccuracyPrediction = load_approach('offmar-accuracy').OffMARAccuracyPrediction
from metalearner import load_approach_config
SEGMENTATION_CONFIG = load_approach_config('offmar-accuracy').SEGMENTATION_CONFIG
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication

# Initialize
app = SegmentationCompositionApplication(dataset_name='bsd500', random_seed=42)
config = SEGMENTATION_CONFIG['rf']

offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)

# Run
results = offmar.run_full_offmar(dataset_name='bsd500', timesteps=40)
```

---

## Meta-Learners

### k-Nearest Neighbors (kNN)

**Best for:**
- Small to medium knowledge repositories
- Fast predictions
- Non-parametric learning

**Parameters:**
```python
meta_learner_params = {
    'k': 5  # Number of neighbors (default: 5)
}
```

**Pros:**
- ✅ Simple and interpretable
- ✅ No training time (just stores data)
- ✅ Works well with local patterns

**Cons:**
- ❌ Slower prediction for large repositories
- ❌ Sensitive to irrelevant features

---

### Random Forest (RF)

**Best for:**
- Complex, non-linear relationships
- Robustness to noise
- Medium to large knowledge repositories

**Parameters:**
```python
meta_learner_params = {
    'n_estimators': 100,  # Number of trees (default: 100)
    'max_depth': None     # Maximum depth (default: None = unlimited)
}
```

**Pros:**
- ✅ Excellent generalization
- ✅ Handles complex interactions
- ✅ Built-in feature importance

**Cons:**
- ❌ Slower training than kNN
- ❌ Larger memory footprint

---

### XGBoost

**Best for:**
- Maximum predictive accuracy
- Large knowledge repositories
- Production deployments

**Parameters:**
```python
meta_learner_params = {
    'n_estimators': 100,    # Number of boosting rounds (default: 100)
    'max_depth': 6,         # Maximum tree depth (default: 6)
    'learning_rate': 0.1    # Learning rate (default: 0.1)
}
```

**Pros:**
- ✅ State-of-the-art performance
- ✅ Regularization built-in
- ✅ Handles missing values

**Cons:**
- ❌ Slowest training
- ❌ More hyperparameters to tune
- ❌ Requires xgboost installation

---

## Hyperparameters

### θp (Theta-p): Reuse Threshold

**Purpose**: Minimum predicted accuracy to reuse current design

**Default**: `0.85`

**Range**: `[0.70, 0.95]`

**Impact on Phase 2:**

| Value | Effect | Reuse Frequency | Design Algorithm Calls |
|-------|--------|----------------|------------------------|
| **Low** (0.70-0.80) | Easy to reuse | High (~70-80%) | Low (~20-30%) |
| **Medium** (0.85) | Balanced | Medium (~50-60%) | Medium (~40-50%) |
| **High** (0.90-0.95) | Hard to reuse | Low (~20-30%) | High (~70-80%) |

**Tuning Guide:**

```python
# Aggressive reuse (minimize design algorithm calls)
theta_p = 0.75

# Balanced (default)
theta_p = 0.85

# Conservative (prioritize quality over efficiency)
theta_p = 0.92
```

**Phase 1 Impact**: Also used for pruning knowledge repository

---

## Performance Expectations

### Computational Efficiency

**Phase 1:**
- Design algorithm calls: **100%** of timesteps
- Purpose: Build high-quality knowledge repository
- One-time cost

**Phase 2:**
- Design algorithm calls: **Variable** (depends on θp and data)
- Typical range: 30-60% of timesteps
- Can reuse meta-learner across multiple Phase 2 runs

**Example** (N=50 total, split 25/25):
- **Phase 1**: 25 calls (100% of Phase 1)
- **Phase 2**: ~10-15 calls (40-60% of Phase 2, depending on θp)
- **Total**: 35-40 calls (70-80% overall)

### Comparison with Other Approaches

| Approach | Phase 1 Calls | Phase 2 Calls | Total (N=50) | Reusable? |
|----------|--------------|---------------|--------------|-----------|
| **OffMAR Accuracy** | 25 (100%) | 10-15 (40-60%) | **35-40** | ✅ Yes |
| OnMAR Accuracy | 25 (100%) | ~10 (conditional) | **35** | ❌ No |
| OnMAR Design | 25 (100%) | 0 (0%) | **25** | ❌ No |
| OffMAR Design | 25 (100%) | 0 (0%) | **25** | ✅ Yes |

**Key Advantage**: Reusable meta-learner means Phase 1 cost is amortized across multiple deployments!

---

## Troubleshooting

### Problem 1: Knowledge Repository Empty After Pruning

**Symptoms:**
```
ValueError: Knowledge repository is empty after pruning. Lower theta_p threshold.
```

**Cause:** θp is too high for Phase 1 performance

**Solution:**
```python
# Check Phase 1 performance distribution first
phase_1 = offmar.phase_1_collect_data('mnist', timesteps=30)
performances = phase_1['performance_history']
print(f"Mean: {np.mean(performances):.3f}")
print(f"Min: {np.min(performances):.3f}")
print(f"Max: {np.max(performances):.3f}")

# Set θp based on performance statistics
theta_p = np.mean(performances) - 0.5 * np.std(performances)

# Re-initialize with appropriate θp
offmar = OffMARAccuracyPrediction(app, meta_learner_type='rf', theta_p=theta_p)
```

---

### Problem 2: Too Much Reuse in Phase 2 (Degrading Performance)

**Symptoms:** High reuse percentage but declining performance

**Cause:** θp too low, meta-learner over-optimistic

**Solution:**
```python
# Increase θp threshold
offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type='rf',
    theta_p=0.90  # Increase from default 0.85
)
```

---

### Problem 3: Too Few Reuses in Phase 2 (High Computational Cost)

**Symptoms:** Low reuse percentage, many design algorithm calls

**Cause:** θp too high, meta-learner too pessimistic

**Solution:**
```python
# Decrease θp threshold
offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type='rf',
    theta_p=0.80  # Decrease from default 0.85
)

# Or try different meta-learner
offmar = OffMARAccuracyPrediction(
    application=app,
    meta_learner_type='xgboost',  # May predict more accurately
    theta_p=0.85
)
```

---

### Problem 4: Poor Meta-Learner Predictions

**Symptoms:** Predicted accuracy very different from actual

**Possible Causes:**
1. **Insufficient Phase 1 data**
2. **Wrong meta-learner for problem**
3. **Poor pruning strategy**

**Solutions:**

```python
# Solution 1: More Phase 1 data
phase_1 = offmar.phase_1_collect_data(
    'mnist',
    timesteps=50  # Increase from 25
)

# Solution 2: Try different meta-learner
offmar = OffMARAccuracyPrediction(
    app,
    meta_learner_type='xgboost',  # Try instead of kNN
    theta_p=0.85
)

# Solution 3: Adjust pruning
offmar = OffMARAccuracyPrediction(
    app,
    meta_learner_type='rf',
    theta_p=0.80  # Keep more diverse samples
)
```

---

## Advanced Usage

### Multi-Dataset Deployment

```python
# Train on one dataset
offmar = OffMARAccuracyPrediction(app, meta_learner_type='rf', theta_p=0.85)
phase_1 = offmar.phase_1_collect_data('mnist', timesteps=50)
offmar.save_model('models/mnist_meta_learner.pkl')

# Deploy on similar dataset
offmar_new = OffMARAccuracyPrediction(app, meta_learner_type='rf')
offmar_new.load_model('models/mnist_meta_learner.pkl')

# Use on fashion-mnist (similar to mnist)
phase_2 = offmar_new.phase_2_predict_and_reuse('fashion-mnist', timesteps=30)
```

**Note**: Transfer learning works best with similar datasets/domains

---

### Analyzing Reuse Decisions

```python
results = offmar.run_full_offmar('mnist', timesteps=50)

# Phase 2 statistics
print(f"Total Phase 2 timesteps: {len(results['phase_2']['performance_history'])}")
print(f"Reuses: {results['phase_2']['num_reuses']}")
print(f"New designs: {results['phase_2']['num_new_designs']}")
print(f"Reuse percentage: {results['phase_2']['reuse_percentage']:.1f}%")

# Performance comparison
phase_1_best = results['phase_1']['best_performance']
phase_2_best = results['phase_2']['best_performance']
print(f"Phase 1 best: {phase_1_best:.4f}")
print(f"Phase 2 best: {phase_2_best:.4f}")
print(f"Improvement: {((phase_2_best - phase_1_best) / phase_1_best * 100):.1f}%")
```

---

## Comparison Summary

### OffMAR Accuracy-Prediction vs. Other Approaches

| Priority | Best Approach | Why |
|----------|---------------|-----|
| **Reproducibility** | OffMAR Accuracy | Fixed meta-learner, deterministic |
| **Reusability** | OffMAR Accuracy or OffMAR Design | Train once, deploy many times |
| **Flexibility** | OffMAR Accuracy | Binary reuse decision (not forced) |
| **Efficiency (single run)** | OnMAR Design | Only 50% design algorithm calls |
| **Adaptability** | OnMAR Accuracy | Continuous online learning |
| **Maximum Performance** | OnMAR Accuracy | Best results in thesis |

---

## References

- **OffMAR Design-Prediction**: `OffMAR/design-prediction/README.md`
- **OnMAR Accuracy-Prediction**: `OnMAR/accuracy-prediction/README.md`
- **OnMAR Design-Prediction**: `OnMAR/design-prediction/README.md`
- **Complete Comparison**: `COMPARISON_ALL_APPROACHES.md`

---

## Contact & Support

For issues or questions:
- Check existing documentation
- Review example scripts in `OffMAR/accuracy-prediction/examples/`
- Compare with other approaches in `COMPARISON_ALL_APPROACHES.md`
