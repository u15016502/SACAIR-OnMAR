# OnMAR Design Prediction

**Online Meta-learning for AutoML in Real-time** with direct design prediction.

This is a variant of OnMAR where the meta-learner **predicts designs directly** (like OffMAR) but learns **online continuously** (like OnMAR accuracy-prediction). This combines the benefits of online adaptation with direct design generation.

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

OnMAR Design Prediction is a **hybrid approach** that combines:

- **Online Learning** (from OnMAR-AccuracyPrediction): Meta-learner is continuously updated during execution
- **Design Prediction** (from OffMAR-DesignPrediction): Meta-learner directly predicts complete designs from meta-features

### When to Use This Approach

✅ **Use OnMAR Design Prediction when:**
- You want online adaptation to changing data
- You prefer direct design prediction over reuse decisions
- You have sufficient timesteps (N ≥ 40-50)
- You want the meta-learner to learn and improve during execution

❌ **Don't use this approach when:**
- You need offline training and deployment (use OffMAR)
- You prefer accuracy-based reuse decisions (use OnMAR-AccuracyPrediction)
- You have very few timesteps (N < 30)

---

## How It Works

### Algorithm Overview

```
For each timestep t:
    1. Extract meta-features from current state

    2. If t < θt (Phase 1):
          Run design algorithm (the GA) → get design

       Else (Phase 2):
          Use meta-learner to predict design from meta-features

    3. Apply the design for this timestep → get performance
       (aa.exec: one more epoch of the network already being trained)

    4. Add (meta-features, design, performance) to the knowledge repository

    5. Re-train the meta-learner. On the FIRST training only - at θt, half
       way through the run - prune entries with performance < θp.

    6. Continue to next timestep
```

### When pruning happens

Pruning is a single event, immediately before the meta-learner is first
trained, not something applied on every update.

This matters. θp defaults to 0.85, and for most of a run performance sits
below it. Pruning on every update would therefore discard each new entry as
soon as it arrived, keep the repository empty, and leave the meta-learner
untrained - so Phase 2 would emit random designs for the whole run while
appearing to work.

If nothing in the repository clears θp when pruning does happen, the best
entries are kept instead of emptying it, and the run reports that it did so.

### Two Phases

#### Phase 1 (Timesteps 1 to θt)
- **Always** runs the design algorithm
- Builds initial knowledge repository
- Collects diverse (meta-features, design, performance) samples
- Default: θt = N/2 (half of total timesteps)

#### Phase 2 (Timesteps θt+1 to N)
- **Always** uses meta-learner to predict designs
- No design algorithm calls in Phase 2
- Continuously re-trains meta-learner with new data
- Knowledge repository grows; pruning already happened once, at θt

---

## Key Differences

### vs. OnMAR Accuracy-Prediction

| Aspect | OnMAR Design-Prediction | OnMAR Accuracy-Prediction |
|--------|------------------------|---------------------------|
| **Meta-learner output** | Complete design | Scalar accuracy |
| **Decision in Phase 2** | Always predict design | Reuse OR create (conditional) |
| **Flexibility** | Lower (must use predicted design) | Higher (binary decision) |
| **Design algorithm calls** | Only in Phase 1 | Phase 1 + conditional in Phase 2 |

### vs. OffMAR Design-Prediction

| Aspect | OnMAR Design-Prediction | OffMAR Design-Prediction |
|--------|------------------------|--------------------------|
| **Learning type** | Online (continuous) | Offline (batch) |
| **Meta-learner updates** | Every timestep | Once (end of Phase 1) |
| **Adaptability** | High (adapts to changes) | Low (fixed after Phase 1) |
| **Knowledge repository** | Continuously grows/pruned | Fixed after Phase 1 |
| **Phases** | 1 (two modes) | 2 (separate) |

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
OnMARDesignPrediction = load_approach('onmar-design').OnMARDesignPrediction
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize application
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Initialize OnMAR Design Prediction
onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type='knn',  # or 'rf', 'xgboost'
    theta_t=None,  # Default: N/2
    theta_p=0.85,  # Pruning threshold
    meta_learner_params={'k': 5}
)

# Run OnMAR
results = onmar.run_onmar(
    dataset_name='mnist',
    timesteps=50
)

# Access results
print(f"Best performance: {results['best_performance']}")
print(f"Design predictions: {results['num_design_predictions']}")
print(f"Test performance: {results['test_results']['test_performance']}")
```

---

## Configuration

### Using Predefined Configurations

```python
from metalearner import load_approach_config
CNN_CONFIG = load_approach_config('onmar-design').CNN_CONFIG

# Get configuration for specific meta-learner
config = CNN_CONFIG['knn']  # or 'rf', 'xgboost'

onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_t=config.theta_t,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)
```

### Preset Strategies

```python
from metalearner import load_approach_config
config = load_approach_config('onmar-design')
CONSERVATIVE_CONFIG = config.CONSERVATIVE_CONFIG  # Higher θp (0.90)
AGGRESSIVE_CONFIG = config.AGGRESSIVE_CONFIG      # Lower θp (0.75)
BALANCED_CONFIG = config.BALANCED_CONFIG          # Standard θp (0.85)

onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type=BALANCED_CONFIG.meta_learner_type,
    theta_t=BALANCED_CONFIG.theta_t,
    theta_p=BALANCED_CONFIG.theta_p,
    meta_learner_params=BALANCED_CONFIG.meta_learner_params
)
```

### Custom Configuration

```python
onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type='rf',
    theta_t=30,  # Start using meta-learner at timestep 30
    theta_p=0.80,  # Keep designs with performance ≥ 0.80
    meta_learner_params={
        'n_estimators': 150,
        'max_depth': 8
    }
)
```

---

## Usage Examples

### Example 1: CNN Configuration

```python
from metalearner import load_approach
OnMARDesignPrediction = load_approach('onmar-design').OnMARDesignPrediction
from metalearner import load_approach_config
CNN_CONFIG = load_approach_config('onmar-design').CNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)
config = CNN_CONFIG['knn']

onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_t=config.theta_t,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)

# Run
results = onmar.run_onmar(dataset_name='mnist', timesteps=50)

# Save model
onmar.save_model('models/onmar_cnn_mnist.pkl')
```

### Example 2: Segmentation Composition

```python
from metalearner import load_approach
OnMARDesignPrediction = load_approach('onmar-design').OnMARDesignPrediction
from metalearner import load_approach_config
SEGMENTATION_CONFIG = load_approach_config('onmar-design').SEGMENTATION_CONFIG
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication

# Initialize
app = SegmentationCompositionApplication(dataset_name='bsd500', random_seed=42)
config = SEGMENTATION_CONFIG['rf']  # RF recommended for segmentation

onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_t=config.theta_t,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)

# Run
results = onmar.run_onmar(dataset_name='bsd500', timesteps=40)
```

### Example 3: Fuzzy ART Generation

```python
from metalearner import load_approach
OnMARDesignPrediction = load_approach('onmar-design').OnMARDesignPrediction
from metalearner import load_approach_config
FUZZYART_CONFIG = load_approach_config('onmar-design').FUZZYART_CONFIG
from applications.generation.fuzzyart.fuzzyart_application import FuzzyARTGenerationApplication

# Initialize
app = FuzzyARTGenerationApplication(dataset_name='enron', random_seed=42)
config = FUZZYART_CONFIG['knn']  # kNN recommended for Fuzzy ART

onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_t=config.theta_t,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)

# Run
results = onmar.run_onmar(dataset_name='enron', timesteps=60)
```

---

## Meta-Learners

### k-Nearest Neighbors (kNN)

**Best for:**
- Fast online updates (instant re-training)
- Non-parametric learning
- Small to medium design spaces

**Parameters:**
```python
meta_learner_params = {
    'k': 5  # Number of neighbors (default: 5)
}
```

**Pros:**
- ✅ Extremely fast to update
- ✅ No explicit training needed
- ✅ Works well with changing data

**Cons:**
- ❌ Slower prediction for large repositories
- ❌ Memory grows with repository size

---

### Random Forest (RF)

**Best for:**
- Complex design spaces
- Robustness to noise
- Feature importance analysis

**Parameters:**
```python
meta_learner_params = {
    'n_estimators': 100,  # Number of trees (default: 100)
    'max_depth': None     # Maximum depth (default: None = unlimited)
}
```

**Pros:**
- ✅ Handles complex interactions
- ✅ Robust to outliers
- ✅ Good generalization

**Cons:**
- ❌ Slower re-training each timestep
- ❌ Larger memory footprint

---

### XGBoost

**Best for:**
- Maximum predictive accuracy
- Large design spaces
- Gradient-based optimization

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
- ✅ Handles complex patterns
- ✅ Regularization built-in

**Cons:**
- ❌ Slowest re-training
- ❌ More hyperparameters to tune
- ❌ Requires xgboost installation

---

## Hyperparameters

### θt (Theta-t): Timestep Threshold

**Purpose**: When to start using meta-learner for design prediction

**Default**: `N/2` (half of total timesteps)

**Range**: `[N/4, 3N/4]`

**Tuning Guide:**

| Value | Effect | Use When |
|-------|--------|----------|
| **Small** (N/4) | Start predicting early | Fast convergence expected |
| **Medium** (N/2) | Balanced approach | **Default (recommended)** |
| **Large** (3N/4) | More training data | Complex design space |

**Example:**
```python
# For 50 timesteps
theta_t = None      # Auto: 25 (N/2)
theta_t = 12        # Manual: Start predicting at timestep 12
theta_t = 38        # Manual: Start predicting at timestep 38
```

---

### θp (Theta-p): Pruning Threshold

**Purpose**: Minimum performance to keep designs in knowledge repository

**Default**: `0.85`

**Range**: `[0.70, 0.95]`

**Tuning Guide:**

| Value | Effect | Repository Size | Use When |
|-------|--------|----------------|----------|
| **Low** (0.70-0.80) | Keep more designs | Larger | More diversity needed |
| **Medium** (0.85) | Balanced | Medium | **Default (recommended)** |
| **High** (0.90-0.95) | Keep only best | Smaller | Quality over quantity |

**Example:**
```python
theta_p = 0.70   # Aggressive: Keep more designs
theta_p = 0.85   # Balanced (default)
theta_p = 0.92   # Conservative: Only keep excellent designs
```

**Warning:** If θp is too high, repository may become empty after pruning!

---

## Performance Expectations

### Computational Efficiency

**Design algorithm calls:**
- Phase 1: `θt` calls (e.g., 25 for N=50, θt=25)
- Phase 2: `0` calls (always use meta-learner)
- **Total**: `θt` calls = `~50%` of timesteps

**Example** (N=50, θt=25):
- OnMAR Design-Prediction: **25 calls** (50%)
- OnMAR Accuracy-Prediction: **~35 calls** (70%, conditional reuse)
- OffMAR Phase 1: **50 calls** (100%)

---

### Accuracy vs. Speed Trade-off

| Approach | Speed | Adaptation | Direct Prediction |
|----------|-------|------------|-------------------|
| **OnMAR Design-Prediction** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| OnMAR Accuracy-Prediction | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| OffMAR Design-Prediction | ⭐⭐⭐⭐⭐ (Phase 2) | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting

### Problem 1: Knowledge Repository Empty After Pruning

**Symptoms:**
```
Warning: Knowledge repository empty after pruning. Skipping training.
```

**Cause:** θp threshold is too high, all designs are pruned

**Solution:**
```python
# Lower theta_p threshold
onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type='knn',
    theta_p=0.75  # Lower from default 0.85
)
```

---

### Problem 2: Meta-Learner Not Improving Performance

**Symptoms:** Predicted designs perform worse than Phase 1 designs

**Possible Causes:**
1. **Insufficient Phase 1 data**: θt too small
2. **Poor pruning**: θp too low, keeping bad designs
3. **Wrong meta-learner**: Not suited for design space

**Solutions:**

```python
# Solution 1: Increase Phase 1 length
onmar = OnMARDesignPrediction(
    application=app,
    theta_t=int(0.6 * timesteps)  # Use 60% instead of 50%
)

# Solution 2: Increase pruning threshold
onmar = OnMARDesignPrediction(
    application=app,
    theta_p=0.90  # Keep only best designs
)

# Solution 3: Try different meta-learner
onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type='xgboost'  # Try XGBoost instead of kNN
)
```

---

### Problem 3: Slow Re-training Each Timestep

**Symptoms:** Long runtime in Phase 2

**Cause:** Meta-learner (RF/XGBoost) takes time to re-train

**Solution:**

```python
# Use kNN for instant updates
onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type='knn',  # Fast re-training
    meta_learner_params={'k': 5}
)

# Or reduce RF complexity
onmar = OnMARDesignPrediction(
    application=app,
    meta_learner_type='rf',
    meta_learner_params={
        'n_estimators': 50,   # Reduce from 100
        'max_depth': 10       # Limit depth
    }
)
```

---

### Problem 4: Predicted Designs Are Invalid

**Symptoms:** Designs violate design space constraints

**Cause:** Decoder might produce out-of-range values

**Solution:**

The implementation automatically clips values to valid ranges. If issues persist:

1. Check design space definition
2. Verify encoding/decoding consistency
3. Use more training data (increase θt)

---

## Advanced Usage

### Saving and Loading Models

```python
# Save trained meta-learner
onmar.save_model('models/onmar_design_prediction.pkl')

# Load in new session
onmar_new = OnMARDesignPrediction(
    application=app,
    meta_learner_type='knn'
)
onmar_new.load_model('models/onmar_design_prediction.pkl')

# Use loaded model for prediction
predicted_design = onmar_new._predict_design(meta_features)
```

---

### Accessing Results

```python
results = onmar.run_onmar(dataset_name='mnist', timesteps=50)

# Performance metrics
best_performance = results['best_performance']
final_performance = results['final_performance']
test_performance = results['test_results']['test_performance']

# Design history
best_design = results['best_design']
designs_history = results['designs_history']
performance_history = results['performance_history']

# Efficiency metrics
design_calls = results['num_design_algorithm_calls']
predictions = results['num_design_predictions']
repo_size = results['knowledge_repository_size']

# Percentages
design_call_pct = results['design_algorithm_percentage']
prediction_pct = results['design_prediction_percentage']
```

---

## Comparison Summary

### OnMAR Design-Prediction vs. Other Approaches

| Feature | OnMAR Design | OnMAR Accuracy | OffMAR Design |
|---------|--------------|----------------|---------------|
| **Learning** | Online | Online | Offline |
| **Prediction** | Design | Accuracy | Design |
| **Adaptability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Efficiency** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (Phase 2) |
| **Simplicity** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Reproducibility** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## References

- **Chapter 8**: Meta-learning for Real-time Design (Thesis)
- **OnMAR Accuracy-Prediction**: `OnMAR/accuracy-prediction/README.md`
- **OffMAR Design-Prediction**: `OffMAR/design-prediction/README.md`
- **Comparison Guide**: `COMPARISON_ONMAR_OFFMAR.md`

---

## Contact & Support

For issues or questions:
- Check existing documentation
- Review example scripts in `OnMAR/design-prediction/examples/`
- Compare with other approaches in `COMPARISON_ONMAR_OFFMAR.md`
