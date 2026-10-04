# OnMAR: Online Meta-learning for AutoML in Real-time - Accuracy Prediction

This implementation follows Chapter 8 of the thesis, specifically the OnMAR approach where the meta-learner predicts accuracy to decide whether to reuse designs.

## Overview

**OnMAR (Online Meta-learning for AutoML in Real-time)** is a single-phase online meta-learning approach that continuously learns and adapts during execution.

### How OnMAR Works

1. **Phase 1 (t < θt)**: Build knowledge repository
   - Run design algorithm for each timestep
   - Extract meta-features and designs
   - Measure actual performance
   - Add to knowledge repository
   - Train meta-learner incrementally

2. **Phase 2 (t ≥ θt)**: Use meta-learner for decisions
   - Extract current meta-features
   - Use meta-learner to predict performance of current design
   - **If predicted performance ≥ θp**: Reuse current design (save computation!)
   - **If predicted performance < θp**: Run design algorithm to create new design
   - Update knowledge repository and re-train meta-learner

### Key Innovation

Unlike OffMAR which predicts **designs**, OnMAR predicts **accuracy** and makes a binary decision:
- ✅ **Reuse design** (if predicted accuracy is good enough)
- 🔄 **Create new design** (if predicted accuracy is too low)

This makes OnMAR more flexible and often more performant than OffMAR.

## Key Differences: OnMAR vs OffMAR

| Aspect | OnMAR | OffMAR |
|--------|-------|---------|
| **Training** | **Online** (during execution) | Offline (before deployment) |
| **Phases** | **Single phase** (two modes) | Two separate phases |
| **Meta-learner predicts** | **Accuracy** | Design |
| **Decision** | **Reuse or create new** | Direct design prediction |
| **Flexibility** | **High** (adapts continuously) | Lower (fixed after Phase 1) |
| **Computational cost** | **Lower** (fewer design calls) | High in Phase 1, low in Phase 2 |
| **Performance** | **Often best** (Table 8.2) | Good for some applications |
| **Knowledge** | **Continuously updated** | Fixed after Phase 1 |

## Algorithm (from Chapter 8, Algorithm 9)

```
1: t ← 0; kr ← {}; θt ← N/2; θp ← 0.85
2: while t ≤ N do
3:     meta_features = aa.calculate_features()
4:     if t > θt then
5:         p = ml.predict(meta_features, c)
6:         if p < θp then
7:             c ← design_algorithm(dataset, t)
8:         end if
9:     else
10:        c ← design_algorithm(dataset, t)
11:    end if
12:    p ← aa.exec(c, dataset, t)
13:    kr ← meta_features, c, p
14:    ml.train(kr)
15:    t + +
16: end while
```

## Directory Structure

```
OnMAR/accuracy-prediction/
├── onmar.py               # Main OnMAR implementation
├── config.py              # Configuration and hyperparameters
├── README.md              # This file
├── examples/              # Example scripts
│   ├── run_cnn.py        # CNN configuration example
│   ├── run_segmentation.py  # Segmentation composition example
│   └── run_fuzzyart.py   # Fuzzy ART generation example
├── models/                # Saved trained meta-learners
└── results/               # Performance plots and logs
```

## Supported Applications

1. **CNN Configuration** (`applications/configuration/cnn`)
   - Automated hyperparameter optimization
   - Meta-learner predicts validation accuracy

2. **Image Segmentation Composition** (`applications/composition/segmentation`)
   - Automated algorithm composition
   - Meta-learner predicts IoU score

3. **Fuzzy ART Choice Function Generation** (`applications/generation/fuzzyart`)
   - Automated function generation
   - Meta-learner predicts clustering fitness

## Meta-Learners

Three meta-learners are implemented (as tested in Chapter 8):

1. **kNN (k-Nearest Neighbors)** - Best for OnMAR according to Table 8.2
   - Instance-based learning
   - Fast updates (just add to repository)
   - Good generalization
   - Default k=5

2. **RF (Random Forest)**
   - Ensemble of decision trees
   - Robust to noise
   - Good feature importance
   - Default: 100 trees

3. **XGBoost (Extreme Gradient Boosting)**
   - Advanced gradient boosting
   - Often competitive with kNN
   - Higher computational cost
   - Default: 100 estimators, depth=6

## Usage

### Basic Usage

```python
from metalearner import load_approach
OnMARAccuracyPrediction = load_approach('onmar-accuracy').OnMARAccuracyPrediction
from metalearner import load_approach_config
KNN_CONFIG = load_approach_config('onmar-accuracy').KNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize application
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Initialize OnMAR
onmar = OnMARAccuracyPrediction(
    application=app,
    meta_learner_type='knn',
    theta_t=None,  # N/2 (set automatically)
    theta_p=0.85,
    meta_learner_params={'k': 5}
)

# Run OnMAR
results = onmar.run_onmar(
    dataset_name='mnist',
    timesteps=50,
    initial_design=None
)

# Results include:
# - performance_history: accuracy at each timestep
# - num_design_algorithm_calls: how many times design algorithm ran
# - num_design_reuses: how many times design was reused
# - best_performance: highest accuracy achieved
```

### Using Configuration Presets

```python
from metalearner import load_approach_config
CNN_CONFIG = load_approach_config('onmar-accuracy').CNN_CONFIG

config = CNN_CONFIG['knn']  # Get kNN config for CNN

onmar = OnMARAccuracyPrediction(
    application=app,
    meta_learner_type=config.meta_learner_type,
    theta_t=config.theta_t,
    theta_p=config.theta_p,
    meta_learner_params=config.meta_learner_params
)
```

### Save and Load Models

```python
# Save after training
onmar.save_model('models/cnn_mnist_knn.pkl')

# Load for continued use
onmar.load_model('models/cnn_mnist_knn.pkl')

# Continue from where you left off
results = onmar.run_onmar(dataset_name='mnist', timesteps=50)
```

## Configuration

### Hyperparameters

**θt (Timestep Threshold)**
- Default: N/2 (from Algorithm 9 in thesis)
- When to start using meta-learner
- Before θt: Build knowledge repository
- After θt: Use meta-learner for decisions
- **Recommendation**: Keep at N/2 for balanced approach

**θp (Performance Threshold)**
- Default: 0.85 (from Algorithm 9 in thesis)
- Minimum predicted performance to reuse design
- Range: [0.0, 1.0]
- Lower values → more reuse → fewer design algorithm calls
- Higher values → less reuse → more design algorithm calls
- **Tuning guide**:
  - 0.75-0.80: Aggressive (maximize reuse)
  - 0.85: Balanced (recommended)
  - 0.90-0.95: Conservative (maximize quality)

**Meta-learner Parameters**

*kNN:*
- `k`: Number of neighbors (default: 5)
- Smaller k → more sensitive to noise
- Larger k → smoother predictions

*Random Forest:*
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)

*XGBoost:*
- `n_estimators`: Number of boosting rounds (default: 100)
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Learning rate (default: 0.1)

### Preset Configurations

```python
from metalearner import load_approach_config
config = load_approach_config('onmar-accuracy')
CONSERVATIVE_CONFIG = config.CONSERVATIVE_CONFIG
BALANCED_CONFIG = config.BALANCED_CONFIG
AGGRESSIVE_CONFIG = config.AGGRESSIVE_CONFIG

# Use conservative approach
onmar = OnMARAccuracyPrediction(
    application=app,
    meta_learner_type=CONSERVATIVE_CONFIG.meta_learner_type,
    theta_t=CONSERVATIVE_CONFIG.theta_t,
    theta_p=CONSERVATIVE_CONFIG.theta_p,
    meta_learner_params=CONSERVATIVE_CONFIG.meta_learner_params
)
```

## Performance Expectations

Based on Chapter 8 results (Table 8.2):

### CNN Configuration
- **kNN**: Best performer (rank 1.6 normalized)
- **RF**: Second best (rank 2.0 normalized)
- **XGBoost**: Third (rank 3.0 normalized)
- **Efficiency**: 40-60% design reuse in Phase 2

### Segmentation Composition
- **kNN**: Best performer (rank 1.0 normalized)
- **RF**: Second (rank 2.0 normalized)
- **XGBoost**: Third (rank 3.0 normalized)
- **Efficiency**: 30-50% design reuse in Phase 2

### Fuzzy ART Generation
- **kNN/XGBoost**: Tied for best (rank 1.0-2.0)
- **RF**: Close second (rank 2.0)
- **Efficiency**: 50-70% design reuse in Phase 2

### General Observations
- ✅ **OnMAR generally outperforms OffMAR** (except for some composition datasets)
- ✅ **kNN is often the best meta-learner** for OnMAR
- ✅ **Significant computational savings** vs. running design algorithm every timestep
- ✅ **Adapts to changing data** (online learning)

## Examples

### Example 1: Basic CNN Configuration

```bash
python OnMAR/accuracy-prediction/examples/run_cnn.py
```

### Example 2: Compare All Meta-Learners

```python
from examples.run_cnn import compare_meta_learners

results = compare_meta_learners(
    dataset_name='mnist',
    timesteps=50
)

# Prints comparison table:
# Meta-Learner  Best Perf   Final Perf  Test Perf   Reuse %    Runtime (s)
# KNN           0.9234      0.9189      0.9156      45.0       123.45
# RF            0.9201      0.9167      0.9134      42.0       156.78
# XGBOOST       0.9189      0.9145      0.9123      40.0       189.12
```

### Example 3: Custom Configuration

```python
from metalearner import load_approach_config
OnMARConfig = load_approach_config('onmar-accuracy').OnMARConfig

# Very aggressive: maximize reuse
aggressive = OnMARConfig(
    meta_learner_type='knn',
    theta_t=20,  # Start early (instead of N/2=25)
    theta_p=0.70,  # Low threshold
    meta_learner_params={'k': 3}  # Fewer neighbors
)

onmar = OnMARAccuracyPrediction(
    application=app,
    meta_learner_type=aggressive.meta_learner_type,
    theta_t=aggressive.theta_t,
    theta_p=aggressive.theta_p,
    meta_learner_params=aggressive.meta_learner_params
)
```

## Advantages of OnMAR

1. **Continuous Learning**: Adapts to data distribution changes
2. **Computational Efficiency**: Reuses designs when appropriate
3. **Single Phase**: No need for separate offline training
4. **Flexible**: Can adjust θp dynamically based on performance
5. **Best Performance**: Often ranks #1 in empirical comparisons (Table 8.2)

## Disadvantages of OnMAR

1. **Slower Start**: Needs θt timesteps to build knowledge repository
2. **Memory**: Grows with knowledge repository (can prune old entries)
3. **Computational Overhead**: Re-trains meta-learner each timestep (after θt)
4. **No Transfer**: Each run starts from scratch (unlike OffMAR which can reuse Phase 1)

## When to Use OnMAR vs OffMAR

**Use OnMAR when:**
- ✅ You have enough timesteps (N ≥ 40-60)
- ✅ Data distribution may change over time
- ✅ You want best possible performance
- ✅ You're running on a single dataset
- ✅ You value adaptability over reproducibility

**Use OffMAR when:**
- ✅ You have limited timesteps (N < 40)
- ✅ Data distribution is stable
- ✅ You want to reuse meta-learner across multiple runs
- ✅ You need reproducible results
- ✅ You can afford expensive Phase 1

## Troubleshooting

### Issue: Poor performance in Phase 2

**Possible causes:**
- θp too high (meta-learner never predicts good enough accuracy)
- θp too low (reusing bad designs)
- θt too small (insufficient training data)

**Solutions:**
- Tune θp based on Phase 1 performance distribution
- Increase θt to N*2/3 for more training data
- Try different meta-learner (kNN often works best)

### Issue: Too many design algorithm calls

**Cause:** θp too high

**Solution:**
- Lower θp (try 0.80 or 0.75)
- Check if meta-learner is well-calibrated
- Inspect predicted vs actual performance correlation

### Issue: Too many design reuses (performance degrading)

**Cause:** θp too low or poor meta-learner calibration

**Solution:**
- Increase θp (try 0.90)
- Ensure meta-features are informative
- Try more expressive meta-learner (RF or XGBoost instead of kNN)

### Issue: Memory usage growing

**Cause:** Knowledge repository growing without bounds

**Solution:**
```python
# Implement repository pruning in _update_knowledge_repository
max_repository_size = 1000
if len(self.knowledge_repository) > max_repository_size:
    # Keep only recent entries
    self.knowledge_repository = self.knowledge_repository[-max_repository_size:]
    # Or keep only best performing entries
    # self.knowledge_repository = sorted(...)[-max_repository_size:]
```

## Meta-Features

OnMAR uses the same comprehensive meta-features as OffMAR:

### Application-Agnostic
- Dataset properties, class imbalance
- ELA features (y-distribution, meta-model, dispersion, information content, NBC)

### CNN Configuration
- Confusion matrix metrics, performance metrics
- Weight distances, loss metrics

### Segmentation Composition
- LCE/GCE, ROC/AUC, Dice coefficient
- Distance metrics, distribution features

### Fuzzy ART Generation
- Classification metrics, weight analysis
- Clustering quality metrics

See [../applications/*/meta_features.py](../applications/configuration/cnn/meta_features.py) for full details.

## Advanced Usage

### Dynamic θp Adjustment

```python
class AdaptiveOnMAR(OnMARAccuracyPrediction):
    def run_onmar(self, ...):
        results = super().run_onmar(...)

        # Adjust θp based on performance in Phase 1
        phase1_perfs = results['performance_history'][:self.theta_t]
        mean_perf = np.mean(phase1_perfs)
        std_perf = np.std(phase1_perfs)

        # Set θp to mean - 0.5*std (adaptive)
        self.theta_p = mean_perf - 0.5 * std_perf

        return results
```

### Ensemble Meta-Learner

```python
# Train multiple meta-learners and ensemble predictions
predicted_knn = knn_model.predict(x)
predicted_rf = rf_model.predict(x)
predicted_xgb = xgb_model.predict(x)

# Average prediction
predicted_performance = (predicted_knn + predicted_rf + predicted_xgb) / 3
```

## References

- Chapter 8: Meta-learning for Real-time Design (Thesis)
- Section 8.2.1: OnMAR approach description
- Algorithm 9: OnMAR pseudo-code
- Table 8.2: Performance rankings (OnMAR often ranks #1)
- Figure 8.5: Example of effective vs ineffective meta-learner behavior

## Citation

If you use this implementation in your research, please cite the thesis:

```bibtex
@phdthesis{gerber2024automl,
  title={Meta-learning for Real-time AutoML},
  author={Gerber, Mia},
  year={2024},
  school={...},
  chapter={8}
}
```

## Future Improvements

1. **Warm start**: Initialize with OffMAR Phase 1 knowledge
2. **Multi-fidelity**: Use cheap approximations for predictions
3. **Bayesian optimization**: Replace meta-learner with GP
4. **Active learning**: Strategically choose when to run design algorithm
5. **Transfer learning**: Reuse knowledge across datasets
6. **Confidence-based**: Use prediction uncertainty for decisions
