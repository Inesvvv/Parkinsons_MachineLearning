# Parkinson's Disease Detection from Speech Features

Binary classification of Parkinson's disease from voice recordings. The models are neural networks with **patient-grouped validation**, so no subject ever appears in both training and evaluation data, and their hyperparameters are tuned with **Bayesian optimization (Optuna TPE)**.

## Data

`pd_speech_features.csv` contains the *Parkinson's Disease Classification* dataset (Sakar et al., 2019, UCI Machine Learning Repository):
- 756 recordings from **252 subjects**, with 3 recordings per subject
- 753 acoustic features: baseline features, intensity, formants, vocal fold, MFCC, wavelet and TQWT
- Class balance: 564 with Parkinson's, 192 healthy controls. A majority-class baseline already reaches 72.5% accuracy, so **ROC-AUC** is the main metric.

## Method

1. **Leakage-safe splits.** A grouped 80/20 train/test split and a 5-fold `GroupKFold` both use subject ID as the group, so the 3 recordings of one person always stay together.
2. **Feature reduction.** Highly correlated features are dropped (753 → 539), then the top-*k* features are selected by mutual information. Every step is fit only on training folds.
3. **Feature-count study.** Grouped CV is run for *k* ∈ {50, 100, 200, 300, all}. The best result is with **50 features (CV ROC-AUC 0.837)**, more than with all 539 (0.809).
4. **PCA experiment.** The baseline network is compared with and without PCA, which retains 95% of variance and is fit inside each fold.
5. **Hyperparameter search.** Optuna runs 50 trials with a TPE sampler and median pruning. It searches over depth (1–4 layers), width, activation, dropout, L2, learning rate, batch size and batch normalization, and optimizes mean grouped-CV ROC-AUC. Class weights correct for the class imbalance.
6. **Threshold analysis.** Sensitivity, specificity and F1 are measured across decision thresholds, which matters for a screening setting.

## Results (held-out test subjects)

| Model | Test ROC-AUC |
|---|---|
| Baseline network (1 hidden layer, 32 units) | 0.811 |
| **Optuna-tuned network (4 hidden layers)** | **0.879** |

The tuned model's best grouped-CV ROC-AUC was about 0.89.

This repo is my individual extension of the neural-network part of a group project. The group comparison against tree ensembles, logistic regression and k-NN is in [ML_Project](https://github.com/Inesvvv/ML_Project).

## Files

| File | Contents |
|---|---|
| `hyperparameter_tuning.ipynb` | Main pipeline: preprocessing, feature-count and PCA studies, baseline, Optuna search, test evaluation |
| `neural_net_keras.ipynb` | TensorFlow/Keras network with threshold analysis |
| `neural_net.ipynb` | scikit-learn `MLPClassifier` grid search, the first iteration |
| `neural_net_colab.py` | Script version of `neural_net.ipynb` for Google Colab |

## Running

```bash
pip install -r requirements.txt
jupyter notebook hyperparameter_tuning.ipynb
```

The full 50-trial search takes about an hour on a laptop CPU.

## Reference

Sakar, C. O. et al. (2019). *A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform.* Applied Soft Computing, 74, 255–263.

**Tools:** Python · TensorFlow/Keras · scikit-learn · Optuna · pandas · NumPy
