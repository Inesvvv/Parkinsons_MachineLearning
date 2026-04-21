import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from google.colab import drive

# ----------------------------
# 1. Load data
# ----------------------------
drive.mount('/content/drive')

df = pd.read_csv(
    "/content/drive/MyDrive/Machine_Learning_Project/ML_notebooks/pd_speech_features.csv",
    skiprows=1
)

# ----------------------------
# 2. Split variables
# ----------------------------
X = df.drop(columns=["class", "id"])
y = df["class"]
groups = df["id"]

# ----------------------------
# 3. Grouped train/test split
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()
train_groups = groups.iloc[train_idx]

# ----------------------------
# 4. Correlation filter
# ----------------------------
corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.96)]

X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

print("Features remaining after correlation filter:", X_train.shape[1])

# ----------------------------
# 5. Dummy baseline
# ----------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print("Dummy baseline accuracy:", round(dummy.score(X_test, y_test), 3))

# ----------------------------
# 6. Grouped CV setup
# ----------------------------
group_kfold = GroupKFold(n_splits=5)

# ----------------------------
# 7. Hyperparameter search space
# ----------------------------
hidden_layers_range = [
    (32,),
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (64, 32, 16)
]

activation_range = ["relu", "tanh"]
alpha_range = [1e-4, 1e-3, 1e-2, 1e-1]
learning_rate_range = [1e-4, 5e-4, 1e-3, 1e-2]

results = []

# ----------------------------
# 8. Manual search with grouped CV
# ----------------------------
for layers in hidden_layers_range:
    for activation in activation_range:
        for alpha in alpha_range:
            for lr in learning_rate_range:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation=activation,
                        alpha=alpha,
                        learning_rate_init=lr,
                        max_iter=1000,
                        random_state=42,
                        early_stopping=True,
                        n_iter_no_change=20
                    ))
                ])

                scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=group_kfold,
                    groups=train_groups,
                    scoring="roc_auc"
                )

                result = {
                    "hidden_layers": layers,
                    "activation": activation,
                    "alpha": alpha,
                    "learning_rate_init": lr,
                    "cv_auc_mean": scores.mean(),
                    "cv_auc_std": scores.std()
                }
                results.append(result)

                print(
                    f"layers={str(layers):<14} "
                    f"act={activation:<5} "
                    f"alpha={alpha:<8} "
                    f"lr={lr:<8} "
                    f"-> CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}"
                )

# ----------------------------
# 9. Collect results
# ----------------------------
results_df = pd.DataFrame(results).sort_values("cv_auc_mean", ascending=False)
print("\nTop 10 configurations:")
print(results_df.head(10))

best_config = results_df.iloc[0].to_dict()
print("\nBest config:")
print(best_config)

# ----------------------------
# 10. Retrain best model on full training set
# ----------------------------
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=best_config["hidden_layers"],
        activation=best_config["activation"],
        alpha=best_config["alpha"],
        learning_rate_init=best_config["learning_rate_init"],
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    ))
])

best_pipe.fit(X_train, y_train)

# ----------------------------
# 11. Evaluate on test set
# ----------------------------
proba = best_pipe.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\nTest set results at threshold 0.5")
print(classification_report(y_test, pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("Test ROC-AUC:", roc_auc_score(y_test, proba))

# ----------------------------
# 12. Threshold tuning
# ----------------------------
best_threshold = None
best_f1_class1 = -1

for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    pred_t = (proba >= t).astype(int)
    print(f"\nThreshold = {t}")
    print(classification_report(y_test, pred_t))

# ----------------------------
# 13. Optional: plot CV results
# ----------------------------
plt.figure(figsize=(8, 5))
plt.hist(results_df["cv_auc_mean"], bins=20)
plt.xlabel("Mean CV ROC-AUC")
plt.ylabel("Number of configurations")
plt.title("Distribution of tested configurations")
plt.show()