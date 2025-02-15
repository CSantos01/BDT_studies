import time
import numpy as np
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
# from fastbdt import FastBDT
from interpret.glassbox import ExplainableBoostingClassifier

# Create output directory if it does not exist
from pathlib import Path
__this_file__ = Path(__file__).resolve()
__output_dir__ = __this_file__.parent / "output"
__output_dir__.mkdir(parents=True, exist_ok=True)

# Import argparse to parse command line arguments
import argparse
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
argparser = argparse.ArgumentParser(description="Script to compute various metrics with respect to the weights of the classes")
argparser.add_argument("--weights", type=float, nargs=2, default=[0.83, 0.17], help="Weights of the classes")
argparser.add_argument("--n_sample", type=int, default=6666, help="Number of samples")
argparser.add_argument("--n_features", type=int, default=20, help="Number of features")
argparser.add_argument("--test_size", type=float, default=0.3, help="Test size")
argparser.add_argument("--extra_label", type=str, default="", help="Extra label to add to the output files")
args = argparser.parse_args()

# Define arguments
weights = args.weights
n_samples = args.n_sample
n_features = args.n_features
test_size = args.test_size
extra_label = f"_{args.extra_label}" if args.extra_label else ""

# Generate a binary classification dataset
X, y = make_classification(
    n_samples=n_samples, 
    n_features=n_features, 
    weights=weights, 
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate classifier
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = clf.predict(X_test)
    
    metrics = {
        "training_time": training_time,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
        "pr_auc": average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])
    }
    return metrics

# Define classifiers and their hyperparameter search spaces
def objective(trial, classifier_name):
    if classifier_name == "GradientBoostingClassifier":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 5)
        }
        clf = GradientBoostingClassifier(**params)
    elif classifier_name == "LightGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 50)
        }
        clf = lgb.LGBMClassifier(**params)
    elif classifier_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 5)
        }
        clf = xgb.XGBClassifier(**params)
    elif classifier_name == "EBM":
        params = {
            'max_bins': trial.suggest_int('max_bins', 128, 256),
            'max_interaction_bins': trial.suggest_int('max_interaction_bins', 16, 32)
        }
        clf = ExplainableBoostingClassifier(**params)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Optimize hyperparameters and compare classifiers
results = {}
classifiers = ["GradientBoostingClassifier", "LightGBM", "XGBoost", "EBM"]
for classifier_name in classifiers:
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, classifier_name), n_trials=20)
    best_params = study.best_params
    
    if classifier_name == "GradientBoostingClassifier":
        best_clf = GradientBoostingClassifier(**best_params)
    elif classifier_name == "LightGBM":
        best_clf = lgb.LGBMClassifier(**best_params)
    elif classifier_name == "XGBoost":
        best_clf = xgb.XGBClassifier(**best_params)
    elif classifier_name == "EBM":
        best_clf = ExplainableBoostingClassifier(**best_params)
    
    results[classifier_name] = evaluate_classifier(best_clf, X_train, X_test, y_train, y_test)

# Write results to a .txt file
output_file = __output_dir__ / f"results{extra_label}.txt"
with open(output_file, "w") as f:
    f.write(f"Number of samples: {n_samples}\n")
    f.write(f"Number of features: {n_features}\n")
    f.write(f"Test size: {test_size}\n")
    f.write(f"Weights: {weights}\n")
    f.write("#" * 50)
    f.write("\n")
    
    for name, metrics in results.items():
        f.write(f"Classifier: {name}\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("#" * 50)
        f.write("\n")