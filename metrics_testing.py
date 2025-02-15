import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Create output directory if it does not exist
from pathlib import Path
__this_file__ = Path(__file__).resolve()
__output_dir__ = __this_file__.parent / "output"
__output_dir__.mkdir(parents=True, exist_ok=True)

# Import argparse to parse command line arguments
import argparse
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    random_state=42
)

# Define the model
model = GradientBoostingClassifier()

# Define the hyperparameters grid to search
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
    # 'max_depth': [3, 5, 7]
}

dict_r = {
    "n_estimators": [],
    "lr": [],
    'roc_auc': [],
    "pr_auc": [],
    'f1': [],
    "mcc": []
}

with open(__output_dir__ / f'metrics_results{extra_label}.txt', 'w') as f:
    f.write(f"Number of samples: {n_samples}\n")
    f.write(f"Number of features: {n_features}\n")
    f.write(f"Test size: {test_size}\n")
    f.write(f"Weights: {weights}\n")
    f.write("#" * 50)
    f.write("\n")
    
    for lr in param_grid['learning_rate']:
        for n_estimators in param_grid['n_estimators']:
            # Perform grid search with cross-validation
            grid_search = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_estimators)
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search

            # Predict probabilities
            y_scores = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)

            # Compute various metrics
            roc_auc = roc_auc_score(y_test, y_scores)
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            pr_auc = auc(recall, precision)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Save the metrics to a dictionary
            dict_r["n_estimators"].append(n_estimators)
            dict_r["lr"].append(lr)
            dict_r['roc_auc'].append(roc_auc)
            dict_r['f1'].append(f1)
            dict_r['pr_auc'].append(pr_auc)
            dict_r['mcc'].append(mcc)

            # Print the metrics
            print(f"Number of estimators: {n_estimators}")
            print(f"Learning rate: {lr}")
            print(f"ROC AUC: {roc_auc:.2f}")
            print(f"PR AUC: {pr_auc:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print(f"Matthews Correlation Coefficient: {mcc:.2f}")

            # Save the metrics to a .txt file
            f.write(f"Number of estimators: {n_estimators}\n")
            f.write(f"Learning rate: {lr}\n")
            f.write(f"ROC AUC: {roc_auc:.2f}\n")
            f.write(f"PR AUC: {pr_auc:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")
            f.write(f"Matthews Correlation Coefficient: {mcc:.2f}\n")
            f.write("\n")

            # Compute the confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Save the confusion matrix to a .txt file
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write("#" * 50)
            f.write("\n")

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df = pd.DataFrame(dict_r)
df.to_csv(__output_dir__ / f'metrics_results{extra_label}.csv', index=False)

# Plot the dataframe with learning rate as x-axis and other columns as lines
# plt.figure(figsize=(12, 8))
# for column in df.columns:
#     if column != 'lr':
#         plt.plot(df['lr'], df[column], label=column)

# plt.xlabel('Learning Rate')
# plt.ylabel('Metrics')
# plt.title('Metrics as a Function of Learning Rate')
# plt.legend()
# plt.grid(True)
# plt.savefig(__output_dir__ / f'metrics_vs_learning_rate{extra_label}.pdf')
# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each metric in the 3D plot
for metric in ['roc_auc', 'pr_auc', 'f1', 'mcc']:
    ax.plot_trisurf(df['lr'], df['n_estimators'], df[metric], label=metric, alpha=0.7)

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Number of Estimators')
ax.set_zlabel('Metric Value')
ax.set_title('Metrics as a Function of Learning Rate and Number of Estimators')
ax.legend()
plt.savefig(__output_dir__ / f'metrics_3d_plot{extra_label}.pdf')
plt.show()
