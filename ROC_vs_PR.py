import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef

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

# Generate a binary classification dataset with 5 times more background than signal
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
    random_state=42,
)

# Train a Gradient Boosting Classifier model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Compute confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure()
plt.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.savefig(__output_dir__ / f'confusion_matrix{extra_label}.png')

# Compute ROC curve and ROC AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)
print(f"ROC AUC: {roc_auc:.2f}")

# Compute Precision-Recall curve and PR AUC
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.2f}")

# Plot ROC curve
plt.figure()
plt.plot(tpr, 1-fpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([1, 0], [0, 1], color='gray', lw=2, linestyle='--', label='Random choice')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Signal efficiency')
plt.ylabel('Background rejection')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(__output_dir__ / f'roc_curve{extra_label}.png')

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='green', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.plot([0, 1], [0.5, 0.5], color='gray', lw=2, linestyle='--', label='Random choice')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.savefig(__output_dir__ / f'pr_curve{extra_label}.png')

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.2f}")

# Compute Matthews correlation coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews correlation coefficient (MCC): {mcc:.2f}")

# Saving the information to a text file
with open(__output_dir__ / f'metrics{extra_label}.txt', 'w') as f:
    f.write(f"ROC AUC: {roc_auc:.2f}\n")
    f.write(f"PR AUC: {pr_auc:.2f}\n")
    f.write(f"F1-score: {f1:.2f}\n")
    f.write(f"Matthews correlation coefficient (MCC): {mcc:.2f}\n")
    f.write(f"Confusion matrix:\n{cm}")
f.close()