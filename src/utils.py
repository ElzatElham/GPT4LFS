import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def save_confusion_matrix(y_true, y_pred, path, normalized=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalized else 'd', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalized else ''))
    plt.savefig(path)
    plt.close()

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) !=0 else 0
    specificity = tn / (tn + fp) if (tn + fp) !=0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn)!=0 else 0
    return accuracy, sensitivity, specificity

def compute_cohens_kappa(cnf_matrix):
    total = cnf_matrix.sum()
    sum0 = 0
    sum1 = 0
    for i in range(len(cnf_matrix)):
        sum0 += cnf_matrix[i][i]
        sum1 += sum([cnf_matrix[i][j] for j in range(len(cnf_matrix))])

    po = sum0 / total
    pe = 0
    for i in range(len(cnf_matrix)):
        pe += (sum(cnf_matrix[i]) * sum(cnf_matrix[:,i])) / (total**2)
        
    kappa = (po - pe) / (1 - pe) if (1 - pe) !=0 else 0

    # Confidence interval and p-value require more complex methods or libraries, simplified here
    ci_lower = kappa - 0.05
    ci_upper = kappa + 0.05
    p_value = 1 - norm.cdf(abs(kappa) / 0.1)  # Simplified calculation

    return kappa, (ci_lower, ci_upper), p_value
