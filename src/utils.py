import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def get_accuracy(y, pred):
    y_label = np.array(y)
    y_pred = np.array(pred)
    y_label[np.array(y) >= 0.5] = 1
    y_label[np.array(y) < 0.5] = 0
    y_pred[np.array(pred) >= 0.5] = 1
    y_pred[np.array(pred) < 0.5] = 0

    micro_accuracy = roc_auc_score(y_label, y_pred, average="micro")
    macro_accuracy = roc_auc_score(y_label, y_pred, average="macro")
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    roc_auc = auc(fpr, tpr)
    precision_mean = precision_score(y_label, y_pred, average="micro")
    recall_mean = recall_score(y_label, y_pred, average="micro")
    f1_mean = f1_score(y_label, y_pred, average="micro")
    return micro_accuracy, macro_accuracy, precision_mean, recall_mean, f1_mean, roc_auc