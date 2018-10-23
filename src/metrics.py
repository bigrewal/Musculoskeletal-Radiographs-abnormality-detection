from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, f1_score


def print_all_metrics(y_true, y_pred):
    print("roc_auc_score: ", roc_auc_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("Sensitivity: ", get_sensitivity(tp, fn))
    print("Specificity: ", get_specificity(tn, fp))
    print("Cohen-Cappa-Score: ", cohen_kappa_score(y_true, y_pred))
    print("F1 Score: ", f1_score(y_true, y_pred))


def get_sensitivity(tp, fn):
    return tp / (tp + fn)


def get_specificity(tn, fp):
    return tn / (tn + fp)
