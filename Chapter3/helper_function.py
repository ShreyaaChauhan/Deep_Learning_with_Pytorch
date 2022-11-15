import numpy as np


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def split_cm(cm):
    actual_negative = cm[0]
    tn = actual_negative[0]
    fp = actual_negative[1]
    actual_positive = cm[1]
    fn = actual_positive[0]
    tp = actual_positive[1]
    return tn, fp, fn, tp


def tpr_fpr(cm):
    tn, fp, fn, tp = split_cm(cm)

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return tpr, fpr


def precision_recall(cm):
    tn, fp, fn, tp = split_cm(cm)

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    return precision, recall
