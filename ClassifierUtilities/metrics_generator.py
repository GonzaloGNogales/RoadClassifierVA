import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


def f1(Y_test, predicted, avg):
    return '{0:0.2f}'.format(f1_score(np.int32(Y_test), np.array(predicted), average=avg) * 100)


def set_up_metrics_directory():
    # Prepare the metrics results directory
    # Check if metrics folder already exists, if not create it
    if not os.path.isdir('metrics/'):
        os.mkdir('metrics/')


def save_training_metrics_graphic(classes, pred, clf):
    x = set(list(np.int32(classes)))  # classes for the x axis
    y_train_counter = dict.fromkeys(range(len(x)), 0)
    y_pred_counter = dict.fromkeys(range(len(x)), 0)

    pred = np.array(pred).reshape(-1)  # Normalize predictions array to 1D

    for i in classes:
        y_train_counter[int(i)] += 1
    for i in pred:
        y_pred_counter[int(i)] += 1

    x = np.array(list(x))
    y_train_counter = np.array(list(y_train_counter.values()))
    y_pred_counter = np.array(list(y_pred_counter.values()))

    accuracy_comparison = plt.figure(figsize=(14, 7))
    plt.plot(x, y_train_counter, 'o-', color='blue', label='Target Accuracy')
    plt.plot(x, y_pred_counter, 'o-', color='orange', label='Classifier Accuracy')
    plt.title(clf + ' detections/class in Training')
    plt.xlabel("Class")
    plt.ylabel("Detections")
    plt.legend()
    accuracy_comparison.savefig('metrics/' + clf + '_classifier_training_performance.png')
    print("Training metrics saved on classifier_training_performance.png")


def save_testing_metrics_graphic(total, classes, pred, clf):
    total_x = set(list(np.int32(total)))  # total classes for the x axis
    y_test_counter = dict.fromkeys(range(len(total_x)), 0)
    y_pred_counter = dict.fromkeys(range(len(total_x)), 0)

    pred = np.array(pred).reshape(-1)  # Normalize predictions array to 1D

    for i in classes:
        y_test_counter[int(i)] += 1
    for i in pred:
        y_pred_counter[int(i)] += 1

    x = np.array(list(total_x))
    y_test_counter = np.array(list(y_test_counter.values()))
    y_pred_counter = np.array(list(y_pred_counter.values()))

    accuracy_comparison = plt.figure(figsize=(14, 7))
    plt.plot(x, y_test_counter, 'o-', color='blue', label='Target Accuracy')
    plt.plot(x, y_pred_counter, 'o-', color='orange', label='Classifier Accuracy')
    plt.title(clf + ' detections/class in Testing')
    plt.xlabel("Class")
    plt.ylabel("Detections")
    plt.legend()
    accuracy_comparison.savefig('metrics/' + clf + '_classifier_testing_performance.png')
    print("Testing metrics saved on classifier_testing_performance.png")
