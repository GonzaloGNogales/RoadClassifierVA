import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Bayes_LDA:
    def __init__(self):
        # Instantiate a linear discriminant analysis object (LDA) for reducing the data dimensionality
        self.lda = LinearDiscriminantAnalysis()
        # Instantiate a Normal Bayes Classifier already implemented in the machine learning module of OpenCV
        self.clf = cv2.ml.NormalBayesClassifier_create()

    def train(self, X_train, Y_train):
        # Perform the LDA fit function with the preprocessed data
        # X = Matrix of Feature Vectors and Y = class labels
        # as this is a supervised classifier
        X_train = self.lda.fit_transform(X_train, Y_train)

        # Perform the training on the X_train matrix previously reduced by LDA
        self.clf.train(np.float32(X_train), cv2.ml.ROW_SAMPLE, np.int32(Y_train))
        print('Normal Bayes Train completed successfully!')

        # Perform predict for extracting training accuracy
        _, predicted_train = self.clf.predict(np.float32(X_train))
        print("Accuracy in training =>",
              "{0:.4}".format((np.sum(1 * (np.int32(Y_train) == predicted_train[:, 0])) / Y_train.shape[0]) * 100), "%")

        return predicted_train

    def predict(self, X_test, Y_test):
        # Dimensionality reduction with pre-trained LDA for predicting on test data batch
        X_test = self.lda.transform(X_test)

        # Perform predict on the X_test matrix for obtaining the classifier probable class prediction
        _, predicted_test = self.clf.predict(np.float32(X_test))
        print("Accuracy in testing =>",
              "{0:.4}".format((np.sum(1 * (np.int32(Y_test) == predicted_test[:, 0])) / Y_test.shape[0]) * 100), "%")

        return predicted_test
