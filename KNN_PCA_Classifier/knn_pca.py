import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


class KNN_PCA:
    def __init__(self):
        # Instantiate a principal components analysis object (PCA) for reducing the data dimensionality
        self.pca = PCA()
        # Instantiate a KNN Classifier already implemented in the neighbors module of sklearn
        self.clf = KNeighborsClassifier(n_neighbors=5)

    def train(self, X_train, Y_train):
        # Perform the PCA fit function with the preprocessed data and then transform it
        # X = Matrix of Feature Vectors (HOG)
        # as this is a non supervised classifier
        X_train = self.pca.fit_transform(X_train)

        # Perform the training on the X_train matrix previously reduced by LDA
        self.clf.fit(np.float32(X_train), np.int32(Y_train))
        print('KNN Train completed successfully!')

        # Perform predict for extracting training accuracy
        predicted_train = self.clf.predict(np.float32(X_train))
        print("Accuracy in training =>",
              "{0:.4}".format((np.sum(1 * (np.int32(Y_train) == predicted_train)) / Y_train.shape[0]) * 100), "%")

        return predicted_train

    def predict(self, X_test, Y_test):
        # Dimensionality reduction with pre-trained LDA for predicting on test data batch
        X_test = self.pca.transform(X_test)

        # Perform predict on the X_test matrix for obtaining the classifier probable class prediction
        predicted_test = self.clf.predict(np.float32(X_test))
        print("Accuracy in testing =>",
              "{0:.4}".format((np.sum(1 * (np.int32(Y_test) == predicted_test)) / Y_test.shape[0]) * 100), "%")

        return predicted_test
