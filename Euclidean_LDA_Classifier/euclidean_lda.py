import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Euclidean_LDA:
    def __init__(self):
        # Instantiate a linear discriminant analysis object (LDA) for reducing the data dimensionality
        self.lda = LinearDiscriminantAnalysis()
        self.centroids = list()

    def train(self, X_train, Y_train):
        # Perform the LDA fit function with the preprocessed data
        # X = Matrix of Feature Vectors and Y = class labels
        # as this is a supervised classifier
        X_train = self.lda.fit_transform(X_train, Y_train)

        # Order the training data by classes to compute centroids later
        data_distribution = list()
        unique_classes = set(list(Y_train))  # Extract the unique classes into a unique classes set
        for i in range(len(unique_classes)):
            data_distribution.append(X_train[np.int32(Y_train) == i])

        # Calculate the centroids of the classes
        for i in range(len(data_distribution)):
            self.centroids.append(np.mean(data_distribution[i], axis=0))

        # Perform the training on the X_train matrix previously reduced by LDA
        print('Euclidean Distance Train completed successfully!')

        # Perform predict for extracting training accuracy
        predicted_train = list()
        for i in range(len(X_train)):
            actual_distances = list()
            for j in range(len(self.centroids)):
                actual_distances.append(np.sqrt(np.sum((self.centroids[j] - X_train[i]) ** 2)))
            predicted_train.append(np.argmin(actual_distances))

        print("Accuracy in training =>",
              "{0:.4}".format((np.sum(1 * (np.int32(Y_train) == predicted_train)) / Y_train.shape[0]) * 100), "%")

        return predicted_train

    def predict(self, X_test, Y_test):
        # Dimensionality reduction with pre-trained LDA for predicting on test data batch
        X_test = self.lda.transform(X_test)

        # Perform predict for extracting training accuracy
        predicted_test = list()
        for i in range(len(X_test)):
            actual_distances = list()
            for j in range(len(self.centroids)):
                actual_distances.append(np.sqrt(np.sum((self.centroids[j] - X_test[i]) ** 2)))
            predicted_test.append(np.argmin(actual_distances))

        print("Accuracy in testing =>",
              "{0:.4}".format((np.sum(1 * (np.int32(Y_test) == predicted_test)) / Y_test.shape[0]) * 100), "%")

        return predicted_test
