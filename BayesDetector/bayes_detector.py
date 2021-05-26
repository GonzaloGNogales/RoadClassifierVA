import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class BayesDetector:
    def __init__(self):
        # Instantiate a linear discriminant analysis object (LDA) for reducing the data dimensionality
        self.lda = LinearDiscriminantAnalysis()
        self.X = None  # Data
        self.Y = None  # Labels

    def preprocess_data(self, directory):
        X_feature_vector = list()
        Y_labels = list()
        for label in os.listdir(directory):
            for image_name in os.listdir(directory + label):
                image = cv2.imread(directory + label + '/' + image_name, 0)  # Read the image in greyscale
                equalized_image = cv2.equalizeHist(image)  # Equalize the greyscale image
                resized_equalized_image = cv2.resize(equalized_image, (30, 30))  # Resize the equalized image to 30x30

                # HOG Descriptor extraction from a 30x30 image using 2x2 blocks of 15x15 pixels
                # divided in 3x3 cells of 5x5 pixels, and the algorithm moves a cell of 5x5 pixels each iteration
                hog = cv2.HOGDescriptor(_winSize=(30, 30), _blockSize=(15, 15), _blockStride=(5, 5),
                                        _cellSize=(5, 5), _nbins=9)
                act_feature_vector = hog.compute(resized_equalized_image)
                X_feature_vector.append(act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix
                Y_labels.append(label)  # Add the corresponding label to the y labels vector that determines the class

        self.X = np.array(X_feature_vector)
        self.Y = np.array(Y_labels)
        self.X = self.X[:, :, 0]

    def dimensionality_reduction(self):
        # Perform the LDA fit function with the preprocessed data
        # X = Matrix of Feature Vectors and Y = class labels
        # as this is a supervised classifier
        plt.figure()
        for each_class in range(len(self.Y)):
            plt.plot(self.X[each_class])
        plt.show()
        self.lda.fit(self.X, self.Y)
        Z = self.lda.transform(self.X)

        plt.hist(Z[0:100, 0], 20, facecolor='green', alpha=0.75)
        plt.hist(Z[100:200, 0], 20, facecolor='red', alpha=0.75)
        plt.show()

        # result = self.lda.predict(Z)
        # print(result)

    def train(self):
        return None

    def predict(self):
        return None
