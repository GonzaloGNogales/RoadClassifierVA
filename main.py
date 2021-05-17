import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train_classifier(directory):
    X_feature_vector = list()
    y_labels = list()
    for label in os.listdir(directory):
        for image_name in os.listdir(directory+label):
            image = cv2.imread(directory+label+'/'+image_name, 0)  # Read the image in greyscale
            equalized_image = cv2.equalizeHist(image)  # Equalize the greyscale image
            resized_equalized_image = cv2.resize(equalized_image, (30, 30))  # Resize the equalized image to 30x30

            # HOG Descriptor extraction from a 30x30 image using 2x2 blocks of 15x15 pixels
            # divided in 3x3 cells of 5x5 pixels, and the algorithm moves a cell of 5x5 pixels each iteration
            hog = cv2.HOGDescriptor(_winSize=(30, 30), _blockSize=(15, 15), _blockStride=(5, 5),
                                    _cellSize=(5, 5), _nbins=9)
            act_feature_vector = hog.compute(resized_equalized_image)
            X_feature_vector.append(act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix
            y_labels.append(label)  # Add the corresponding label to the y labels vector that determines the class

    X = np.array(X_feature_vector)
    y = np.array(y_labels)
    X = X[:, :, 0]

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    Z = lda.transform(X)
    return lda


def recognize(trained_classifier, img):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--classifier', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    trained_classifier = train_classifier('./train_recortadas/')

    # For loop on testing images and pass them through the already trained classifier
    for file in os.listdir('./test_reconocimiento/'):
        if file != '.directory':
            img = cv2.imread(file)
            recognize(trained_classifier, img)
