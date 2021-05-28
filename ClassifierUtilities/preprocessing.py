import os
import cv2
import numpy as np


def preprocess_data(directory, train):
    X_feature_vector = list()
    Y_labels = list()
    original_names = list()  # List for saving the testing file image names for building the 'resultado.txt'
    if train:
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
                X_feature_vector.append(
                    act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix
                Y_labels.append(label)  # Add the corresponding label to the y labels set that determines the class

        return np.array(X_feature_vector)[:, :, 0], np.array(Y_labels)
    else:
        for image_name in os.listdir(directory):
            if image_name[-4:] == '.ppm':
                # Separate the class of the image name from the identifier to create the Y_test vector for metrics
                label = image_name[0:2]
                image = cv2.imread(directory + '/' + image_name, 0)  # Read the image in greyscale
                equalized_image = cv2.equalizeHist(image)  # Equalize the greyscale image
                resized_equalized_image = cv2.resize(equalized_image, (30, 30))  # Resize the equalized image to 30x30

                # HOG Descriptor extraction from a 30x30 image using 2x2 blocks of 15x15 pixels
                # divided in 3x3 cells of 5x5 pixels, and the algorithm moves a cell of 5x5 pixels each iteration
                hog = cv2.HOGDescriptor(_winSize=(30, 30), _blockSize=(15, 15), _blockStride=(5, 5),
                                        _cellSize=(5, 5), _nbins=9)
                act_feature_vector = hog.compute(resized_equalized_image)
                X_feature_vector.append(
                    act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix
                Y_labels.append(label)  # Add the corresponding label to the y labels set that determines the class
                original_names.append(image_name)  # Save the image name for the results file later

        return np.array(X_feature_vector)[:, :, 0], np.array(Y_labels), original_names


# EXERCISE 3 PREPROCESS REGIONS AND RETURNS THE X_TRAIN OR X_TEST MATRIX
def extract_HOG_features(regions, labels=None):
    X_feature_vector = list()
    for reg in regions:
        reg = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(reg)  # Equalize the greyscale image
        resized_equalized_image = cv2.resize(equalized_image, (30, 30))  # Resize the equalized image to 30x30

        # HOG Descriptor extraction from a 30x30 image using 2x2 blocks of 15x15 pixels
        # divided in 3x3 cells of 5x5 pixels, and the algorithm moves a cell of 5x5 pixels each iteration
        hog = cv2.HOGDescriptor(_winSize=(30, 30), _blockSize=(15, 15), _blockStride=(5, 5),
                                _cellSize=(5, 5), _nbins=9)
        act_feature_vector = hog.compute(resized_equalized_image)
        X_feature_vector.append(act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix

    return np.array(X_feature_vector)[:, :, 0], np.array(labels)
