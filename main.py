import argparse
import os
import cv2
import numpy as np
from Bayes_LDA_Classifier.bayes_lda import Bayes_LDA
from KNN_PCA_Classifier.knn_pca import KNN_PCA
from Euclidean_LDA_Classifier.euclidean_lda import Euclidean_LDA


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


def save_results(f_names, predicts, bayes=False, knn=False, euclidean=False):
    # Prepare the detection results file
    # Check if results file already exists and remove it, if not create it
    if os.path.isfile('resultado.txt'):
        os.remove('resultado.txt')
    results = open('resultado.txt', 'w')

    # Save results on 'resultado.txt'
    for i in range(len(f_names)):
        if bayes:
            results.write(f_names[i] + '; ' + str('{:02}'.format(predicts[i][0])) + '\n')
        elif knn:
            results.write(f_names[i] + '; ' + str('{:02}'.format(predicts[i])) + '\n')
        elif euclidean:
            results.write(f_names[i] + '; ' + str('{:02}'.format(predicts[i])) + '\n')

    print('Results correctly saved! >>>> check them in resultado.txt')
    results.close()  # Results txt file communication closed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default='./train_recortadas/', help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default='./test_reconocimiento/', help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default='BAYES', help='String con el nombre del clasificador')

    args = parser.parse_args()

    # Instantiate a custom Bayes Detector object
    if args.classifier == 'BAYES':
        classifier = Bayes_LDA()
    elif args.classifier == 'KNN-PCA':
        classifier = KNN_PCA()
    elif args.classifier == 'EUCLIDEAN':
        classifier = Euclidean_LDA()
    else:
        raise ValueError('Wrong classifier type :(')

    # Load train data
    X_train, Y_train = preprocess_data(vars(args)['train_path'], True)

    # Train the classifier
    classifier.train(X_train, Y_train)

    # Load test data
    X_test, Y_test, filenames = preprocess_data(vars(args)['test_path'], False)

    # Predict on test data
    predictions = classifier.predict(X_test, Y_test)

    # Save the results on the 'resultado.txt' file
    if args.classifier == 'BAYES':
        save_results(filenames, predictions, bayes=True)
    elif args.classifier == 'KNN-PCA':
        save_results(filenames, predictions, knn=True)
    elif args.classifier == 'EUCLIDEAN':
        save_results(filenames, predictions, euclidean=True)
