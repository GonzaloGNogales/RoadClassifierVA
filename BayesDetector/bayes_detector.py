import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class BayesDetector:
    def __init__(self):
        # Instantiate a linear discriminant analysis object (LDA) for reducing the data dimensionality
        self.lda = LinearDiscriminantAnalysis()
        # Instantiate a Normal Bayes Classifier already implemented in the machine learning module of OpenCV
        self.clf = cv2.ml.NormalBayesClassifier_create()

        self.X_train = None  # Train Data
        self.Y_train = None  # Train Labels

        self.original_names = list()  # List for saving the testing file image names for building the 'resultado.txt'
        self.X_test = None  # Test Data
        self.Y_test = None  # Test Labels

    def preprocess_data(self, directory, train):
        X_feature_vector = list()
        Y_labels = list()
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
                    X_feature_vector.append(act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix
                    Y_labels.append(label)  # Add the corresponding label to the y labels set that determines the class

            self.X_train = np.array(X_feature_vector)
            self.Y_train = np.array(Y_labels)
            self.X_train = self.X_train[:, :, 0]
        else:
            for image_name in os.listdir(directory):
                if image_name[-4:] == ".ppm":
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
                    X_feature_vector.append(act_feature_vector)  # Add the computed HOG feature vector to the X feature matrix
                    Y_labels.append(label)  # Add the corresponding label to the y labels set that determines the class
                    self.original_names.append(image_name)  # Save the image name for the results file later

            self.X_test = np.array(X_feature_vector)
            self.Y_test = np.array(Y_labels)
            self.X_test = self.X_test[:, :, 0]

    def dimensionality_reduction(self):
        # Perform the LDA fit function with the preprocessed data
        # X = Matrix of Feature Vectors and Y = class labels
        # as this is a supervised classifier
        self.lda.fit(self.X_train, self.Y_train)
        self.X_train = self.lda.transform(self.X_train)

    def train(self):
        # Perform the training on the X_train matrix previously reduced by LDA
        self.clf.train(np.float32(self.X_train), cv2.ml.ROW_SAMPLE, np.int32(self.Y_train))
        print('Train completed successfully!')

        # Perform predict for extracting training accuracy
        _, predicted_train = self.clf.predict(np.float32(self.X_train))
        self.Y_train = self.Y_train.reshape((self.Y_train.shape[0], 1))
        count = 0
        for i in range(len(self.Y_train)):
            if int(self.Y_train[i][0]) == predicted_train[i][0]:
                count += 1
        print("Accuracy in training => ", (count/self.Y_train.shape[0]) * 100, "%")

    def predict(self):
        # Prepare the detection results file
        # Check if results file already exists and remove it, if not create it
        if os.path.isfile('resultado.txt'):
            os.remove('resultado.txt')
        results = open('resultado.txt', 'w')

        # Perform predict on the X_test matrix for obtaining the classifier probable class prediction
        _, predicted_test = self.clf.predict(np.float32(self.X_test))
        self.Y_test = self.Y_test.reshape((self.Y_test.shape[0], 1))
        count = 0
        for i in range(len(self.Y_test)):
            if int(self.Y_test[i][0]) == predicted_test[i][0]:
                count += 1
        print("Accuracy in test => ", (count / self.Y_test.shape[0]) * 100, "%")

        # Save results on 'resultado.txt'
        for i in range(len(self.original_names)):
            results.write(self.original_names[i] + '; ' + str(predicted_test[i]) + '\n')

        results.close()  # Results txt file communication closed
