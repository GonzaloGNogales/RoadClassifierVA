import argparse
from Bayes_LDA_Classifier.bayes_lda import Bayes_LDA
from KNN_PCA_Classifier.knn_pca import KNN_PCA
from ClassifierUtilities.results_computation import save_results
from MSERDetector.mser_detector import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default='./train_recortadas/', help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default='./test_reconocimiento/', help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default='BAYES-LDA', help='String con el nombre del clasificador')

    args = parser.parse_args()
    classifier = None
    detector = None

    # Instantiate a custom Bayes Detector object
    if args.classifier == 'BAYES-LDA':
        classifier = Bayes_LDA()
    elif args.classifier == 'KNN-PCA':
        classifier = KNN_PCA()
    elif args.classifier == 'EUCLIDEAN-LDA':
        classifier = Euclidean_LDA()
    elif args.classifier == 'MSER-EUCLIDEAN-LDA':
        detector = MSER_Detector()
    else:
        raise ValueError('Wrong classifier type :(')

    if classifier is not None:
        # Load train data
        X_train, Y_train = preprocess_data(vars(args)['train_path'], True)

        # Train the classifier
        train_predictions = classifier.train(X_train, Y_train)

        # Plot training metrics graphic
        set_up_metrics_directory()
        save_training_metrics_graphic(Y_train, train_predictions, args.classifier)

        # Load test data
        X_test, Y_test, filenames = preprocess_data(vars(args)['test_path'], False)

        # Predict on test data
        test_predictions = classifier.predict(X_test, Y_test)

        # Calculate f1 metrics and generate graphics
        f1_score_micro = f1(Y_test, test_predictions, 'micro')
        print('The selected classifier micro f1 score is:', f1_score_micro)
        f1_score_macro = f1(Y_test, test_predictions, 'macro')
        print('The selected classifier macro f1 score is:', f1_score_macro)

        # Plot testing metrics graphic
        save_testing_metrics_graphic(Y_train, Y_test, test_predictions, args.classifier)

        # Save the results on the 'resultado.txt' file
        if args.classifier == 'BAYES-LDA':
            save_results(filenames, test_predictions, bayes=True)
        elif args.classifier == 'KNN-PCA':
            save_results(filenames, test_predictions, knn=True)
        elif args.classifier == 'EUCLIDEAN-LDA':
            save_results(filenames, test_predictions, euclidean=True)

    elif detector is not None:
        # Load training data
        detector.preprocess_data(vars(args)['train_path'], True)

        # Training
        status = detector.fit()

        # Load testing data
        detector.preprocess_data(vars(args)['test_path'], False)

        # Testing -> see results in ./resultado_imgs/
        detector.predict(status)

    else:
        print("Please introduce the options \"--train_path <train_path> --test_path <test_path> "
              "--classifier <classifier_name>\" in the command line of the terminal")
