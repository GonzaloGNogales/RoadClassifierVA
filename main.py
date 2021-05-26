import argparse
from BayesDetector.bayes_detector import BayesDetector


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default="./train_recortadas/", help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default="./test_reconocimiento/", help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default="BAYES", help='String con el nombre del clasificador')

    args = parser.parse_args()

    # Instantiate a custom Bayes Detector object
    if args.classifier == "BAYES":
        detector = BayesDetector()
    else:
        raise ValueError('Wrong classifier type :(')

    # Load train data
    detector.preprocess_data(vars(args)['train_path'], True)

    # LDA dimensionality reduction
    detector.dimensionality_reduction()

    # Train the classifier
    detector.train()

    # Load test data
    detector.preprocess_data(vars(args)['test_path'], False)

    # Predict con test data, this automatically saves the results on the 'resultado.txt' file
    detector.predict()
