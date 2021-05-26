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
    detector.preprocess_data(vars(args)['train_path'])

    # Tratamiento de los datos
    detector.dimensionality_reduction()

    # Entrenar el clasificador si es necesario ...
    # detector ...

    # Cargar y procesar imgs de test
    # args.train_path ...

    # Guardar los resultados en ficheros de texto (en el directorio donde se
    # ejecuta el main.py) tal y como se pide en el enunciado.
