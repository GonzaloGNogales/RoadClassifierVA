import os


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
