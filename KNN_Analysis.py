import time
from sklearn import neighbors
from KNN_plotit import NN, getExamples, accuracy
import matplotlib.pyplot as plt


def sample_and_dimension_size_comparision(max_size, option):
    run_times = [[], []]
    k = 5  # No. of neighbors to use for prediction
    for size in range(k, max_size):
        n = size if option == 'n' else 100  # number of examples of each class
        d = size if option == 'd' else 2  # number of dimensions

        xtr, ytr = getExamples(n=n, d=d)  # Generate Training Examples
        xtt, ytt = getExamples(n=n, d=d)  # Generate Testing Examples

        # Our KNN implementation
        start = time.time()
        my_clf = NN(k)
        my_clf.fit(xtr, ytr)
        my_clf.predict(xtt)
        run_times[0].append(time.time() - start)

        # Sklearn's implementation
        start = time.time()
        sklearn_clf = neighbors.KNeighborsClassifier(k)
        sklearn_clf.fit(xtr, ytr)
        sklearn_clf.predict(xtt)
        run_times[1].append(time.time() - start)

    return run_times, k


def K_comparision(max_k):
    results = [[], []]
    n = 100  # number of examples of each class
    d = 2  # number of dimensions
    xtr, ytr = getExamples(n=n, d=d)  # Generate Training Examples
    xtt, ytt = getExamples(n=n, d=d)  # Generate Testing Examples

    for k in range(1, max_k, 2):
        sklearn_clf = neighbors.KNeighborsClassifier(k)
        sklearn_clf.fit(xtr, ytr)

        y_target = sklearn_clf.predict(xtr)
        results[0].append(accuracy(y_target, ytr))
        y_target = sklearn_clf.predict(xtt)
        results[1].append(accuracy(y_target, ytt))

    return results


if __name__ == '__main__':
    max_sample_size = 200
    times, start_size = sample_and_dimension_size_comparision(max_sample_size, 'n')
    plt.title('Run-times for varying sample sizes')
    plt.scatter(range(start_size, max_sample_size), times[0], color='r')
    plt.scatter(range(start_size, max_sample_size), times[1], color='b')
    plt.xlabel('No. of samples')
    plt.ylabel('Time taken (s)')
    plt.legend(['Our implementation', 'Sklearn\'s implementation'])
    plt.savefig(fname='samples_run_time.svg', format='svg')
    plt.show()

    max_dimension_size = 200
    times, start_size = sample_and_dimension_size_comparision(max_dimension_size, 'd')
    plt.title('Run-times for varying dimension sizes')
    plt.scatter(range(start_size, max_dimension_size), times[0], color='r')
    plt.scatter(range(start_size, max_dimension_size), times[1], color='b')
    plt.xlabel('No. of dimensions')
    plt.ylabel('Time taken (s)')
    plt.legend(['Our implementation', 'Sklearn\'s implementation'])
    plt.savefig(fname='dimensions_run_time.svg', format='svg')
    plt.show()

    max_K_size = 50
    accuracies = K_comparision(max_K_size)
    plt.title('Accuracies for various K values')
    plt.plot(range(1, max_K_size, 2), accuracies[0], color='b')
    plt.plot(range(1, max_K_size, 2), accuracies[1], color='r')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend(['Train data', 'Test data'])
    plt.savefig(fname='k_accuracies.svg', format='svg')
    plt.show()
