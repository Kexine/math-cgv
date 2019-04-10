import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
###

if __name__ == "__main__":

    # choose file to be calculated for each algorithm
    file_index = 1

    # suggested parameters
    V_abs = 1
    dVd_abs = 2*np.pi  # stays constant even when normalized
    sigma = 0.25
    ra = 0.25
    rb = 10

    step = (rb - ra)/100
    r_values = np.arange(ra, rb, step)

    algorithms = ["Balzer", "Dart", "FPO", "Matern"]
    n_algoritms = algorithms.__len__()

    results = np.zeros((100, n_algoritms))

    for algorithm_index in range(0, n_algoritms):
        algorithm = algorithms[algorithm_index]
        filename = "../PART I/Data/" + algorithm + "/" + str(file_index) + ".txt"
        data = np.loadtxt(filename)
        N = np.size(data, 0)
        r_max = np.sqrt(1/(2*np.sqrt(3)*N))  # (Lagae and Dutre, 2008) or (Gamito and Maddock, 2009 for n=2)
        data_normalized = data / r_max
        V_normalized = V_abs / np.square(r_max)
        pairwise_distances = scipy.spatial.distance.pdist(data_normalized)

        common_scaling = V_normalized / (dVd_abs * np.square(N))
        for r_index in range(0, np.size(r_values)):
            r = r_values[r_index]
            r_minus_pairwise_distances = r - pairwise_distances
            exponent = -np.square(r_minus_pairwise_distances) / np.square(sigma)
            k_sigma_values = 1/(np.sqrt(np.pi) * sigma) * np.exp(exponent)
            sum_k_sigma_values = 2*np.sum(k_sigma_values)  # x2 because of i,j and j,i indices for pairwise distance
            results[r_index, algorithm_index] = common_scaling * sum_k_sigma_values / r

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(results[:, 0])
    axes[0, 0].set_title("Balzer")
    axes[0, 1].plot(results[:, 1])
    axes[0, 1].set_title("Dart")
    axes[1, 0].plot(results[:, 2])
    axes[1, 0].set_title("FPO")
    axes[1, 1].plot(results[:, 3])
    axes[1, 1].set_title("Matern")
    plt.tight_layout()
    plt.show()
    print()