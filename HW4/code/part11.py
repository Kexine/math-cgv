import numpy as np
import matplotlib.pyplot as plt
###

if __name__ == "__main__":

    image_height = 400
    image_width = 400

    algorithms = ["Balzer", "Dart", "FPO", "Matern"]
    n_algoritms = algorithms.__len__()

    periodograms = np.zeros((image_height, image_width, 10, 4))

    for algorithm_index in range(0, n_algoritms):
        algorithm = algorithms[algorithm_index]
        for file_index in range(1, 11):
            initial_image = np.zeros((image_height, image_width))

            filename = "../PART I/Data/" + algorithm + "/" + str(file_index) + ".txt"
            data = np.loadtxt(filename)
            data[:, 0] = np.floor(data[:, 0] * image_width)
            data[:, 1] = np.floor(data[:, 1] * image_height)
            data = data.astype(np.int)
            n = np.size(data, 0)

            for point in data:
                i = point[1]
                j = point[0]
                initial_image[i, j] = initial_image[i, j] + 1/n

            periodograms[:, :, file_index-1, algorithm_index] = np.square(np.abs(np.fft.fftshift(np.fft.fft2(initial_image))))

    periodograms_averaged = np.mean(periodograms, 2)

    periodograms_averaged_scaled = periodograms_averaged * 200


    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(periodograms_averaged_scaled[:, :, 0], cmap='gray', clim=(0, 1))
    axes[0, 0].set_title("Balzer")
    axes[0, 1].imshow(periodograms_averaged_scaled[:, :, 1], cmap='gray', clim=(0, 1))
    axes[0, 1].set_title("Dart")
    axes[1, 0].imshow(periodograms_averaged_scaled[:, :, 2], cmap='gray', clim=(0, 1))
    axes[1, 0].set_title("FPO")
    axes[1, 1].imshow(periodograms_averaged_scaled[:, :, 3], cmap='gray', clim=(0, 1))
    axes[1, 1].set_title("Matern")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
