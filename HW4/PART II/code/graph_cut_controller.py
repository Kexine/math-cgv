import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage import data, color, img_as_float
from tkinter import *
from PIL import Image

from graph_cut import GraphCut
from graph_cut_gui import GraphCutGui


class GraphCutController:

    def __init__(self):
        self.__init_view()

    def __init_view(self):
        root = Tk()
        root.geometry("700x500")
        self._view = GraphCutGui(self, root)
        root.mainloop()

    # TODO: TASK 2.1
    def __get_color_histogram(self, image, seed, hist_res):
        """
        Compute a color histograms based on selected points from an image
        :param image: color image
        :param seed: Nx2 matrix containing the the position of pixels which will be used to compute the color histogram
        :param histRes: resolution of the histogram
        :return hist: color histogram
        """
        seed_r_values = image[seed[:, 1], seed[:, 0], 0]
        seed_g_values = image[seed[:, 1], seed[:, 0], 1]
        seed_b_values = image[seed[:, 1], seed[:, 0], 2]

        data = np.transpose(np.vstack((seed_r_values, seed_g_values, seed_b_values)))
        histogram, _ = np.histogramdd(data, hist_res, range=[(0, 255), (0, 255), (0, 255)])

        # w = 2*int(truncate*sigma + 0.5) + 1
        # sigma = 0.65 is taken from MATLAB default, truncate = 4 in scipy default which results in w = 7
        smoothed_histogram = ndimage.gaussian_filter(histogram, 0.65)
        normalized_smoothed_histogram = smoothed_histogram / np.sum(smoothed_histogram.ravel())

        return normalized_smoothed_histogram

    # TODO: TASK 2.2
    # Hint: Set K very high using numpy's inf parameter
    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        """
        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
        print("Calcuating unaries...")
        hist_step = 255.0 / 32.0
        image_rows = np.size(image, 0)
        image_cols = np.size(image, 1)

        unaries = np.empty((image_rows, image_cols, 2))

        for i in range(0, image_rows):
            for j in range(0, image_cols):
                pixel = image[i, j, :]
                pixel_bins = np.floor(pixel / hist_step).astype(int)
                pixel_bins[pixel_bins == 32] = 31

                cost_fg = -np.log(hist_fg[pixel_bins[0], pixel_bins[1], pixel_bins[2]] + 1e-10)
                cost_bg = -np.log(hist_bg[pixel_bins[0], pixel_bins[1], pixel_bins[2]] + 1e-10)
                unaries[i, j, 1] = lambda_param * cost_bg
                unaries[i, j, 0] = lambda_param * cost_fg

        for j, i in seed_fg:
            unaries[i, j, 1] = np.inf
            unaries[i, j, 0] = 0

        for j, i in seed_bg:
            unaries[i, j, 1] = 0
            unaries[i, j, 0] = np.inf

        unariesN = np.reshape(unaries, (-1, 2))

        return unariesN

    # TASK 2.3
    def __get_pairwise(self, image, sigma):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :param sigma: ad-hoc cost function parameter
        :return: pairwise : ivj (triplet or coo) formatted list of lists containing the pairwise costs for image
        """
        def get_neighbours(i, j, image_rows, image_cols):
            neighbours = np.array([[i - 1, j - 1],  # upper left
                                   [i - 1, j],  # upper
                                   [i - 1, j + 1],  # upper right
                                   [i, j + 1],  # right
                                   [i + 1, j + 1],  # lower right
                                   [i + 1, j],  # lower
                                   [i + 1, j - 1],  # lower left
                                   [i, j - 1]])  # left

            is_boundary_1 = 0 <= neighbours[:, 0]
            is_boundary_2 = image_rows > neighbours[:, 0]
            is_boundary_3 = 0 <= neighbours[:, 1]
            is_boundary_4 = image_cols > neighbours[:, 1]

            valid = np.logical_and(np.logical_and(is_boundary_1, is_boundary_2), np.logical_and(is_boundary_3, is_boundary_4))

            return neighbours[valid, :]

        print("Calcuating pairwises...")

        image_rows = np.size(image, 0)
        image_cols = np.size(image, 1)

        pairwise = []
        for i in range(0, image_rows):
            for j in range(0, image_cols):
                current_coordinates = np.array([i, j])
                current_index = i * image_cols + j
                current_pixel = image[i, j].astype(float)
                neighbour_coordinates = get_neighbours(i, j, image_rows, image_cols)
                neighbour_indices = neighbour_coordinates[:, 0] * image_cols + neighbour_coordinates[:, 1]
                neighbour_pixels = image[neighbour_coordinates[:, 0], neighbour_coordinates[:, 1]].astype(float)

                pixel_differences = np.subtract(neighbour_pixels, current_pixel)
                pixel_distances = np.linalg.norm(pixel_differences, axis=1)
                spatial_differences = current_coordinates - neighbour_coordinates
                spatial_differences = np.linalg.norm(spatial_differences, axis=1)

                neighbour_costs = np.divide(np.exp(-np.square(pixel_distances) / (2 * np.square(sigma))),
                                            spatial_differences)

                for k in range(0, np.size(neighbour_indices.ravel())):
                    neighbour_index = neighbour_indices[k]
                    cost = neighbour_costs[k]
                    pairwise.append([current_index, neighbour_index, 0, cost, 0, 0])

                if current_index%1000 == 0:
                    print(current_index, '/', image_rows*image_cols)


        pairwise = np.asarray(pairwise)
        return pairwise


    # TODO TASK 2.4 get segmented image to the view
    def __get_segmented_image(self, image, labels, background=None):
        """
        Return a segmented image, as well as an image with new background 
        :param image: color image as a numpy array
        :param label: labels a numpy array
        :param background: color image as a numpy array
        :return image_segmented: image as a numpy array with red foreground, blue background
        :return image_with_background: image as a numpy array with changed background if any (None if not)
        """
        image_rows = np.size(image, 0)
        image_cols = np.size(image, 1)

        not_labels = np.logical_not(labels)
        mask = np.zeros((image_rows, image_cols, 3), dtype=np.uint8)
        mask[not_labels, :] = np.array([255, 0, 0], dtype=np.uint8)
        mask[labels, :] = np.array([0, 0, 255])

        image_PIL = Image.fromarray(image)
        mask_PIL = Image.fromarray(mask)
        result_PIL = Image.blend(image_PIL, mask_PIL, 0.6)

        segmented_image = np.array(result_PIL)

        if background is not None:
            mask = np.zeros((image_rows, image_cols), dtype=np.bool)
            mask[np.logical_not(labels)] = np.bool(1)
            result = np.copy(background[0:image_rows, 0:image_cols, :])
            result.setflags(write=1)
            result[not_labels, 0:3] = image[not_labels, 0:3]
            segmented_image_with_background = result
        else:
            segmented_image_with_background = None

        return segmented_image, segmented_image_with_background

    def segment_image(self, image, seed_fg, seed_bg, lambda_value, background=None):
        image_array = np.asarray(image)
        background_array = None
        if background:
            background = background.convert("RGB")
            background_array = np.asarray(background)
        seed_fg = np.array(seed_fg)
        seed_bg = np.array(seed_bg)
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        # TASK 2.1 - get the color histogram for the unaries
        hist_res = 32
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array, sigma=5)

        # TODO: TASK 2.4 - perform graph cut
        g = GraphCut(num_pixels, pairwise.__len__())
        g.set_unary(unaries)
        g.set_pairwise(pairwise)
        g.minimize()
        labels = g.get_labeling()

        labels = np.reshape(labels, (height, width))

        # plt.imshow(labels)
        # plt.show()

        # TODO TASK 2.4 get segmented image to the view
        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels, background_array)
        # transform image array to an rgb image
        segmented_image = Image.fromarray(segmented_image, 'RGB')
        self._view.set_canvas_image(segmented_image)
        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            plt.imshow(segmented_image_with_background)
            plt.show()
