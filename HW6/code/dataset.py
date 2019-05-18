import glob
import numpy as np 
import os
import skimage
import tensorflow as tf
from config import Config

use_random_crop = Config.use_random_crop

class Dataset:

    def __init__(self, data_path):
        self.dataset_tf = None
        self.data = None
        self.mean = None
        self.std = None

        self.batch_size = None
        self.num_batches = None

        self.data_path = data_path
        self.read_data(data_path)

        self.num_data = self.data.shape[0]


        if not use_random_crop:
            self.crop_center_img()


    def read_data(self, data_path):
        # Get list of files
        data = []

        for img_type in ["jpg", "png"]:
            for img_path in glob.glob("{}/*.{}".format(data_path, img_type)):
                img_name = os.path.basename(img_path)
                
                img = skimage.io.imread(img_path).astype(np.float32)
                img = (img - 127.0) / 255.0  # should be zero centered as paper suggests

                # Append to output list
                data.append(img.astype(np.float32))

        self.data = np.stack(data, axis=0)

        self.mean = self.data.mean()
        self.std = self.data.std()


    def create_tf_dataset(self, batch_size=32):
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.num_data / batch_size).astype(int)

        self.dataset_tf = tf.data.Dataset.from_tensor_slices(self.data)

        if use_random_crop:
            self.dataset_tf = self.dataset_tf.map(self.crop_random_img)

        self.dataset_tf = self.dataset_tf.shuffle(200)
        self.dataset_tf = self.dataset_tf.batch(batch_size)
        self.dataset_tf = self.dataset_tf.repeat()            


    def crop_center_img(self):
        """
        Modifies self.data in order to remove the center region
        self.data should now be a tuple 
        (img_with_missing_crop, groundtruth_crop)
        """
        # Task 1.1
        batch, height, width, channel = self.data.shape
        left = int((width - 64) / 2)
        top = int((height - 64) / 2)
        right = int((width + 64) / 2)
        bottom = int((height + 64) / 2)

        img_with_missing_crop = np.copy(self.data)
        img_with_missing_crop[:, top+7:bottom-7, left+7:right-7, :] = 0.0
        groundtruth_crop = np.copy(self.data[:, top:bottom, left:right, :])

        self.data = (img_with_missing_crop, groundtruth_crop)


    def crop_random_img(self, input_img):
        #TODO Task 1.2
        """
        Modifies the input_img in order to remove a random region
        It must return the following tuple 
        (img_with_missing_crop, groundtruth_crop)
        """
        return (img_with_missing_crop, groundtruth_crop)