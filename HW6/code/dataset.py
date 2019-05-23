import glob
import numpy as np 
import os
import skimage
import tensorflow as tf
from config import Config
import functools

use_random_crop = Config.use_random_crop

class Dataset:

    def __init__(self, data_path, is_training_set=False):
        self.is_training_set = is_training_set
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
                img /= 255.0

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

        if Config.use_random_augment and self.is_training_set:
            self.dataset_tf = self.dataset_tf.map(functools.partial(self.random_augment), num_parallel_calls=8)
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
        img_with_missing_crop[:, top+7:bottom-7, left+7:right-7, :] = Config.missing_patch_fill_value
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


    def random_augment(self, image, gt):

        # left right flip
        toss_flip = tf.random.uniform([], minval=0, maxval=1)
        image = tf.cond(toss_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        gt = tf.cond(toss_flip > 0.5, lambda: tf.image.flip_left_right(gt), lambda: gt)

        # gaussian additive noise
        toss_gauss = tf.random.uniform([], minval=0, maxval=1)
        noise_image = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.03, dtype=tf.float32)
        image = tf.cond(toss_gauss > 0.75, lambda: tf.add(image, noise_image), lambda: image)

        # random color change
        toss_color = tf.random.uniform([], minval=0, maxval=1)
        random_hue = tf.random.uniform([], minval=0, maxval=0.1)
        random_contrast = tf.random.uniform([], minval=0.5, maxval=1.5)
        random_saturation = tf.random.uniform([], minval=0.5, maxval=1.5)
        image = tf.cond(toss_color > 0.75, lambda: tf.image.adjust_hue(image, random_hue), lambda: image)
        image = tf.cond(toss_color > 0.75, lambda: tf.image.adjust_contrast(image, random_contrast), lambda: image)
        image = tf.cond(toss_color > 0.75, lambda: tf.image.adjust_saturation(image, random_saturation), lambda: image)
        gt = tf.cond(toss_color > 0.75, lambda: tf.image.adjust_hue(gt, random_hue), lambda: gt)
        gt = tf.cond(toss_color > 0.75, lambda: tf.image.adjust_contrast(gt, random_contrast), lambda: gt)
        gt = tf.cond(toss_color > 0.75, lambda: tf.image.adjust_saturation(gt, random_saturation), lambda: gt)

        return tuple([image, gt])

