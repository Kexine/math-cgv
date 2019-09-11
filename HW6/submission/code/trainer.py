from datetime import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers, models
from model import Model
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config


class Trainer:

    def __init__(self, model, train_dataset, val_dataset, test_dataset, out_path='../Results'):
        self.use_adv = model.use_adv

        self.model = model.build_model()
        self.sess = keras.backend.get_session()

        # Separate member for the tf.data type
        self.train_dataset = train_dataset
        self.train_dataset_tf = self.train_dataset.dataset_tf

        self.val_dataset = val_dataset
        self.val_dataset_tf = self.val_dataset.dataset_tf

        self.test_dataset = test_dataset
        self.test_dataset_tf = self.test_dataset.dataset_tf

        self.callbacks = []

        # Output path
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    def train(self, n_epochs=5):
        self.create_callbacks()

        self.callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.show_prediction(epoch)
            )
        )

        if not self.use_adv:
            self.model.fit(
                self.train_dataset_tf,
                epochs=n_epochs,
                steps_per_epoch=5*self.train_dataset.num_batches,
                callbacks=self.callbacks,
                validation_data=self.val_dataset_tf,
                validation_steps=self.val_dataset.num_batches,
            )

        else:
            self.train_adv_loss(n_epochs)

    def create_callbacks(self):
        """ Saves a  tensorboard with evolution of the loss """

        logdir = os.path.join(
            "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir,
            write_graph=True
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(logdir, "model_{epoch:04d}-{loss:.3f}-{val_loss:.3f}-.hdf5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            period=5,
            save_weights_only=True)

        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=15,
                                                                verbose=1)

        reduce_lr_on_plateau_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                          factor=0.85,
                                                                          patience=3,
                                                                          min_delta=0.03,
                                                                          min_lr=5e-7,
                                                                          cooldown=3,
                                                                          verbose=1)

        self.callbacks.append(tensorboard_callback)
        self.callbacks.append(model_checkpoint_callback)
        self.callbacks.append(early_stopping_callback)
        self.callbacks.append(reduce_lr_on_plateau_callback)

    def show_prediction(self, epoch):
        """ Save prediction image every 50 epochs """
        if epoch % 10 == 0:
            # Use the model to predict the values from the training dataset.
            train_predictions = self.run_prediction(self.train_dataset_tf)
            val_predictions = self.run_prediction(self.val_dataset_tf)

            fig = plt.figure()

            fig.add_subplot(2, 3, 1)
            plt.imshow(train_predictions[0])
            plt.axis('off')

            fig.add_subplot(2, 3, 2)
            plt.imshow(train_predictions[1])
            plt.axis('off')

            fig.add_subplot(2, 3, 3)
            plt.imshow(train_predictions[2])
            plt.axis('off')

            fig.add_subplot(2, 3, 4)
            plt.imshow(val_predictions[0])
            plt.axis('off')

            fig.add_subplot(2, 3, 5)
            plt.imshow(val_predictions[1])
            plt.axis('off')

            fig.add_subplot(2, 3, 6)
            plt.imshow(val_predictions[2])
            plt.axis('off')

            plt.savefig(os.path.join(
                self.out_path, "res_epoch_{:03d}.png".format(epoch)
            ))
            plt.close()

    def convert_to_img(self, np_array):
        img = np_array.copy()
        img *= 255.0
        img = np.maximum(np.minimum(255, img), 0)
        img = img.astype(np.uint8)

        return img

    def run_prediction(self, dataset):
        # Get a input, groundtruth pair from a tf.dataset
        input_img_array, gt_img_array = self.sess.run(
            dataset.make_one_shot_iterator().get_next()
        )

        return self.create_full_images(input_img_array, gt_img_array)

    def create_full_images(self, input_img_array, gt_img_array):
        """ 
        Create full images for visualization for one input image
    
        Input
           * input_img_array: input_img with cropped region, as (floats)
                - size: (batch_size, 128, 128, 3)
           * gt_img_array: groundtruth corresponding to the cropped region (floats)
                - size: (batch_size, 64, 64, 3)

        Output
           * input_img: input_img with cropped region, as numpy array (uint8)
                - size: (128, 128, 3)
           * full_gt: input_img with with cropped region replaced by gt (uint8)
                - size: (128, 128, 3)
           * full_pred: input_img with with cropped region replaced by prediction
                - size: (128, 128, 3)
        """
        prediction = self.model.predict(input_img_array, batch_size=Config.batch_size)[0, :, :, :]
        gt = gt_img_array[0, :, :, :]

        input_img = np.copy(input_img_array[0, :, :, :])
        full_gt = np.copy(input_img)
        full_pred = np.copy(input_img)

        if not Config.use_random_crop:
            batch_size, height, width, channel = input_img_array.shape
            left = int((width - 64) / 2)
            top = int((height - 64) / 2)
            right = int((width + 64) / 2)
            bottom = int((height + 64) / 2)

            full_gt[top:bottom, left:right, :] = np.copy(gt)
            full_pred[top+7:bottom-7, left+7:right-7, :] = np.copy(prediction[7:57, 7:57, :])
        else:
            is_cropped = input_img == 0.5
            is_cropped = np.logical_and(np.logical_and(is_cropped[:, :, 0], is_cropped[:, :, 1]), is_cropped[:, :, 2])

            row_locations, column_locations = np.where(is_cropped)

            left = np.min(column_locations)
            top = np.min(row_locations)
            right = np.max(column_locations)
            bottom = np.max(row_locations)

            full_gt[top:bottom+1, left:right+1, :] = np.copy(gt[7:57, 7:57, :])
            full_pred[top:bottom+1, left:right+1, :] = np.copy(prediction[7:57, 7:57, :])

        input_img = self.convert_to_img(input_img)
        full_gt = self.convert_to_img(full_gt)
        full_pred = self.convert_to_img(full_pred)

        return input_img, full_gt, full_pred

    def evaluate(self, dataset):
        # TODO Task 3.3
        pass

    def train_adv_loss(self, nepochs):
        dataset_iterator = self.train_dataset_tf.make_one_shot_iterator()
        dataset_next_data_pair = dataset_iterator.get_next()

        for epoch in range(nepochs):
            for _ in tqdm(range(self.train_dataset.num_batches)):
                input_img_array, gt_img_array = self.sess.run(
                    dataset_next_data_pair
                )

                # TODO Task 4.2

            self.show_prediction(epoch)
