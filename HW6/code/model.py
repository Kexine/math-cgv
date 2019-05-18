from config import Config
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers, regularizers, optimizers


class Model:

    def __init__(self, data_shape):
        self.data_shape = data_shape
        self.use_adv = Config.use_adversarial
        self.model = None
        self.discriminator_graph = None
        self.model_discriminator_graph = None

        if Config.use_overlapping_loss:
            self.loss_function = self.reconstruction_loss_overlap
        else:
            self.loss_function = self.reconstruction_loss

        self.loss_weights = np.ones((1, 64, 64, 3), dtype=np.float32) * 10.0
        self.loss_weights[:, 7:57, 7:57, :] = 1.0
        self.loss_weights = tf.convert_to_tensor(self.loss_weights)

    def build_model(self):
        if not self.use_adv:
            # Change loss to reconstruction_loss_overlap when needed
            self.build_reconstruction_model(self.loss_function)
        else:
            self.create_discriminator_graph()
            self.build_reconstruction_adversarial_model()

        return self.model

    def build_reconstruction_model(self, loss):
        inputs = keras.Input(shape=self.data_shape)
        outputs = self.encoder_decoder_graph(inputs)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(lr=Config.learning_rate),
            loss=loss,
        )

    def build_reconstruction_adversarial_model(self):
        # Create adversarial graph
        inputs_discriminator_graph = keras.Input(shape=(64, 64, 3))
        output = self.discriminator_graph(inputs_discriminator_graph)

        self.model_discriminator_graph = keras.Model(inputs_discriminator_graph, output)

        self.model_discriminator_graph.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=1.0e-4),
            loss_weights=[0.001]
        )

        # Input to the whole model
        inputs = keras.Input(shape=self.data_shape)

        # Reconstruction
        reconstruction_output = self.encoder_decoder_graph(inputs)
        adversarial_output = self.discriminator_graph(reconstruction_output)

        self.model = keras.Model(
            inputs,
            outputs=[reconstruction_output, adversarial_output]
        )

        self.model.compile(
            optimizer=optimizers.Adam(lr=1.0e-4),
            loss=[self.reconstruction_loss, 'binary_crossentropy'],
            loss_weights=[0.999, 0.001]
        )

    def encoder_decoder_graph(self, inputs):
        """ 
        Creates a graph and store it in self.encoder_decoder_graph 
        No return
        """

        # Task 2.1
        def encoder_block(a, n_filters):
            a = layers.Conv2D(filters=n_filters, kernel_size=(4, 4), padding='same',
                              kernel_regularizer=regularizers.l1_l2(
                                  l1=Config.l1_kernel_regularization,
                                  l2=Config.l2_kernel_regularization))(a)
            a = layers.BatchNormalization()(a)
            a = layers.LeakyReLU()(a)
            a = layers.MaxPool2D(pool_size=(2, 2))(a)
            return a

        def decoder_block(a, n_filters):
            a = layers.Conv2DTranspose(filters=n_filters, kernel_size=(4, 4), padding='same', strides=(2, 2),
                                       kernel_regularizer=regularizers.l1_l2(
                                           l1=Config.l1_kernel_regularization,
                                           l2=Config.l2_kernel_regularization))(a)
            a = layers.BatchNormalization()(a)
            a = layers.ReLU()(a)
            return a

        # Encoder
        x = inputs
        for n_filters in [64, 64, 128, 256, 512]:
            x = encoder_block(x, n_filters=n_filters)

        # Encoder-to-Dense Layer
        x = layers.Conv2D(filters=4000, kernel_size=(4, 4), padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Dense-to-Decoder Layer
        x = layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Decoder
        for n_filters in [256, 128, 64, 3]:
            x = decoder_block(x, n_filters=n_filters)

        return x

    def create_discriminator_graph(self):
        """
        Creates a discriminator graph in self.discriminator_graph
        """
        # TODO Task 4.1
        pass

    def reconstruction_loss(self, y_true, y_pred):
        """ 
        Creates the reconstruction loss between y_true and y_pred
        """
        # Task 2.2
        reconstruction_loss_value = tf.losses.mean_squared_error(y_true, y_pred)
        return reconstruction_loss_value

    def reconstruction_loss_overlap(self, y_true, y_pred):
        """ 
        Similar to reconstruction loss, but with predicted region overlapping
        with the input
        """
        # Task 2.3
        batch_size = tf.keras.backend.shape(y_pred)[0]
        weights = tf.tile(self.loss_weights, [batch_size, 1, 1, 1])
        reconstruction_loss_value = tf.losses.mean_squared_error(y_true, y_pred, weights)

        return reconstruction_loss_value
