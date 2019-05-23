import tensorflow as tf
import numpy as np
from tensorflow import keras
from config import Config
from dataset import Dataset


def reconstruction_loss_overlap(y_true, y_pred):
    loss_weights = np.ones((1, 64, 64, 3), dtype=np.float32) * 10.0
    loss_weights[:, 7:57, 7:57, :] = 1.0
    loss_weights = tf.convert_to_tensor(loss_weights)

    batch_size = tf.keras.backend.shape(y_pred)[0]
    weights = tf.tile(loss_weights, [batch_size, 1, 1, 1])
    reconstruction_loss_value = tf.losses.mean_squared_error(y_true, y_pred, weights)

    return reconstruction_loss_value


model_path = 'logs/20190519-000147/model_0115-0.287-0.299-.hdf5'
trained_model = keras.models.load_model(model_path, custom_objects={'reconstruction_loss_overlap': reconstruction_loss_overlap})

test_data_path = '../dataset/test'

# Read the data
test_dataset  = Dataset(test_data_path)

# Create dataset
batch_size = Config.batch_size

test_dataset.create_tf_dataset(batch_size)

session = keras.backend.get_session()
input_img_array, gt_img_array = session.run(test_dataset.dataset_tf.make_one_shot_iterator().get_next())

predictions = trained_model.predict(input_img_array, batch_size=batch_size)

print()