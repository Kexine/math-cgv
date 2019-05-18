from datetime import datetime
import glob
import numpy as np 
import os 
from PIL import Image
import skimage
import tensorflow as tf
from tensorflow import keras 

from model import Model
from trainer import Trainer 
from dataset import Dataset
from config import Config
import tensorflow.contrib.eager as tfe


def main():
    train_data_path = '../dataset/train'
    val_data_path = '../dataset/val'
    test_data_path = '../dataset/test'

    # Read the data
    train_dataset = Dataset(train_data_path)
    val_dataset   = Dataset(val_data_path)
    test_dataset  = Dataset(test_data_path)

    # Create dataset
    batch_size = Config.batch_size

    train_dataset.create_tf_dataset(batch_size)
    val_dataset.create_tf_dataset(batch_size)
    test_dataset.create_tf_dataset(batch_size)

    # Create the model
    model = Model([128, 128, 3])

    # Train the model
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset)
    trainer.train(n_epochs=Config.n_epochs)


if __name__ == '__main__':
    main()
