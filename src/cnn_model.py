import cv2
import numpy as np
import pickle
from glob import glob
from keras import Sequential
from keras import optimizers
from keras import utils
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.src.layers import Dense


def get_num_signs():
    return len(glob("gesture/*"))

def get_img_size():
    img = cv2.imread('gesture/1/100.jpg', 0)
    return img.shape

image_x, image_y = get_img_size()

def cnn_model():
    num_signs = get_num_signs()
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape = (image_x, image_y, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))
    model.add(Conv2D(32, (3, 3), activation= "relu"))
    model.add(MaxPooling2D(pool_size = (3, 3), strides  = (3, 3), padding = "same"))
    model.add(Conv2D(64, (5, 5), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (5, 5), strides = (5, 5), padding = "same"))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_signs, activation = "softmax"))

    stochastic = optimizers.SGD(lr = 1e-2)
    model.compile(loss = "categorical_crossentropy", optimizer = stochastic, metrics = ["accuracy"])
    filepath = "cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor = "val_acc", verbose = 1, save_best_only = True, mode = "max")
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype = np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype = np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    train_labels = utils.to_categorical(train_labels)
    val_labels = utils.to_categorical(val_labels)
