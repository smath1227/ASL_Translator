import cv2
import numpy as np
import pickle
from glob import glob
from keras import Sequential
from keras import optimizers
from keras import utils
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.src.layers import Dense
from keras.src.optimizers import Adam

def get_num_signs():
    return len(glob("signs/*"))

def get_img_size():
    img = cv2.imread('signs/1/1.jpg', 0)
    return img.shape

image_x, image_y = get_img_size()
filepath = "cnn_model.keras"


def cnn_model():
    num_signs = get_num_signs()
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape = (image_x, image_y, 1), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
    model.add(Conv2D(32, (3, 3), activation= "relu"))
    model.add(MaxPooling2D(pool_size = (3, 3), padding = "same"))
    model.add(Conv2D(64, (5, 5), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (5, 5), padding = "same"))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_signs, activation = "softmax"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    checkpoint1 = ModelCheckpoint(filepath, monitor = "val_accuracy", verbose = 1, save_best_only = True, mode = "max")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    callbacks_list = [checkpoint1, reduce_lr]

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

    train_labels = np.clip(train_labels, 0, get_num_signs()-1)
    val_labels = np.clip(val_labels, 0, get_num_signs()-1)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    train_labels = utils.to_categorical(train_labels, num_classes = get_num_signs())
    val_labels = utils.to_categorical(val_labels, num_classes = get_num_signs())

    model, callbacks_list = cnn_model()
    model.summary()
    model.fit(train_images, train_labels, validation_data = (val_images, val_labels), epochs = 50, batch_size = 128, callbacks = callbacks_list)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    model.save(filepath)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))


train()
