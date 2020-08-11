import cv2
import numpy as np
import os
from keras.models import Sequential
# from keras.layers import Dropout, MaxPooling2D, Conv2D, Dense, Flatten
from keras.layers import Dropout, Convolution2D, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam
from keras_squeezenet import SqueezeNet
from keras.callbacks import Callback
from keras.utils import np_utils

# Variables
IMG_SAVE_PATH = 'DataSet'
CLASS_MAP = {
    'rock': 0,
    'paper': 1,
    'scissor': 2,
    'none': 3
}
NUM_CLASSES = len(CLASS_MAP)


# Mapper
def mapper(val):
    return CLASS_MAP[val]


# myCallbacks class
class myCallbacks(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print("\n !-- Accuracy level Reached - Stopping training --!\n")
            self.model.stop_training = True


# Model creation
def create_model():
    # Using already created model
    fun_model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(4, (1, 1), padding='valid'),
        Activation('relu'),
        Dropout(0.5),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])

    # Using our own model
    # fun_model = Sequential([
    #     Conv2D(64, (3, 3), input_shape=(227, 227, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dense(4, activation='softmax')
    #
    # ])

    fun_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return fun_model


# Load images from directory
train_data = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH,directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # Avoiding the hidden files
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path,item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        train_data.append([img, directory])

# Assigning data to variable
data, label = zip(*train_data)
label = list(map(mapper, label))

# Little bit of encoding
label = np_utils.to_categorical(label)

# Creating the model
model = create_model()
callbacks = myCallbacks()

# Time to train
model.fit(np.array(data),np.array(label), epochs=15, callbacks=[callbacks])

# Saving the model for latter use
model.save("rock-paper-scissor-model.h5")
