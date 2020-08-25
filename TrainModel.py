import cv2
import numpy as np
import os
from keras.models import Sequential
# from keras.layers import Dropout, MaxPooling2D, Conv2D, Dense, Flatten
from keras.layers import Dropout, Convolution2D, GlobalAveragePooling2D, Activation, MaxPooling2D

from keras.optimizers import Adam
from keras_squeezenet import SqueezeNet
from keras.callbacks import Callback
from keras.utils import np_utils

# nas net mobile import
# import keras
# from keras.models import Model, load_model
# from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D, GlobalAveragePooling2D, Activation

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
        Convolution2D(4, (3, 3), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])

    # Using nas net mobile
    # nas_net = keras.applications.NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # nas_net.trainable = False
    # x = nas_net.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(712, activation='relu')(x)
    # x = Dropout(0.4)(x)
    # final_layer = Dense(4, activation='softmax')(x)

    # fun_model = Model(inputs=nas_net.input, outputs=final_layer)

    # Using our own model
    # fun_model = Sequential([
    #     Conv2D(64, (3, 3), input_shape=(150, 150, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #
    #     Dropout(0.5),
    #     Flatten(),
    #     Dense(512, activation='relu'),
    #     Dense(4, activation='softmax')
    #
    # ])

    fun_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return fun_model


# Load images from directory
# train_data = []
# for directory in os.listdir(IMG_SAVE_PATH):
#     path = os.path.join(IMG_SAVE_PATH, directory)
#     if not os.path.isdir(path):
#         continue
#     for item in os.listdir(path):
#         # Avoiding the hidden files
#         if item.startswith("."):
#             continue
#         img = cv2.imread(os.path.join(path, item))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (227, 227))
#         train_data.append([img, directory])
#
# valid_data = []
# for directory in os.listdir(IMG_SAVE_PATH):
#     path = os.path.join(IMG_SAVE_PATH, directory)
#     if not os.path.isdir(path):
#         continue
#     for item in os.listdir(path):
#         # Avoiding the hidden files
#         if item.startswith("."):
#             continue
#         img = cv2.imread(os.path.join(path, item))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (227, 227))
#         valid_data.append([img, directory])
#
# # Assigning data to variable
# data, label = zip(*train_data)
# label = list(map(mapper, label))
# test_data, test_label = zip(*valid_data)
# test_label = list(map(mapper, test_label))
#
# # Little bit of encoding
# label = np_utils.to_categorical(label)
# test_label = np_utils.to_categorical(test_label)

# validation data gen
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_data = train_gen.flow_from_directory(
    IMG_SAVE_PATH,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

valid_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40
)
valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'test_data',
    target_size=(150, 150),
    batch_size=5,
    class_mode='categorical'
)
#
# print(train_data, valid_data)

# Creating the model
model = create_model()
callbacks = myCallbacks()

# Time to train
# model.fit(np.array(data), np.array(label), epochs=10, validation_data=(np.array(test_data), np.array(test_label)), validation_steps=5, verbose=1)
model.fit(train_data, steps_per_epoch=15, epochs=50, validation_data=valid_data, validation_steps=5, verbose=1)

# Saving the model for latter use
model.save("rock-paper-scissor-model.h5")
