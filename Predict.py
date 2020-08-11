from keras.models import load_model
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

# Reading the file path as argument
try:
    file_path = sys.argv[1]
except:
    print("Specify a file to predict")
    print("Usage: python Predict.py <file_path>")
    exit(-1)

# Map
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissor",
    3: "none"
}


# Rev mapper function
def mapper(val):
    return REV_CLASS_MAP[val]


# Model loading
model = load_model('rock-paper-scissor-model.h5')

# Reformat the test image
img = cv2.imread(file_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# Showing the image
plt.imshow(img)
plt.show()

# Predicting
my_pred = model.predict(np.array([img]))
my_pred = mapper(np.argmax(my_pred[0]))

print("Prediction for image {} is -- {} --".format(file_path, my_pred))
