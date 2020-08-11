import cv2
import os
import sys

# We are taking an argument for label and number of pictures
# noinspection PyBroadException
try:
    label_name = sys.argv[1]
    num_image = int(sys.argv[2])
except:
    print("\n!---Argument missing---!")
    print("Usage : python DataCollector.py <label-name> <number-picture>\n")
    exit(-1)

# Path to save the data
IMG_SAVE_PATH = 'DataSet'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

# Making the directory
# noinspection PyBroadException
try:
    os.mkdir(IMG_SAVE_PATH)
except:
    pass
# noinspection PyBroadException
try:
    os.mkdir(IMG_CLASS_PATH)
except:
    pass

# Initializing camera objects and variable
cap = cv2.VideoCapture(0)
start = False
count = 0

# Looping for collecting the data
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, None, fx=1.7, fy=1.5)

    if not ret:
        continue

    if count == num_image:
        break

    # Square on the frame
    cv2.rectangle(frame, (50, 150), (380, 500), (255, 255, 255), 2)

    # Collecting image
    if start:
        roi = frame[150:500, 50:380]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1

    # Showing text that collecting data
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count), (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    # Reading key from  keyboard
    key = cv2.waitKey(20)
    if key == ord('s'):
        start = not start

    if key == ord('q'):
        break

# Ya that's it time to stop
print("\n{} image's saved to {} directory".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
