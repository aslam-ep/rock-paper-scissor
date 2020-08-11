from keras.models import load_model
import cv2
import numpy as np
from random import choice

# Map
REV_CLASS_MAP = {
    0: 'rock',
    1: 'paper',
    2: 'scissor',
    3: 'none'
}


# Mapper function
def mapper(val):
    return REV_CLASS_MAP[val]


# Calculate winner based on move
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    elif move1 == "paper":
        if move2 == "rock":
            return "User"
        elif move2 == "scissor":
            return "Computer"
    elif move1 == "scissor":
        if move2 == "rock":
            return "Computer"
        elif move2 == "paper":
            return "User"
    elif move1 == "rock":
        if move2 == "paper":
            return "Computer"
        elif move2 == "scissor":
            return "User"


# Loading the model
model = load_model("rock-paper-scissor-model.h5")

# Open-cv starting
cap = cv2.VideoCapture(0)
prev_move = None

# Loop
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, None, fx=1.85, fy=1.5)

    if not ret:
        continue

    # First Rectangle for user
    cv2.rectangle(frame, (50, 150), (380, 500), (0, 0, 255), 2)

    # Second Rectangle for computer
    cv2.rectangle(frame, (720, 150), (1040, 500), (255, 0, 0), 2)

    # Extracting the data
    roi = frame[50:380, 150:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    # Predicting the data
    pred = model.predict(np.array([img]))
    user_move = mapper(np.argmax(pred[0]))

    # Calculating the winner
    if prev_move != user_move:
        if user_move == 'none' or prev_move == user_move:
            computer_move = "none"
            winner = "Waiting..."
        else:
            computer_move = choice(['rock', 'paper', 'scissor'])
            winner = calculate_winner(user_move, computer_move)

    prev_move = user_move

    # At final display the information
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "Your Move : " + user_move, (40, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer Move : " + computer_move, (710, 50), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner : " + winner, (400, 600), font, 2, (0, 255, 0), 4, cv2.LINE_AA)

    # Icon for computer
    if computer_move != "none":
        icon = cv2.imread('Icon/{}.png'.format(computer_move))
        icon = cv2.resize(icon, (318, 348))
        frame[151:499, 721:1039] = icon

    cv2.imshow("Rock Paper Scissor", frame)

    # Reading keyboard entry
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
