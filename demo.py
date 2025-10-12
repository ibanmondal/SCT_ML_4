import cv2
import numpy as np
from model_utils import load_trained_model

model = load_trained_model('hand_gesture_model.h5')
gesture_labels = np.load('gesture_labels.npy', allow_pickle=True)
IMG_SIZE = (64, 64)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, IMG_SIZE)
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)
    gesture_index = np.argmax(prediction)
    gesture_name = gesture_labels[gesture_index]

    cv2.putText(frame, f'Gesture: {gesture_name}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
