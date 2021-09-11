# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited

import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_data/face-trainner.yml")

labels = {"person_name": 1}
with open("trained_data/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

# Capture frame-by-frame
frame = cv2.imread("images/2.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)
for x, y, w, h in faces:
    roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
    roi_color = frame[y:y + h, x:x + w]

    # recognize? deep learned model predict keras tensorflow pytorch scikit learn
    id_, conf = recognizer.predict(roi_gray)
    if 4 <= conf <= 85:
        print(id_)
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

    # img_item = "7.png"
    # cv2.imwrite(img_item, roi_color)

    color = (255, 0, 0)  # BGR 0-255
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    # subitems = smile_cascade.detectMultiScale(roi_gray)
    # for (ex, ey, ew, eh) in subitems:
    #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# Display the resulting frame
cv2.imshow('frame', frame)
cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()
