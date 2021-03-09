import numpy as np
import cv2
from mss import mss
from random import randrange


bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    gray_scale_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
    face_coord = trained_face_data.detectMultiScale(gray_scale_img)
    for f in face_coord:
        cv2.rectangle(sct_img, f, (randrange(70, 256), randrange(70, 256), randrange(70, 256)), 5)
    cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
