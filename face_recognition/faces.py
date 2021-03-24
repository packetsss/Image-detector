import cv2
from cv2.data import *
from random import randrange

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
img = cv2.imread("1.jpg")
gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coord = face_cascade.detectMultiScale(gray_scale_img)

for f in face_coord:
    cv2.rectangle(img, f, (randrange(70, 256), randrange(70, 256), randrange(70, 256)), 5)

cv2.imshow("Face Detector", img)
cv2.waitKey()
