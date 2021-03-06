# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited

import cv2
import numpy as np
from random import randrange
import mss
import mss.tools


class detector:
    def __init__(self):
        self.trained_face_data = cv2.CascadeClassifier('face_recognition/data/haarcascade_frontalface_default.xml')

    def screen_record(self):
        region = {'top': 0, 'left': 0, 'width': 960, 'height': 1080}
        while 1:
            cv2.namedWindow("Face Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Detector", 960, 1080)
            with mss.mss() as sct:
                img = sct.grab(region)

            frame = np.array(img)  # converting the image into numpy array representation
            gray_scale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coord = self.trained_face_data.detectMultiScale(gray_scale_img, scaleFactor=1.6)
            for f in face_coord:
                cv2.rectangle(frame, f, (randrange(70, 256), randrange(70, 256), randrange(70, 256)), 5)
            cv2.imshow('Face Detector', frame)  # display screen/frame being recorded
            if cv2.waitKey(1) == ord('q'):  # Wait for the user to press 'q' key to stop the recording
                break

        cv2.destroyAllWindows()  # destroy

    def phone_cam(self):
        webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        while 1:
            success_frame, frame = webcam.read()
            gray_scale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coord = self.trained_face_data.detectMultiScale(gray_scale_img)
            for f in face_coord:
                cv2.rectangle(frame, f, (randrange(70, 256), randrange(70, 256), randrange(70, 256)), 5)
            cv2.imshow("Face Detector", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows()

    def image(self):
        img = cv2.imread("face_recognition/1.jpg")
        gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_coord = self.trained_face_data.detectMultiScale(gray_scale_img)

        for f in face_coord:
            cv2.rectangle(img, f, (randrange(70, 256), randrange(70, 256), randrange(70, 256)), 5)

        cv2.imshow("Face Detector", img)
        cv2.waitKey()


detector().image()
