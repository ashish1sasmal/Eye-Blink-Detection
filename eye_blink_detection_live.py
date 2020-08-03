import cv2
import numpy as np
import dlib
import sys
from imutils import face_utils
import imutils
from collections import OrderedDict
from scipy.spatial import distance as dist
from live_cam import live

cv2.namedWindow("Output",cv2.WINDOW_NORMAL)



EYES_IDXS = OrderedDict([

	("right_eye", (36, 42)),
	("left_eye", (42, 48)),

])

def detect_blink(eye,shape):
    x=0
    y=0
    if eye == "left":
        part = EYES_IDXS["left_eye"]
    else:
        part = EYES_IDXS["right_eye"]

    eye_blink = shape[part[0]:part[1]]
    print(eye_blink)
    d = (dist.euclidean(tuple(eye_blink[1]), tuple(eye_blink[5])) +
     dist.euclidean(tuple(eye_blink[2]), tuple(eye_blink[4])))/(2*dist.euclidean(tuple(eye_blink[0]), tuple(eye_blink[3])))
    blink = False
    if d<=0.2:
        blink = True
    return (d,eye_blink,blink)

detect = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    img = live()
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    rects = detect(gray,0)
    if rects:
        shape = None
        for (i,rect) in enumerate(rects):
            shape = pred(gray,rect)
            shape  = face_utils.shape_to_np(shape)

        re = detect_blink("right",shape)
        le = detect_blink("left",shape)

        print(le[0],re[0])
        for (x,y) in re[1]:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        for (x,y) in le[1]:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        if le[2] and re[2]:
            cv2.putText(img, "Both Blink", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        elif le[2]:
            cv2.putText(img, "Left Blink", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        elif re[2]:
            cv2.putText(img, "Right Blink", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Both Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)


    cv2.imshow("Output", img)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
