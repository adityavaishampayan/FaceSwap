import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


img = cv2.imread("aditya.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(gray)

faces = detector(gray)
for face in faces:
    landmark_points = []
    x1 = face.left()
    y1 = face.top()

    x2 = face.right()
    y2 = face.bottom()

    landmarks = predictor(gray,face)

    for n in range(0,68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x,y))
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        cv2.imshow("facial landmarks",img)
        cv2.imwrite('facial_landmarks.png',img)

    points = np.array(landmark_points,np.int32)

    convexhull = cv2.convexHull(points)

    cv2.polylines(img,[convexhull],True,(0,255,0),2)

    cv2.fillConvexPoly(mask,convexhull,255)


    face_image_1 = cv2.bitwise_and(img,img,mask = mask)

cv2.imshow("frame",img)
cv2.imwrite('frame.png',img)

cv2.imshow("face_image",face_image_1)
cv2.imwrite('face_image.png',face_image_1)


cv2.waitKey(0)
cv2.destroyAllWindows()




# cap = cv2.VideoCapture(0)
#
# while True:
#     _,frame = cap.read()
#
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#
#     faces = detector(frame)
#
#     for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#
#         x2 = face.right()
#         y2 = face.bottom()
#
#         landmarks = predictor(gray,face)
#
#         for n in range(0,68):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y
#
#             cv2.circle(frame,(x,y),3,(0,0,255),-1)
#
#     cv2.imshow("frame",frame)
#
#     key = cv2.waitKey(1)
#
#     if key == 27:
#         break
