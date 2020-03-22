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

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


#Loading images
img = cv2.imread("bradley_cooper.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(gray)

# Loading Face landmarks detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



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
        # cv2.circle(img,(x,y),3,(0,0,255),1)
        # cv2.imshow("facial landmarks",img)
        # cv2.imwrite('facial_landmarks_1.png',img)

    points = np.array(landmark_points,np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(img,[convexhull],True,(255,0,0),2)
    cv2.fillConvexPoly(mask,convexhull,255)


    face_image_1 = cv2.bitwise_and(img,img,mask = mask)

    # Delaunay Triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles,dtype = np.int32)

    indexes_triangles = []

    for t in triangles:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

        cv2.line(img,pt1,pt2,(0,255,0),1)
        cv2.line(img,pt2,pt3,(0,255,0),1)
        cv2.line(img,pt3,pt1,(0,255,0),1)


# Face 2
faces2 = detector(img2_gray)
for face in faces2:
    landmarks = predictor(img2_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)

for triangle_index in indexes_triangles:
    pt1 = landmarks_points[triangle_index[0]]
    pt2 = landmarks_points[triangle_index[1]]
    pt3 = landmarks_points[triangle_index[2]]
    cv2.line(img2, pt1, pt2, (0, 0, 255), 2)
    cv2.line(img2, pt3, pt2, (0, 0, 255), 2)
    cv2.line(img2, pt1, pt3, (0, 0, 255), 2)


cv2.imshow("frame",img)
cv2.imwrite('frame_1.png',img)

cv2.imshow("face_image",face_image_1)
cv2.imwrite('face_image_1.png',face_image_1)


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
