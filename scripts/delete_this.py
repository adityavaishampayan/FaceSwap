import cv2
import numpy as np
import dlib

img = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/TestSet/Rambo.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/TestSet/Scarlett.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(img_gray)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/aditya/Desktop/to_add/FaceSwap/shape_predictor_68_face_landmarks.dat")
faces = detector(img_gray)
# Face 1
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow("landmarks", img)

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)
        cv2.line(img, pt2, pt3, (0, 255, 0), 1)
        cv2.line(img, pt1, pt3, (0, 255, 0), 1)

# Face 2
faces2 = detector(img2_gray)
for face in faces2:
    landmarks = predictor(img2_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)

    # Triangulation of the second face, from the first face delaunay triangulation
    for triangle_index in indexes_triangles:
        pt1 = landmarks_points[triangle_index[0]]
        pt2 = landmarks_points[triangle_index[1]]
        pt3 = landmarks_points[triangle_index[2]]

        cv2.line(img2, pt1, pt2, (0, 255, 0), 1)
        cv2.line(img2, pt3, pt2, (0, 255, 0), 1)
        cv2.line(img2, pt1, pt3, (0, 255, 0), 1)

cv2.imshow("Image 1", img)
cv2.imshow("Face image 1", face_image_1)
cv2.imshow("image2", img2)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()