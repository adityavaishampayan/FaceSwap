# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2020 Aditya Vaishampayan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# @file    main.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief main file for traditional methods of face swapping

method = 'thin_plate_spline'


import sys

# noinspection PyBroadException
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except BaseException:
    pass

import cv2
import numpy as np
import dlib
import copy
from imutils import face_utils

from scripts.Traditional.forwardwarping.forward import for_warping
from scripts.Traditional.InverseWarping.inverse import inv_warping
from scripts.Traditional.ThinPlateSplines.ThinPlateSpline import *

def facial_landmark_detection(gray_img):
    """
    Function for detecting facial landmarks
    :param gray_img: grayscale image
    :return: facial landmark points
    """
    faces = detector(gray_img)
    for face in faces:
        landmarks = predictor(gray_img, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

    return landmarks_points

def extract_index_nparray(nparray):
    """
    A function to extract numpy array indexes
    :param nparray: numpy array
    :return:index
    """
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def delaunay_triangulation(convex_hull, landmarks_points):
    """
    A function to perform delaunday triangulation
    :param convex_hull: convex hull made from the facial landmark points
    :param landmarks_points: facial landmark points
    :return:
    """
    rect = cv2.boundingRect(convex_hull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    landmarks_points = np.array(landmarks_points,np.int32)
    idx_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((landmarks_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((landmarks_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((landmarks_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            idx_triangles.append(triangle)

    return idx_triangles


def features(img,detector,predictor,resize_val):
    #initialize facial detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray,(resize_val,resize_val))
    # img = cv2.resize(img,(resize_val,resize_val))
    rects = detector(gray,1)
    if np.shape(rects)[0] == 0:
        shape = 0
        feature_found = False
    else:
        feature_found = True
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for (i,rect) in enumerate(rects):
            # print('before shape')
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            # print('shape = ')
            # print(shape)

            (x,y,w,h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


            for (x,y) in shape:
                cv2.circle(img,(x,y),2,(0,0,255),-1)
    # plt.imshow(img)
    # plt.show()
    return img, shape, feature_found

if __name__ == '__main__':
########################################################

    #img_tar = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/TestSet/Rambo.jpg")
    #img_src = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/TestSet/Test1/frame16.jpg")

    # Reading Image 1
    img1 = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/images/bradley_cooper.jpg")
    # converting image 1 to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # making a mask of image 1
    mask = np.zeros_like(img1_gray)
########################################################
    # reading the image 2
    img2 = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/images/aditya.jpg")
    # converting image 2 to gray scale
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)
########################################################
    # Initialising the facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/aditya/FaceSwap/shape_predictor_68_face_landmarks.dat")
########################################################
    # FACE 1
    face1_points = facial_landmark_detection(img1_gray)
    f1_points = np.array(face1_points, np.int32)
    # convex hull for face 1
    convexhull1 = cv2.convexHull(f1_points)
    # drawing the convex hull
    # cv2.polylines(img1, [convexhull1], True, (255, 0, 0), 3)
    # obtaining the mask of face 1
    cv2.fillConvexPoly(mask, convexhull1, 255)
    # extracting the outline of face 1
    face_image_1 = cv2.bitwise_and(img1, img1, mask=mask)

########################################################
    # FACE 2
    face2_points = facial_landmark_detection(img2_gray)
    f2_points = np.array(face2_points, np.int32)
    convexhull2 = cv2.convexHull(f2_points)
    # drawing the convex hull
    #cv2.polylines(img2, [convexhull2], True, (255, 0, 0), 3)
    lines_space_mask = np.zeros_like(img1_gray)
    rect = cv2.boundingRect(convexhull1)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(face1_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangle1 = delaunay_triangulation(convexhull1,face1_points)
########################################################

    if method == 'forward_warp':
        for_warping(indexes_triangle1, img1, img2, face1_points, face2_points,
                                            lines_space_mask, img2_new_face)

        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
        cv2.imshow("seamlessclone", seamlessclone)

    if method == 'inverse_warp':
        inv_warping(indexes_triangle1, img1, img2, face1_points, face2_points, lines_space_mask, img2_new_face)

        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
        cv2.imshow("seamlessclone", seamlessclone)

    if method == 'thin_plate_spline':

        img_src = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/images/bradley_cooper.jpg")
        img_tar = cv2.imread("/home/aditya/Desktop/to_add/FaceSwap/images/aditya.jpg")

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("/home/aditya/Desktop/to_add/FaceSwap/shape_predictor_68_face_landmarks.dat")

        rects = detector(img_src, 1)
        img_source = copy.deepcopy(img_src)
        img_target = img_tar.copy()

        if len(rects) == 1:
            img_source = img_source[rects[0].top() - 40:rects[0].bottom() + 40,
                         rects[0].left() - 40:rects[0].right() + 40, :]

        elif len(rects) > 1:
            img_target = img_source[rects[1].top() - 40:rects[1].bottom() + 40,
                         rects[1].left() - 40:rects[1].right() + 40, :]
            img_source = img_source[rects[0].top() - 40:rects[0].bottom() + 40,
                         rects[0].left() - 40:rects[0].right() + 40, :]
        else:
            # cv2.imwrite('./result1/' + str(i) + '.jpg', img_src)
            # print("2 faces not detected")
            pass

        img1 = img_source.copy()
        img2 = img_target.copy()

        a = 200

        img1, points1, flag1 = features(img_source, detector, predictor, a)
        img2, points2, flag2 = features(img_target, detector, predictor, a)

        print(points1)

        output1 = swap(img_source.copy(), img_target.copy(), points1, points2)
        output1 = cv2.resize(output1, (
        (rects[0].right() + 40) - (rects[0].left() - 40), (rects[0].bottom() + 40) - (rects[0].top() - 40)))
        img_src[rects[0].top() - 40:rects[0].bottom() + 40, rects[0].left() - 40:rects[0].right() + 40, :] = output1
        if len(rects) > 1:
            output2 = swap(img_target.copy(), img_source.copy(), points2, points1)
            output2 = cv2.resize(output2, (
            (rects[1].right() + 40) - (rects[1].left() - 40), (rects[1].bottom() + 40) - (rects[1].top() - 40)))
            img_src[rects[1].top() - 40:rects[1].bottom() + 40, rects[1].left() - 40:rects[1].right() + 40, :] = output2

        cv2.imshow("img src", img_src)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
