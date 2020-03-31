# importing inbuilt libraries
import os
import sys

# importing numpy, opencv, scipy and argparse
import math
from scipy.interpolate import interp2d
import argparse
import numpy as np
import imutils
import random
import cv2



import numpy as np
from scipy.interpolate import interp2d
import cv2


def triangulationWarping(src, srcTri, dstTri, size, epsilon=0.1):
    """
    this function performs inverse warping using barycentric coordinates
    Args:
        src: source image
        srcTri: source triangle obtained after delaunay triangulation
        dstTri: destination triangle having same indices as source triangle is second face
        size: size of the rectangle
        epsilon: a multiplication factor

    Returns: triangle one inverse warped on triangle 2

    """
    dst_tri = dstTri
    src_tri = srcTri

    # coordinates of the bounding rectangle
    x, y, w, h = cv2.boundingRect(np.float32([dst_tri]))

    # obtain the height and width of the rectangle
    rect_left = x
    rect_right = x + w
    rect_top = y
    rect_bottom = y + h

    # obtaining the destination matrix
    matrix_dst = np.linalg.inv([[dst_tri[0][0], dst_tri[1][0], dst_tri[2][0]],
                               [dst_tri[0][1], dst_tri[1][1], dst_tri[2][1]],
                                [1, 1, 1]])

    grid = np.mgrid[rect_left:rect_right, rect_top:rect_bottom].reshape(2, -1)
    # grid 2xN
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    # grid 3xN
    barycentric_coords = np.dot(matrix_dst, grid)

    dst_tri = []
    b = np.all(barycentric_coords > -epsilon, axis=0)
    a = np.all(barycentric_coords < 1 + epsilon, axis=0)
    for i in range(len(a)):
        dst_tri.append(a[i] and b[i])
    dst_y = []
    dst_x = []
    for i in range(len(dst_tri)):
        if dst_tri[i]:
            dst_y.append(i % h)
            dst_x.append(i / h)

    barycentric_coords = barycentric_coords[:, np.all(-epsilon < barycentric_coords, axis=0)]
    barycentric_coords = barycentric_coords[:, np.all(barycentric_coords < 1 + epsilon, axis=0)]

    src_matrix = np.matrix([[src_tri[0][0], src_tri[1][0], src_tri[2][0]],
                            [src_tri[0][1], src_tri[1][1], src_tri[2][1]],
                            [1, 1, 1]])

    # matrix multiplication of source matrix and barycentric coordinates
    pts = np.matmul(src_matrix, barycentric_coords)

    # converting values to homogenous coordinates
    xA = pts[0, :] / pts[2, :]
    yA = pts[1, :] / pts[2, :]

    dst = np.zeros((size[1], size[0], 3), np.uint8)

    #  copy back the value of the pixel at (xA,yA) to the target location.
    #  Using scipy.interpolate.interp2d to perform this operation.
    i = 0
    for x, y in zip(xA.flat, yA.flat):
        y_values = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)
        x_values = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)

        g = src[:, :, 1]
        fg = interp2d(x_values, y_values, g, kind='cubic')
        green = fg(x, y)[0]

        b = src[:, :, 0]
        fb = interp2d(x_values, y_values, b, kind='cubic')
        blue = fb(x, y)[0]

        r = src[:, :, 2]
        fr = interp2d(x_values, y_values, r, kind='cubic')
        red = fr(x, y)[0]

        try:
            dst[dst_y[i], dst_x[i]] = (blue, green, red)
        except:
            pass
        i = i + 1

    return dst


def affine_warping(src, src_tri, dst_tri, size):
    """
    a function to perform affine warping
    Args:
        src: source image
        src_tri: source traingle
        dst_tri: destination triangle
        size: the height and width

    Returns: forward warped triangle

    """
    warpMat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def triangle_warping(img1, img2, t1, t2, method):
    """
    a function to perform direct or inverse triangle warping
    Args:
        img1: image 1
        img2: image 2
        t1: traingle 1
        t2: traingle 2
        method: affine warping (forward warping) or inverse warping

    Returns: None

    """
    # Find bounding rectangle for each triangle
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([t1]))
    x2, y2, w2, h2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - x1), (t1[i][1] - y1)))
        t2Rect.append(((t2[i][0] - x2), (t2[i][1] - y2)))

    # Get mask by filling triangle
    mask = np.zeros((h2, w2, 3), dtype=np.float32)

    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[y1:y1 + h1, x1:x1 + w1]
    img2Rect = np.zeros((h2, w2), dtype=img1Rect.dtype)

    size = (w2, h2)

    if method == "affine":
        img2Rect = affine_warping(img1Rect, t1Rect, t2Rect, size)
    else:
        img2Rect = triangulationWarping(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    a = (1.0, 1.0, 1.0) - mask

    # Copy triangular region of the rectangular patch to the output image
    img2[y2:y2 + h2, x2:x2 + w2] = img2[y2:y2 + h2, x2:x2 + w2] * ((1.0, 1.0, 1.0) - mask)
    img2[y2:y2 + h2, x2:x2 + w2] = img2[y2:y2 + h2, x2:x2 + w2] + img2Rect



def triangulation(img1, img2, img1Warped, hull1, hull2, method):
    """
    a function to implement the forward or inverse triangulation for face swapping
    Args:
        img1: image 1
        img2: image 2
        img1Warped: image 1 warped
        hull1: convex hull of face 1
        hull2: convex hull of face 2
        method: forward or inverse warping

    Returns: face swapped on the second image

    """
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = delaunay_triangle_calculation(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        triangle_warping(img1, img1Warped, t1, t2, method)



    return img1Warped


import cv2
import dlib


def delaunay_triangle_calculation(rect, points):
    """
    a function to perform delaunay triangulation
    Args:
        rect: bounding rectangle
        points: facial landmark points

    Returns: a list of delaunay triangles

    """
    # creating the subdiv class
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv class
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()

    delaunay_tri = []
    pt = []

    for t in triangle_list:
        pt.append((t[0], t[1]))
        pt1 = (t[0], t[1])

        pt.append((t[2], t[3]))
        pt2 = (t[2], t[3])

        pt.append((t[4], t[5]))
        pt3 = (t[4], t[5])

        if in_rectangle(rect, pt1) and in_rectangle(rect, pt2) and in_rectangle(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)

            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunay_tri


def in_rectangle(rect, point):
    """
    to check if a point is contained in the rectangle or not
    Args:
        rect: rectangle
        point: points to be checked

    Returns: a boolean value, true or false. If inside the rectangle it returns True

    """
    if point[0] < rect[0]:
        return False

    elif point[1] < rect[1]:
        return False

    elif point[0] > rect[0] + rect[2]:
        return False

    elif point[1] > rect[1] + rect[3]:
        return False

    return True


import cv2
import dlib
from imutils import face_utils


def facial_landmarks(img):
    """
    a function to perform facial landmark detection
    Args:
        img: the image on which facial landmark detection needs to be performed

    Returns: num of faces detected as well as the facial landmarks

    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/aditya/Desktop/stuff/ComputerVision-CMSC733/FaceSwap/scripts/traditional/shape_predictor_68_face_landmarks.dat')
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangles = detector(grayscale_image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facial_points = []
    no_of_faces = len(rectangles)
    for (i, rect) in enumerate(rectangles):

        shape = predictor(grayscale_image, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            facial_points.append((x, y))

    return no_of_faces, facial_points


import numpy as np
import cv2

def conventional_method(img1, img2, points1, points2, method):
    """

    Args:
        img1: image 1
        img2: image 2
        points1: facial landmark points for face 1
        points2: facial landmark points for face 2
        method: thin plate spline or triangulation

    Returns: face swapped output

    """
    img1Warped = np.copy(img2)

    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull2.append(points2[int(hullIndex[i])])
        hull1.append(points1[int(hullIndex[i])])

    if(method=="tps"):

        img1Warped = thinPlateSpline(img1,img1Warped,points1,points2,hull2)

    elif(method=="affine" or method=="tri"):

        img1Warped = triangulation(img1, img2,img1Warped,hull1,hull2,method)

    cv2.imshow("without blend", img1Warped)
    cv2.waitKey(0)

    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    return output


img1 = cv2.imread('/home/aditya/Desktop/stuff/ComputerVision-CMSC733/FaceSwap/TestSet/bradley_cooper.jpg')
faces_num, points1 = facial_landmarks(img1)

img2 = cv2.imread('/home/aditya/Desktop/stuff/ComputerVision-CMSC733/FaceSwap/TestSet/aditya.jpg')
faces_num, points2 = facial_landmarks(img2)

method = 'affine'

output = conventional_method(img1, img2, points1, points2, method)
cv2.imshow("The faces have been swapepd", output)
cv2.waitKey(0)
cv2.destroyAllWindows()