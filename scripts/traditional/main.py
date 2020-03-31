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
# @brief fa wrapper file for calling traditional cv methods

from scripts.traditional.TPS.thin_plate_splines import thinPlateSpline
from scripts.traditional.Warping.triangulation import triangulation
from scripts.traditional.facial_landmarks import *

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

        img1Warped = triangulation(img1,img2,img1Warped,hull1,hull2,method)

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


