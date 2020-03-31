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

# @file    delaunayTriangulation.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for performing delaunay triangulation based on the facial landmarks

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
            index = []

            # get 68 face points by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    alpha = abs(pt[j][0] - points[k][0])
                    beta = abs(pt[j][1] - points[k][1])
                    if alpha < 1.0 and beta < 1.0:
                        index.append(k)

            if len(index) == 3:
                delaunay_tri.append((index[0], index[1], index[2]))

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
