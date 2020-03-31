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

# @file    Wrapper.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief the main file to run the face swap project

# importing opencv, dlib and imutils libraries
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
    predictor = dlib.shape_predictor('./conventional_method/shape_predictor_68_face_landmarks.dat')
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangles = detector(grayscale_image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facial_points = []
    no_of_faces = len(rectangles)
    for (i, rect) in enumerate(rectangles):
        
        shape = predictor(grayscale_image, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            facial_points.append((x, y))
            
    return no_of_faces, facial_points