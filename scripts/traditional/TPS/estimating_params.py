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

# @file    estimating_params.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for estimating the parameters of TPS for face warping


import numpy as np
from scripts.traditional.TPS.U_r import U


def param_estimation(face_points2, face_points1_1d):
    """
    A function to estimate the parameters of thin plate splines
    Args:
        face_points2: facial landmark points of the second face
        face_points1_1d: only the x or y coordinates of the points from face 1

    Returns: estimated thin plate spline parameters

    """
    p = len(face_points2)
    K = np.zeros((p, p), np.float32)
    P = np.hstack((face_points2, np.ones((p, 1))))

    for i in range(p):
        for j in range(p):
            b = face_points2[j, :]
            a = face_points2[i, :]

            K[i, j] = U(np.linalg.norm((a - b)))

    # obtaining the lower half of matrix by combining P.T and zeros
    alpha = np.hstack((P.transpose(), np.zeros((3, 3))))

    # obtaining the upper half of matrix by concatenating K matrix and P matrix
    beta = np.hstack((K, P))

    # obtaining the whole matrix by stacking the upper and lower matrices
    gamma = np.vstack((beta, alpha))

    # lambda is a value which is extremely close to zero
    lamda = 0.0000001

    # calculating the inverse
    matrix_inv = np.linalg.inv(gamma + lamda * np.identity(p + 3))

    # target points for the TPS
    v = np.concatenate((face_points1_1d, [0, 0, 0]))

    # obtaining the thin plate spline parameters after matrix multiplication
    tps_params = np.matmul(matrix_inv, v)

    return tps_params