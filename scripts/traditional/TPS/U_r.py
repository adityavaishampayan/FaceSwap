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

# @file    U_r.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for obtaining the r**2 * log(r**2)

import numpy as np
import math


def U(r):
    """
    a function to obtain the product of r**2 and log of r**2
    Args:
        r:

    Returns: product of r**2 and log of r**2

    """
    u = (r**2)*np.log(r**2)
    if math.isnan(u):
        u = 0
    return u