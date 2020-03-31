import cv2
import numpy as np

from scripts.traditional.Warping.delaunayTriangulation import *
from scripts.traditional.Warping.barycentric import *


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
    # Offset points by left top corner of the respective rectangles
    triangle1_rect = []
    triangle2_rect = []

    # Find bounding rectangle for each triangle
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([t1]))
    x2, y2, w2, h2 = cv2.boundingRect(np.float32([t2]))

    for i in range(0, 3):
        triangle1_rect.append(((t1[i][0] - x1), (t1[i][1] - y1)))
        triangle2_rect.append(((t2[i][0] - x2), (t2[i][1] - y2)))

    # Get mask by filling triangle
    mask = np.zeros((h2, w2, 3), dtype=np.float32)

    cv2.fillConvexPoly(mask, np.int32(triangle2_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    image1_rect = img1[y1:y1 + h1, x1:x1 + w1]

    size = (w2, h2)

    if method == "affine":
        image2_rect = np.zeros((h2, w2), dtype=image1_rect.dtype)
        image2_rect = affine_warping(image1_rect, triangle1_rect, triangle2_rect, size)
    else:
        image2_rect = np.zeros((h2, w2), dtype=image1_rect.dtype)
        image2_rect = triangulation_warping(image1_rect, triangle1_rect, triangle2_rect, size)

    image2_rect = image2_rect * mask

    a = (1.0, 1.0, 1.0) - mask

    # Copy triangular region of the rectangular patch to the output image
    img2[y2:y2+h2, x2:x2+w2] = img2[y2:y2+h2, x2:x2+w2] * ((1.0, 1.0, 1.0) - mask)
    img2[y2:y2+h2, x2:x2+w2] = img2[y2:y2+h2, x2:x2+w2] + image2_rect


def triangulation(img1, img2, image1_warped, convex_hull1, convex_hull2, method):
    """
    a function to implement the forward or inverse triangulation for face swapping
    Args:
        img1: image 1
        img2: image 2
        image1_warped: image 1 warped
        convex_hull1: convex hull of face 1
        convex_hull2: convex hull of face 2
        method: forward or inverse warping

    Returns: face swapped on the second image

    """
    img2_size = img2.shape
    rect = (0, 0, img2_size[1], img2_size[0])
    dt = delaunay_triangle_calculation(rect, convex_hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        face1_triangles = []
        face2_triangles = []
        
        for j in range(0, 3):
            face1_triangles.append(convex_hull1[dt[i][j]])
            face2_triangles.append(convex_hull2[dt[i][j]])
        
        triangle_warping(img1, image1_warped, face1_triangles, face2_triangles, method)

    return image1_warped