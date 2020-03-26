
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

import cv2
import numpy as np
import dlib
import time
import random

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):

    # The function gives each triangle as a 6 numbers vector, where each two are one of the triangle vertices.
    # i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
    triangleList = subdiv.getTriangleList()

    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)

# Draw a point
def draw_point(img, p, color ) :
    #cv2.circle( img, p, 2, color,cv2.CV_FILLED, cv2.CV_AA, 0 )
    cv2.circle(img, p, 3, (0, 0, 255), -1)

def points(gray):
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            return landmarks_points


# Draw voronoi diagram
def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        #cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0)

        cv2.fillConvexPoly(img, ifacet, color, 0);
        ifacets = np.array([ifacet])
        #cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)

        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, 0)
        #cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), 0)

if __name__ == '__main__':

    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    animate = True
    # reading the images
    img = cv2.imread("/home/aditya/FaceSwap/images/aditya.jpg")


    img_orig = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/aditya/FaceSwap/shape_predictor_68_face_landmarks.dat")

    #obtaining the facial points
    facial_points = points(img_gray)
    cv2.imshow("facial points", img)

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    subdiv.insert(facial_points)
    img_copy = img_orig.copy()
    # Draw delaunay triangles
    draw_delaunay(img_copy, subdiv, (255, 255, 255))
    #cv2.imshow(win_delaunay, img_copy)

    # Insert points into subdiv
    for p in facial_points:
        subdiv.insert(p)

        # Show animation
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 255, 255))
            cv2.imshow(win_delaunay, img_copy)
            #cv2.waitKey(1)

    # Draw delaunay triangles
    draw_delaunay(img, subdiv, (255, 255, 255))

    # Draw points
    for p in facial_points:
        draw_point(img, p, (0, 0, 255))

    #Allocate space for Voronoi Diagram
    img_voronoi = np.zeros(img.shape, dtype=img.dtype)

    # Draw Voronoi diagram
    draw_voronoi(img_voronoi, subdiv)


    cv2.imshow(win_delaunay, img)
    cv2.imshow(win_voronoi, img_voronoi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



