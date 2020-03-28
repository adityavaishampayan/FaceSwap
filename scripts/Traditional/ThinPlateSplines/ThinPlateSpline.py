import numpy as np
import cv2
import math


def U(r):
    try:
        u = (r ** 2) * np.log(r ** 2)
    except:
        u = 0
    return u


def estimateParams(points2, points1_d):
    p = len(points2)

    K = np.zeros((p, p), np.float32)
    P = np.zeros((p, 3), np.float32)

    for i in range(p):
        for j in range(p):
            a = points2[i, :]
            b = points2[j, :]
            K[i, j] = U(np.linalg.norm((a - b)))

    P = np.hstack((points2, np.ones((p, 1))))

    A = np.hstack((P.transpose(), np.zeros((3, 3))))
    B = np.hstack((K, P))
    C = np.vstack((B, A))
    lamda = 0.0000001

    T = np.linalg.inv(C + lamda * np.identity(p + 3))
    target = np.concatenate((points1_d, [0, 0, 0]))
    params = np.matmul(T, target)

    return params


def thinPlateSpline(img1, img2, points1, points2, hull2):
    points1 = np.asarray(points1, np.int32)
    points2 = np.asarray(points2, np.int32)
    p = len(points1)

    r = cv2.boundingRect(np.float32([points2]))
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)

    points2_t = []

    for i in range(len(hull2)):
        points2_t.append(((hull2[i][0][0] - r[0]), (hull2[i][0][1] - r[1])))

    cv2.fillConvexPoly(mask, np.int32(points2_t), (1.0, 1.0, 1.0), 16, 0);

    x_params = estimateParams(points2, points1[:, 0])
    y_params = estimateParams(points2, points1[:, 1])

    a1_x = x_params[p + 2]
    ay_x = x_params[p + 1]
    ax_x = x_params[p]

    a1_y = y_params[p + 2]
    ay_y = y_params[p + 1]
    ax_y = y_params[p]

    warped_img = np.copy(mask)

    for i in range(warped_img.shape[1]):
        for j in range(warped_img.shape[0]):
            t = 0
            l = 0
            n = i + r[0]
            m = j + r[1]
            b = [n, m]
            for k in range(p):
                a = points2[k, :]
                t = t + x_params[k] * U(np.linalg.norm((a - b)))
                l = l + y_params[k] * U(np.linalg.norm((a - b)))

            x = a1_x + ax_x * n + ay_x * m + t
            y = a1_y + ax_y * n + ay_y * m + l

            x = int(x)
            y = int(y)
            x = min(max(x, 0), img1.shape[1] - 1)
            y = min(max(y, 0), img1.shape[0] - 1)

            warped_img[j, i] = img1[y, x, :]

    warped_img = warped_img * mask

    img2[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img2[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img2[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] + warped_img

    return img2