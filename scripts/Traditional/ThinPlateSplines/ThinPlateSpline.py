import numpy as np
import cv2
import sys
import math

def fxy(pt1,pts2,weights):
    K = np.zeros([pts2.shape[0],1])
    for i in range(pts2.shape[0]):
        K[i] = U(np.linalg.norm((pts2[i]-pt1),ord =2)+sys.float_info.epsilon)
    f = weights[-1] + weights[-3]*pt1[0] +weights[-2]*pt1[1]+np.matmul(K.T,weights[0:-3])
    return f


def warp_tps(img_source,img_target,points1,points2,weights_x,weights_y,mask):
    xy1_min = np.float32([min(points1[:,0]), min(points1[:,1])])
    xy1_max = np.float32([max(points1[:,0]),max(points1[:,1])])

    xy2_min = np.float32([min(points2[:,0]),min(points2[:,1])])
    xy2_max = np.float32([max(points2[:,0]),max(points2[:,1])])

    x = np.arange(xy1_min[0],xy1_max[0]).astype(int)
    y = np.arange(xy1_min[1],xy1_max[1]).astype(int)

    X,Y = np.mgrid[x[0]:x[-1]+1,y[0]:y[-1]+1]

    # X,Y = np.mgrid[0:src_shape[2],0:src_shape[3]]
    pts_src = np.vstack((X.ravel(),Y.ravel()))
    xy = pts_src.T
    u = np.zeros_like(xy[:,0])
    v = np.zeros_like(xy[:,0])
    # print(u.shape)
    # print(v.shape)
    for i in range(xy.shape[0]):
        u[i] = fxy(xy[i,:],points1,weights_x)
    u[u<xy2_min[0]]=xy2_min[0]
    u[u>xy2_max[0]]=xy2_max[0]
    for j in range(xy.shape[0]):
        v[j] = fxy(xy[j,:],points1,weights_y)
    v[v<xy2_min[1]]=xy2_min[1]
    v[v>xy2_max[1]]=xy2_max[1]
#     print(u.shape)
#     print(img_source.shape)
    warped_img = img_source.copy()
    mask_warped_img = np.zeros_like(warped_img[:,:,0])
    for a in range(1, u.shape[0]):
        try:
    #     for b in range(v.shape[0]):
    #     warped_img[xy[a,1],xy[a,0],:] = warped_src_face[v[a],u[a],:]

            if mask[v[a],u[a]]>0:
                warped_img[xy[a,1],xy[a,0],:] = img_target[v[a],u[a],:]
                mask_warped_img[xy[a,1],xy[a,0]] = 255
        except:
            pass
    # plt.imshow(warped_img)
    # plt.show()
    return warped_img, mask_warped_img

def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask

def U(r):
    return (r**2)*(math.log(r**2))

def TPS_generate(source,target):
    P = np.append(source,np.ones([source.shape[0],1]),axis=1)
    P_Trans = P.T
    Z = np.zeros([3,3])
    K = np.zeros([source.shape[0],source.shape[0]])
    for p in range(source.shape[0]):
        K[p] = [U(np.linalg.norm((-source[p]+source[i]),ord =2)+sys.float_info.epsilon) for i in range(source.shape[0])]

    M = np.vstack([np.hstack([K,P]),np.hstack([P_Trans,Z])])
    lam = 200
    I = np.identity(M.shape[0])
    L = M+lam*I
    L_inv = np.linalg.inv(L)
    V = np.concatenate([np.array(target),np.zeros([3,])])
    V.resize(V.shape[0],1)
    weights = np.matmul(L_inv,V)
    return weights,K

def swap(img_source,img_target,points1,points2):
    weights_x,K = TPS_generate(points1,points2[:,0])
    weights_y,K = TPS_generate(points1,points2[:,1])
    # plt.imshow(K)

    w, h = img_target.shape[:2]
    # ## Mask for blending
    mask = mask_from_points((w, h), points2)
    # plt.imshow(mask)
    # mask.shape

    warped_img, mask_warped_img = warp_tps(img_source,img_target,points1,points2,weights_x,weights_y,mask)
    # plt.imshow(warped_img)
    # plt.imshow(mask_warped_img)
    # mask_warped_img.shape


    ##Poisson Blending
    r = cv2.boundingRect(mask_warped_img)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_img.copy(), img_source, mask_warped_img, center, cv2.NORMAL_CLONE)
    return output
