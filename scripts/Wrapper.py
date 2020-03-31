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


from scripts.traditional.facial_landmarks import facial_landmarks
from scripts.traditional.main import conventional_method
from scripts.traditional.twoFaces import twoFaces

if __name__ == '__main__' :


	parser = argparse.ArgumentParser(description='Face Swapping')

	parser.add_argument('--input_path', default="../TestSet/", type=str,help='path to the input')
	parser.add_argument('--face', default='Rambo', type=str,help='path to face')
	parser.add_argument('--video', default='Test1', type=str,help='path to the input video')
	parser.add_argument('--method', default='tps', type=str,help='affine, tri, tps, prnet')
	parser.add_argument('--resize', default=False, type=bool,help='True or False input resizing')
	parser.add_argument('--mode', default=1, type=int,help='1- swap face in video with image, 2- swap two faces within video')

	Args = parser.parse_args()
	video_path = Args.input_path+Args.video+'.mp4'
	video = Args.video
	method = Args.method
	resize = Args.resize
	mode = Args.mode

	w = 320

	cap = cv2.VideoCapture(video_path)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("No. of frames = "+str(length))

	ret,img = cap.read()

	if resize:
	    img = imutils.resize(img,width = w)
	height = img.shape[0]
	width = img.shape[1]

	# Defining video writing objects
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	out = cv2.VideoWriter('{}_Output_{}.avi'.format(method,video),fourcc, 15, (width,height))

	count = 0

	if(mode==1):

	    face_path = Args.input_path+Args.face+'.jpg'
	    img1 = cv2.imread(face_path)
	    if resize:
	        img1 = imutils.resize(img1,width = w)
	    
	    faces_num,points1 = facial_landmarks(img1)
	    if(faces_num!=1):
	        print("Exiting because more than one faces have been detected")
	        exit()
	      
	    ret,img2 = cap.read()
	    if resize:
	        img2 = imutils.resize(img2,width = w)
	    height = img2.shape[0]
	    width = img2.shape[1]

	    while(cap.isOpened()):
	        count += 1
	        ret,img2 = cap.read()
	        if(ret==True):

	            if resize:
	                img2 = imutils.resize(img2,width = w)
	            if(method=="prnet"):
	                print("PRNET moethod not accessible")
	            else:
	                faces_num,points2 = facial_landmarks(img2)
	                if(faces_num==0):
	                    continue
	                else:
	                    print("Frame"+str(count))
	                output = conventional_method(img1, img2, points1, points2, method)
					cv2.imshow("The faces have been swapepd", output)
					cv2.waitKey(100)
					out.write(output)

	            if cv2.waitKey(1) & 0xff==ord('q'):
	                cv2.destroyAllWindows()
	                break
	        else:
	            exit()

	else:
		
		print("Mode "+str(mode))
		while(cap.isOpened()):
			count += 1
			ret,img = cap.read()
			if(ret==True):
			    img = imutils.rotate(img,180)
			    if resize:
			        img = imutils.resize(img,width = w)
			    if(method=="prnet"):
					print("PRNET method not accesible")
				else:
			        faces_num,points = twoFaces(img)
			        if(faces_num!=2):
						print("{} faces detected in frame {}".format(faces_num,count))
						continue
			        else:
						points1 = points[0]
						points2 = points[1]
						print("Frame"+str(count))
			        temp = conventional_method(img, img, points1, points2, method)
			        output = conventional_method(img, temp, points2, points1, method)

			    cv2.imshow("The faces have been swapped", output)
			    cv2.waitKey(100)
			    out.write(output)
			    
			    if cv2.waitKey(1) & 0xff==ord('q'):
			        cv2.destroyAllWindows()
			        break
			else:
			    exit()