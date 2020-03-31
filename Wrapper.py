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
import argparse
import numpy as np
import imutils
import random
import cv2

from scripts.traditional.two_faces import swapping_two_faces
from scripts.traditional.facial_landmarks import facial_landmarks
from scripts.traditional.main import conventional_method

if __name__ == '__main__' :

	# argument parser
	parser = argparse.ArgumentParser(description='Project of face swappping')

	parser.add_argument('--mode', default=1, type=int,help='1- swap face in video with image, 2- swap two faces '
														   'within video')

	parser.add_argument('--method', default='tps', type=str,help=',prnet, tps, inv_warp, affine')

	parser.add_argument('--input', default="../TestSet/", type=str,help='input videos path')

	parser.add_argument('--video', default='Test1', type=str,help='name of the video')

	parser.add_argument('--face', default='Rambo', type=str,help='path to face to be swapped with the face in the '
																 'input video')

	Args = parser.parse_args()
	mode = Args.mode
	method = Args.method
	video_path = Args.input_path+Args.video+'.mp4'
	video = Args.video

	# start capturing the video frames
	cap = cv2.VideoCapture(video_path)

	# counting the number of frames
	no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("Number of frames in the video are: "+str(no_of_frames))

	# reading the frames
	ret,img = cap.read()

	# obtaining the size of the video frame
	height = img.shape[0]
	width = img.shape[1]

	# Defining video writing objects
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	out = cv2.VideoWriter('{}_Output_{}.avi'.format(method,video),fourcc, 15, (width,height))

	frame_count = 0

	if mode==1 :
		path_to_face = Args.input_path + Args.face + '.jpg'
		image_1 = cv2.imread(path_to_face)

		# obtaining the facial landmarks from the 1st image
		num_of_faces, face1_points = facial_landmarks(image_1)

		# if zero or more than one faces are detected
		if(num_of_faces!=1):
			print("Exiting because more than one faces have been detected")
	        exit()

		# read the second image
		ret, image_2 = cap.read()

		height = image_2.shape[0]
		width = image_2.shape[1]

	    while(cap.isOpened()):
	        frame_count = frame_count + 1
			ret, image_2 = cap.read()
	        if ret == True:

	            if method=='prnet':
	                print("PRNET moethod not accessible")
	            else:
					num_of_faces, face2_points = facial_landmarks(image_2)

					# by passing the while loop if no facesa re detected
	                if num_of_faces == 0:
						continue
	                else:
	                    print("Frame" + str(frame_count))

	                face_swap_result = conventional_method(image_1, image_2, face1_points, face2_points, method)

					cv2.imshow("The faces have been swapepd", face_swap_result)
					cv2.waitKey(0)
					out.write(face_swap_result)

	            if cv2.waitKey(1) & 0xff==ord('q'):
	                cv2.destroyAllWindows()
	                break
	        else:
	            exit()

	else:

		# this mode is when we need to swap two face in the image
		print("Two faces in the video will be swapped. The mode is: "+str(mode))

		while(cap.isOpened()):
			frame_count += 1
			ret,img = cap.read()
			if(ret==True):

				# rotating the image
			    img = imutils.rotate(img,180)

			    if(method=="prnet"):
					print("PRNET method not accesible")

				else:
					num_of_faces, points = swapping_two_faces(img)

					# if number of faces is not equal to two, skip the iteration
			        if(num_of_faces!=2):
						print("{} faces detected in frame {}".format(num_of_faces, frame_count))
						continue
			        else:
						face1_points = points[0]
						face2_points = points[1]

					# obtaining a temp output for passing it again
			        temporary_output = conventional_method(img, img, face1_points, face2_points, method)

					# passing the temp output through
			        two_face_swap_result = conventional_method(img, temporary_output, face2_points, face1_points,
															   method)

			    cv2.imshow("The faces have been swapped", two_face_swap_result)
			    cv2.waitKey(0)
			    out.write(output)
			    
			    if cv2.waitKey(1) & 0xff==ord('q'):
			        cv2.destroyAllWindows()
			        break
			else:
			    exit()