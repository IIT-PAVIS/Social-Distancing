#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" 
	SocialDistancing: Code for estimating the distance between people and show warning if they are standing too close to each other.
   
	IIT : Istituto italiano di tecnologia
	Pattern Analysis and Computer Vision (PAVIS) research line

	Description: Social-Distancing is an open-source project for automatically estimating interpersonal distance from uncalibrated RGB
	cameras. The software can be freely used for any non-commercial applications to assess compliance with safe distances. Given a frame
	captured from a scene, the algorithm first detects visible people in the scene using an off-the-shelf body pose detector and 
	estimates the height of the people through measuring the distance from their body joints. In the second step, the algorithm estimates
	an area of one meter around all the detected people. This distance is roughly estimated proportional to a typical human body height
	of 160 cm and can be used to draw a circle centered in human position in the scene. In the third step, the Homography of the scene
	is estimated given two parameters which essentially map the rectangular bird’s view model for the scene to the trapezoidal perspective
	view of the scene. These two parameters need to be manually tuned to estimate best the scene perspective. According to the Homography
	matrix, the safe circular distance for each person is converted to ellipsoids in perspective view. The people are considered to be
	staying in safe distance from each other if their ellipsoids do not collide. Conversely, if ellipsoids of two people collide, those
	people are considered as being in risk and their ellipsoids will be shown in red.

	Usage Example:  
		$ python socialDistancing.py --image_in [path to the input image] --image_out [path to the result image] --horizontal_ratio 0.7 --vertical_ratio 0.7

	Tested on ShanghaiTech [1] dataset.
	[1] Zhang, Yingying, et al. "Single-image crowd counting via multi-column convolutional neural network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.


	Disclaimer:
	The information and content provided by this application is for information purposes only. 
	You hereby agree that you shall not make any health or medical related decision based in whole 
	or in part on anything contained within the application without consulting your personal doctor.

	The software is provided "as is", without warranty of any kind, express or implied, 
	including but not limited to the warranties of merchantability, 
	fitness for a particular purpose and noninfringement. In no event shall the authors, 
	PAVIS or IIT be liable for any claim, damages or other liability, whether in an action of contract, 
	tort or otherwise, arising from, out of or in connection with the software 
	or the use or other dealings in the software.

	LICENSE:
	This project is licensed under the terms of the MIT license.
	This project incorporates material from the projects listed below (collectively, "Third Party Code").  
	This Third Party Code is licensed to you under their original license terms.  
	We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.

	The software can be freely used for any non-commercial applications and it is useful
	for maintaining the safe social distance among people in pandemics. The code is open and can be 
	improved with your support, please contact us at socialdistancig@iit.it if you will to help us.
"""

import numpy as np
import cv2
import math
import itertools
import sys
import os
import argparse

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON`'
              'in CMake and have this Python script in the right folder?')
        sys.exit(-1)
except Exception as e:
    print(e)
    sys.exit(-1)

def draw_skeleton(frame, keypoints, colour, dotted=False):
    connections = [(0, 16), (0, 15), (16, 18), (15, 17),
                   (0, 1), (1, 2), (2, 3), (3, 4),
                   (1, 5), (5, 6), (6, 7), (1, 8),
                   (8, 9), (9, 10), (10, 11),
                   (8, 12), (12, 13), (13, 14),
                   (11, 24), (11, 22), (22, 23),
                   (14, 21), (14, 19), (19, 20)]

    for x, y in keypoints:
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        cv2.circle(frame, center=center, radius=4, color=colour, thickness=-1)

    for keypoint_id1, keypoint_id2 in connections:
        x1, y1 = keypoints[keypoint_id1]
        x2, y2 = keypoints[keypoint_id2]
        if 0 in (x1, y1, x2, y2):
            continue
        pt1 = int(round(x1)), int(round(y1))
        pt2 = int(round(x2)), int(round(y2))
        if dotted:
            draw_line(frame, pt1=pt1, pt2=pt2,
                      color=colour, thickness=2, gap=5)
        else:
            cv2.line(frame, pt1=pt1, pt2=pt2, color=colour, thickness=2)
    return frame

def compute_simple_bounding_box(skeleton):
    x = skeleton[::2]
    x = np.where(x == 0.0, np.nan, x)
    left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
    y = skeleton[1::2]
    y = np.where(y == 0.0, np.nan, y)
    top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))
    return left, right, top, bottom

def compute_homography(H_ratio, V_ratio, im_size):
    rationed_hight = im_size[1] * V_ratio
    rationed_width = im_size[0] * H_ratio
    src = np.array([[0, 0], [0, im_size[1]], [
                   im_size[0], im_size[1]], [im_size[0], 0]])
    #dst = np.array([[0+rationed_width/2, 0+rationed_hight], [0, im_size[1]], [im_size[0], im_size[1]], [im_size[0]-rationed_width/2, 0+rationed_hight]], np.int32)
    dst = np.array([[0+rationed_width/2, 0+rationed_hight], [0, im_size[1]], [im_size[0],
                                                                              im_size[1]], [im_size[0]-rationed_width/2, 0+rationed_hight]], np.int32)
    h, status = cv2.findHomography(src, dst)
    return h

def compute_overlap(rect_1, rect_2):  # rect = [left,right,top,bottom]
    x_overlap = max(0, min(rect_1[1], rect_2[1]) - max(rect_1[0], rect_2[0]))
    y_overlap = max(0, min(rect_1[3], rect_2[3]) - max(rect_1[2], rect_2[2]))
    overlapArea = x_overlap * y_overlap
    if overlapArea:
        overlaps = True
    else:
        overlaps = False
    return overlaps

opacity_degree = 0.4
colors = [(0, 255, 0), (0, 0, 255)]

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--image_in", default="/mnt/work1/video/testing/03_0032/124.jpg",
                    help="Process an image ora a video recorder/stream. Read all standard formats (jpg, png, bmp, etc.).")

parser.add_argument("--image_out", default="./output_image.jpg",
                    help="Image/video output")

parser.add_argument("--net_size", default="512x384",
                    help="Openpose network size")

parser.add_argument("--horizontal_ratio", default="0.7",
                    help="Ratio between the closest horizotal line of the scene to the furthest visible. It must be a float value in (0,1)")

parser.add_argument("--vertical_ratio", default="0.7",
                    help="Ratio between the height of the trapezoid wrt the rectangular bird’s view scene (image hight). It must be a float value in (0,1)")

parser.add_argument("--openpose_folder", default="/home/dexmac/openpose/models/",
                    help="Path to the local OpenPose installation directory")

# Parsing arguments
args = parser.parse_known_args()

# params
horizontal_ratio = float(args[0].horizontal_ratio)
vertical_ratio = float(args[0].vertical_ratio)

# Read input image
im = cv2.imread(args[0].image_in)

im_size = (im.shape[1], im.shape[0])

homography_matrix = compute_homography(
    horizontal_ratio, vertical_ratio, im_size)

im_overlay = im.copy()

ellipse_boxes = []

draw_ellipse_requirements = []

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()

# Openpose params

# Model path
params["model_folder"] = args[0].openpose_folder

# Face disabled
params["face"] = False

# Hand disabled
params["hand"] = False

# Net Resolution
params["net_resolution"] = args[0].net_size

# Gpu number
params["num_gpu"] = 1  # Set GPU number

# Gpu Id
params["num_gpu_start"] = 0  # Set GPU start id (not considering previous)

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()

# Assign input image to openpose
datum.cvInputData = im

# Start wrapper
opWrapper.emplaceAndPop([datum])

# Get openpose coordinates (rounding values)
skeletal_coordinates = np.around(
    np.array(datum.poseKeypoints).tolist(), 2).tolist()

# Remove probability from joints and round results
temporary_skeletals = []

if type(skeletal_coordinates) is list:

    for skeletal_coordinate in skeletal_coordinates:
        temporary_skeletals.append([reduced[0:2]
                                    for reduced in skeletal_coordinate])

    skeletal_coordinates = temporary_skeletals

    # Get joints sequence
    bodys_sequence = []
    for body in skeletal_coordinates:
        body_sequence = []
        
        # For each joint put it in vetcor list
        for joint in body:
            body_sequence.append(joint[0])
            body_sequence.append(joint[1])

        # Add body sequence to list
        bodys_sequence.append(body_sequence)

    skeletal_coordinates = bodys_sequence

    is_skeletal_overlapped = np.zeros(np.shape(skeletal_coordinates)[0])

    for skeletal_coordinate in skeletal_coordinates:

        skeletal_coordinate = np.array(skeletal_coordinate)
        im = draw_skeleton(im, skeletal_coordinate.reshape(-1, 2), (255, 0, 0))

        left, right, top, bottom = compute_simple_bounding_box(skeletal_coordinate)
        hight = round((bottom-top))
        bb_center = np.array([(left + right) / 2, (top + bottom) / 2], np.int32)

        calculated_hight = 0
        pointer = -1
        joint_set = [0, 1, 8, 12, 13, 14]  # left side body joints from top to down
        # joint_set_mirror = [0,1,8,9,10,11] # -ToBeCompleted- right side body joints from top to down. In case joint in left side are missing, check the right side
        sub_joint_set = joint_set[:-1]
        for k in sub_joint_set:
            pointer += 1
            if skeletal_coordinate[joint_set[pointer]*2] and skeletal_coordinate[joint_set[pointer+1]*2] and skeletal_coordinate[joint_set[pointer]*2+1] and skeletal_coordinate[joint_set[pointer+1]*2+1]:
                calculated_hight = calculated_hight + math.sqrt(((skeletal_coordinate[joint_set[pointer]*2]-skeletal_coordinate[joint_set[pointer+1]*2])**2) +
                                                                ((skeletal_coordinate[joint_set[pointer]*2+1]-skeletal_coordinate[joint_set[pointer+1]*2+1])**2))

        # computing how the height of the circle varies in perspective
        pts = np.array([[bb_center[0], top], [bb_center[0], bottom]], np.float32)
        pts1 = pts.reshape(-1, 1, 2).astype(np.float32)  # (n, 1, 2)
        dst1 = cv2.perspectiveTransform(pts1, homography_matrix)
        # height of the ellipse in perspective
        width = int(dst1[1, 0][1] - dst1[0, 0][1])

        # bbx surrending the ellipses, to compute whether there is any overlap between two ellipses
        ellipse_bbx = [bb_center[0]-calculated_hight,
                    bb_center[0]+calculated_hight, bottom-width, bottom+width]
        #im = cv2.rectangle(im, (int(bb_center[0]-calculated_hight), int(bottom-(width))), (int(bb_center[0]+(calculated_hight)), int(bottom+(width))), (0,0,0), 2)
        ellipse_boxes.append(ellipse_bbx)

        draw_ellipse_requirements.append(
            [bb_center[0], bottom, calculated_hight, width])


    if len(ellipse_boxes) > 1:
        # checks for overlaps between people's ellipses, to determine risky or not
        for ind1, ind2 in itertools.combinations(list(range(0, len(ellipse_boxes))), 2):
            is_overlap = compute_overlap(ellipse_boxes[ind1], ellipse_boxes[ind2])
            if is_overlap and not((is_skeletal_overlapped[ind1] or is_skeletal_overlapped[ind2])):
                is_skeletal_overlapped[ind1] = is_overlap
                is_skeletal_overlapped[ind2] = is_overlap

    i = -1
    for skeletal_coordinate in skeletal_coordinates:
        i += 1
        color = colors[int(is_skeletal_overlapped[i])]
        im_overlay = cv2.ellipse(im_overlay, (draw_ellipse_requirements[i][0], draw_ellipse_requirements[i][1]),
                                (int(draw_ellipse_requirements[i][2]), draw_ellipse_requirements[i][3]), 0, 0, 360, color, -1)

    im_overlay = cv2.addWeighted(
        im_overlay, opacity_degree, im, 1 - opacity_degree, 0)

# Write output image on disk
cv2.imwrite(args[0].image_out, im_overlay)

# Show image and wait some time
cv2.imshow('im', im_overlay)
cv2.waitKey(5000)
