"""
    Social-Distancing

    IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Description: Social distancing core module, use openpose data to estimate people distance

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
"""

import numpy as np
import cv2
import math
import itertools
import sys
import os
import argparse
import time
import matplotlib.pyplot as plt

from shapely.geometry.polygon import LinearRing


class SocialDistancing:
    """Social distancing: evaluate social distancing (due to covid-19) uncalibrated approach, determinate distances
    proportionally to right body segment. body orthogonality permits to obtain distance ellipse ground projection
    """

    # ellipses color
    colors = [(0, 255, 0), (0, 0, 255)]

    # Joint color array
    nd_color = [(51, 0, 153), (0, 0, 153),
                (0, 51, 153), (0, 102, 153),
                (0, 153, 153), (0, 153, 102),
                (0, 153, 51), (0, 153, 0),
                (153, 102, 0), (51, 153, 0),
                (102, 153, 0), (153, 153, 0),
                (153, 102, 0), (153, 51, 0),
                (153, 0, 0), (102, 0, 153),
                (153, 0, 102), (153, 0, 153),
                (153, 0, 102), (153, 0, 0),
                (153, 0, 0), (153, 0, 0),
                (153, 153, 0), (153, 153, 0),
                (153, 153, 0)
                ]

    # Joint connection array
    connections = [(1, 8), (1, 2), (1, 5), (0, 15),
                   (0, 16), (15, 17), (16, 18), (1, 0),
                   (2, 3), (3, 4), (5, 6), (6, 7),
                   (8, 9), (8, 12), (9, 10),
                   (12, 13), (10, 11), (13, 14),
                   (11, 24), (11, 22), (22, 23),
                   (14, 21), (14, 19), (19, 20)]

    # Body parts, useful to evaluate height
    body_parts = [#[12, 13, 14, 19],  # Left leg
                  #[9, 10, 11, 22],  # Right leg
                  [1, 8]  # Torso
                  #[5, 6, 7],  # Left arm
                  #[2, 3, 4] # Right arm
                  ] 

    def __init__(self, args):
        """Arguments

        Args:
            args (dictionary): library arguments, horizontal ratio, vertical ratio, calibarion, show sketch
        """
        # Ratio params
        self.horizontal_ratio = float(args["horizontal_ratio"])
        self.vertical_ratio = float(args["vertical_ratio"])

        # Calibrate heigh value
        self.calibrate = float(args["calibration"])

        # Show body/ellipse
        self.show_sketch = True if args["show_sketch"] == "enabled" else False

        # Show confidence
        self.show_confidence = True if args["show_confidence"] == "enabled" else False

    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton on image overalyed

        Args:
            frame (np array): target image
            keypoints (list): body joints
        """
        for keypoint_id1, keypoint_id2 in self.connections:
            x1, y1 = keypoints[keypoint_id1]
            x2, y2 = keypoints[keypoint_id2]

            if 0 in (x1, y1, x2, y2):
                continue

            pt1 = int(round(x1)), int(round(y1))
            pt2 = int(round(x2)), int(round(y2))

            cv2.circle(frame, center=pt1, radius=4,
                       color=self.nd_color[keypoint_id2], thickness=-1)
            cv2.line(frame, pt1=pt1, pt2=pt2,
                     color=self.nd_color[keypoint_id2], thickness=2)

    def compute_simple_bounding_box(self, skeleton):
        """Compute bounding box around skeleton

        Args:
            skeleton (list): joint list

        Returns:
            touple: bounding box coordinates
        """
        x = skeleton[::2]
        x = np.where(x == 0.0, np.nan, x)
        left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
        y = skeleton[1::2]
        y = np.where(y == 0.0, np.nan, y)
        top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))
        return (left, right, top, bottom)

    def compute_homography(self, im_size):
        """Calculate homography

        Args:
            im_size (tuple): homograpy matrix
        """
        rationed_hight = im_size[1] * self.vertical_ratio
        rationed_width = im_size[0] * self.horizontal_ratio
        src = np.array([[0, 0], [0, im_size[1]], [
            im_size[0], im_size[1]], [im_size[0], 0]])
        dst = np.array([[0+rationed_width/2, 0+rationed_hight], [0, im_size[1]], [im_size[0],
                                                                                  im_size[1]], [im_size[0]-rationed_width/2, 0+rationed_hight]], np.int32)
        self.homography_matrix, status = cv2.findHomography(src, dst)

    def convert_ellipses(self, ellipse_requirements, founded_violations):
        """Convert ellipses to json data format 

        Args:
            ellipse_requirements (list): ellipse specs
            founded_violations ([type]): violations founded

        Returns:
            [list]: ellipse in right format with violation info
        """

        ellipses = []
        for ellipse, violation in zip(ellipse_requirements, founded_violations):
            ellipses.append([int(ellipse[0]), int(ellipse[1]), int(
                ellipse[2]), int(ellipse[3]), int(violation)])
        return ellipses

    def trace(self, image, skeletal_coordinates, ellipses):
        """Draw features over the image, skeletal and ellipse

        Args:
            image (np array): target image
            skeletals (list): bodys list
            skeletal_coordinates (list): skeletal coordinate
            ellipses (list): ellipse list
            violation_found ([type]): violation founded 
        """

        # Trace ellipses and body on json and target image (if required)
        for skeletal_coordinate, ellipse in zip(skeletal_coordinates, ellipses):

            # Get coordinates
            skeletal_coordinate = np.array(skeletal_coordinate)

            # Trace ellipses skeleton
            cv2.ellipse(image,
                        (ellipse[0],
                        ellipse[1]),
                        (ellipse[2],
                        ellipse[3]), 0, 0, 360,
                        self.colors[ellipse[4]], thickness=2)

            self.draw_skeleton(
                image, skeletal_coordinate.reshape(-1, 2))

    def part_complete(self, joints, skeletal_coordinate):
        """Check if body part is complete

        Args:
            joints (list): body joints
            skeletal_coordinate (): [description]

        Returns:
            [type]: [description]
        """

        for k in joints:
            x = int(skeletal_coordinate[k*2])
            y = int(skeletal_coordinate[k*2+1])

            if x == 0 and y == 0:
                return False

        return True

    def evaluate_height(self, skeletal_coordinate):
        """Evaluate height from bodys part 

        Args:
            skeletal_coordinate (list): skeletal coordinate

        Returns:
            float: height
        """
        # Calculate skeleton height
        calculated_height = 0
        pointer = -1

        # Set joint_set to get height as worst case (left harm)
        joint_set = self.body_parts[0]

        # Check if other more useful parts are complete
        for part in self.body_parts:
            if self.part_complete(part, skeletal_coordinate):
                # Better part founded!
                joint_set = part
                break

        # Evaluate height
        pointer = -1
        for k in joint_set[:-1]:
            pointer += 1
            if skeletal_coordinate[joint_set[pointer]*2]\
                    and skeletal_coordinate[joint_set[pointer+1]*2]\
                    and skeletal_coordinate[joint_set[pointer]*2+1]\
                    and skeletal_coordinate[joint_set[pointer+1]*2+1]:
                calculated_height = calculated_height +\
                    math.sqrt(((skeletal_coordinate[joint_set[pointer]*2] -
                                skeletal_coordinate[joint_set[pointer+1]*2])**2) +
                              ((skeletal_coordinate[joint_set[pointer]*2+1] -
                                skeletal_coordinate[joint_set[pointer+1]*2+1])**2))

        # Set parameter (calibrate) to optimize settings (camera dependent)
        return calculated_height * self.calibrate

    def evaluate_overlapping(self, violation_found, draw_ellipse_requirements):
        """Evaluate if there is an overlapping between two ellippses

        Args:
            violation_found (list): violation
            draw_ellipse_requirements (list): ellipses descriptors
        """
        # checks for overlaps between people's ellipses, to determine risky or not
        body_num = len(draw_ellipse_requirements)

        # Initialize violation list
        for body_idx in range(body_num):
            violation_found[body_idx] = 0

        # Check if detected body are two at least
        if body_num > 1:
            # Find a body,if is already in violation do not consider it
            for body_idx_1 in range(body_num):
                if violation_found[body_idx_1] == 1:
                    continue
                
                # Find another body (if it already in violation it is the same)
                for body_idx_2 in range(body_num):
                    # Same body, skip
                    if body_idx_1 == body_idx_2:
                        continue

                    ellipse1 = tuple(draw_ellipse_requirements[body_idx_1])
                    ellipse2 = tuple(draw_ellipse_requirements[body_idx_2])

                    if not self.contains(ellipse1, ellipse2):
                        continue

                    # ellipse
                    a = self.ellipse_polyline(ellipse1)
                    b = self.ellipse_polyline(ellipse2)

                    # Check if there is polylines intesect, then violation is active
                    if self.intersections(a, b):
                        violation_found[body_idx_1] = 1
                        violation_found[body_idx_2] = 1
                        break
                    
    def create_joint_array(self, skeletal_coordinates):
        """convert openpose output in custom format (joint sequence array)

        Args:
            skeletal_coordinates (list): body coordinates

        Returns:
            [list]: joint sequecence      
        """
        # Get joints sequence
        bodies_sequence = []
        for body in skeletal_coordinates:
            body_sequence = []
            # For each joint put it in vector list
            for joint in body:
                body_sequence.append(joint[0])
                body_sequence.append(joint[1])

            # Add body sequence to list
            bodies_sequence.append(body_sequence)

        # Assign coordiates sequence
        return bodies_sequence

    def evaluate_ellipses(self, skeletal_coordinates, draw_ellipse_requirements,
                          ellipse_boxes, ellipse_pool, ellipses, masked_status):
        """Calculate ellipses around body for each in list

        Args:
            skeletal_coordinates (list): skeletal coordinates
            draw_ellipse_requirements (list): ellipses info
            ellipse_boxes (list): ellipses roundig boxes
            ellipse_pool (list): ellipses pool
            ellipses (list): ellipses
            masked_status (list): store if body is detected in masked zone
        """

        index = 0
        for skeletal_coordinate in skeletal_coordinates:

            # Evaluate skeleton bounding box
            left, right, top, bottom = self.compute_simple_bounding_box(
                np.array(skeletal_coordinate))

            bb_center = np.array(
                [(left + right) / 2, (top + bottom) / 2], np.int32)

            calculated_height = self.evaluate_height(skeletal_coordinate)

            if masked_status[index] == 0:
                calculated_height /= 2

            # computing how the height of the circle varies in perspective
            pts = np.array(
                [[bb_center[0], top], [bb_center[0], bottom]], np.float32)

            pts1 = pts.reshape(-1, 1, 2).astype(np.float32)  # (n, 1, 2)

            dst1 = cv2.perspectiveTransform(pts1, self.homography_matrix)

            # height of the ellipse in perspective
            width = int(dst1[1, 0][1] - dst1[0, 0][1])

            # Basic control to avoid very wrong ellipses -> To be fixed
            if width > 0.5*calculated_height:
                width = int(0.5*calculated_height)

            # Bounding box surrending the ellipses, useful to compute whether there is any overlap between two ellipses
            ellipse_bbx = (bb_center[0] - calculated_height,
                           bb_center[0] + calculated_height, bottom - width, bottom + width)

            # Add boundig box to ellipse list
            ellipse_boxes.append(ellipse_bbx)

            ellipse = [int(bb_center[0]), int(bottom),
                       int(calculated_height), int(width)]

            draw_ellipse_requirements.append(ellipse)

            index += 1

    def distances_calculate(self, image, skeletals, masked_status):
        """Calculate distances between skeletals

        Args:
            image (np array): original image
            skeletals (list): skeletal list
            masked_status (list): masked status list

        Returns:
            [tuple]: (new image, new bodies list, ellipse list)
        """
        ellipse_boxes = []
        draw_ellipse_requirements = []
        ellipse_pool = []
        ellipses = []

        # Remove probability from joints and get a joint position list
        skeletal_coordinates = self.create_joint_array(
            skeletals)

        # Initialize overlapped buffer
        is_skeletal_overlapped = np.zeros(
            np.shape(skeletal_coordinates)[0])
        
        # Evaluate ellipses for each body detected by openpose
        self.evaluate_ellipses(skeletal_coordinates,
                               draw_ellipse_requirements, ellipse_boxes, ellipse_pool, ellipses, masked_status)

        # # Evaluate overlapping
        self.evaluate_overlapping(
            is_skeletal_overlapped, draw_ellipse_requirements)

        ellipses = self.convert_ellipses(draw_ellipse_requirements, is_skeletal_overlapped)

        # Trace results over output image and return data lists
        if self.show_sketch:
            self.trace(image, skeletal_coordinates, ellipses)

        return (image, skeletals, ellipses)

    def ellipse_polyline(self, ellipse, n=32):
        """Convert ellipse into a polyline, for each ellipse

        Args:
            ellipse (list): ellipse list 
            n (int, optional): segments. Defaults to 100.

        Returns:
            list: [description]
        """
        
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        st = np.sin(t)
        ct = np.cos(t)

        x0, y0, a, b = ellipse
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ct
        p[:, 1] = y0 + b * st

        return p

    def intersections(self, a, b):
        """check if two polylines are intersected

        Args:
            a (polyline): polyline a
            b (polyline): polyline b

        Returns:
            boolean: true if polylines are intersected
        """
        try:
            ea = LinearRing(a)
            eb = LinearRing(b)
            return ea.intersection(eb)
        except:
            return False

    def to_rectangle(self, ellipse):   
        """Convert ellipse to rectangle (top, left, bottom, right)

        Args:
            ellipse (tuple): bounding rectangle descriptor
        """
        x, y, a, b = ellipse
        return(x-a,y-b,x+a,y+b)
 
    def contains(self, ellipse1, ellipse2):
        """check if ellipses bounding rectangles are overlapped 

        Args:
            ellipse1 (tuple): ellipse one
            ellipse2 (tuple): ellipse two

        Returns:
            boolean:     
        """
        r1 = self.to_rectangle(ellipse1)
        r2 = self.to_rectangle(ellipse2)
        
        if (r1[0]>=r2[2]) or (r1[2]<=r2[0]) or (r1[3]<=r2[1]) or (r1[1]>=r2[3]):
            return False
        else:
            return True