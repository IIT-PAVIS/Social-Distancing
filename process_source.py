"""
    Social-Distancing

    IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Description: Social distancing process source ove single source (file, video, stream, etc, test purpouse)

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
import json
import queue

from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

from stream_server import StreamServer
from response_server import ResponseServer
from mjpeg_reader import MjpegReader
from social_distancing import SocialDistancing

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON`'
              'in CMake and have this Python script in the right folder?', flush=True)
        os._exit(-1)
except Exception as e:
    print(e, flush=True)
    os._exit(-1)

# Jetson native camera capture command line (opencv input)
def gstreamer_pipeline(capture_width=640, capture_height=480, display_width=640, display_height=480, framerate=25, flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (capture_width, capture_height, framerate, flip_method, display_width, display_height))


class ProcessSource:
    '''
        Initialize
    '''

    def __init__(self, args):

        # Social Distancing arguments
        arguments = {}

        # Arguments
        arguments["horizontal_ratio"] = args[0].horizontal_ratio
        arguments["vertical_ratio"] = args[0].vertical_ratio
        arguments["calibration"] = args[0].calibration
        arguments["body_threshold"] = args[0].body_threshold
        arguments["show_confidence"] = args[0].show_confidence
        arguments["show_sketch"] = args[0].show_sketch

        # Initialize social distancing
        self.social_distancing = SocialDistancing(arguments)

        # Initialize Openpose
        self.initialize_openpose(args)

        # Initialize file opening/writing and streaming
        self.initialize_others(args)

    '''
        Initialize openpose
    '''

    def initialize_openpose(self, args):
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
        # Set GPU start id (not considering previous)
        params["num_gpu_start"] = 0

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        self.datum = op.Datum()

    '''
        Initialize acquiring methods (video, mjpeg preprocessed json, jetson, etc), sockets, output files
    '''

    def initialize_others(self, args):
        # Convert args to boolean
        self.use_video = True if args[0].video == "enabled" else False

        # Process image
        self.use_image = True if args[0].image == "enabled" else False

        # Use preview
        self.use_preview = True if args[0].preview == "enabled" else False

        # Jetson internal camera enabled
        self.jetson_video = True if args[0].jetson_video == "enabled" else False

        # Mjpeg video reader
        self.use_mjpeg = True if args[0].use_mjpeg == "enabled" else False

        # Enable streaming ption
        self.streaming = True if args[0].streaming == "enabled" else False

        # Use json as input
        self.use_js = True if args[0].use_js == "enabled" else False

        # Json input file
        self.js_in = args[0].js_in

        if self.use_video:
            # Open video capture
            if not self.jetson_video:
                # Use standard cv2 capture library
                self.cap = cv2.VideoCapture(args[0].stream_in)
            else:
                # Connect Standard cv2 capture library to gstreamer
                print(gstreamer_pipeline(flip_method=0))
                self.cap = cv2.VideoCapture(
                    gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                print("Error: Opening video stream or file {0}".format(
                    args[0].stream_in), flush=True)
                sys.exit(-1)

            # Get input size
            width = int(self.cap.get(3))
            height = int(self.cap.get(4))

            self.mask_in = cv2.imread(args[0].mask_in)

            if not self.streaming:
                # Open video output (if output is not an image)
                self.out = cv2.VideoWriter(args[0].stream_out, cv2.VideoWriter_fourcc(*args[0].encoding_codec),
                                           25, (width, height))

                if self.out is None:
                    print("Error: Unable to open output video file {0}".format(
                        args[0].stream_out), flush=True)
                    sys.exit(-1)

            # Get image size
            im_size = (width, height)

        if self.use_image:
            self.mask_in = cv2.imread(args[0].mask_in)

            self.image = cv2.imread(args[0].image_in)
            
            if self.image is None:
                print("Error: Unable to open input image file {0}".format(
                    args[0].image_in), flush=True)
                sys.exit(-1)

            self.image_out = args[0].image_out

            # Get image size
            im_size = (self.image.shape[1], self.image.shape[0])

        if self.use_mjpeg:
            # Create mjpeg reader
            self.mjpeg_reader = MjpegReader(args[0].stream_in)

            # Read first image to get image size
            image = self.mjpeg_reader.get_image()

            if not self.mjpeg_reader.is_opened():
                print("Error: Unable to open input image file {0}".format(
                    args[0].image_in), flush=True)
                exit(-1)

            # Get input size
            width = int(image.shape[1])
            height = int(image.shape[0])

            if not self.streaming:
                # Open video output (if output is not an image)
                self.out = cv2.VideoWriter(args[0].stream_out, cv2.VideoWriter_fourcc(*args[0].encoding_codec),
                                           int(args[0].dummy_fps), (width, height))

                if self.out is None:
                    print("Error: Unable to open output video file {0}".format(
                        args[0].stream_out), flush=True)
                    sys.exit(-1)

            print("Mjpeg multipart file:{0}x{1}".format(width, height))

            # Get image size
            im_size = (width, height)

        if not self.use_js:
            # Compute Homograpy
            self.social_distancing.compute_homography(im_size)

        self.background_masked = False
        # Open image backgrouns, if it is necessary
        if args[0].masked == "enabled":
            # Set masked flag
            self.background_masked = True

            # Load static background
            self.background_image = cv2.imread(args[0].background_in)

            # Close, if no background, but required
            if self.background_image is None:
                print(
                    "Error: Unable to load background image (flag enabled)", flush=True)
                sys.exit(-1)

        if self.use_js:
            im_size = (
                self.background_image.shape[1], self.background_image.shape[0])

            self.out = cv2.VideoWriter(args[0].stream_out, cv2.VideoWriter_fourcc(*args[0].encoding_codec),
                                       int(args[0].dummy_fps), im_size)

            self.social_distancing.compute_homography(im_size)

        # Json server
        self.dt_vector = {}

        # Client list
        self.stream_list = []
        self.js_list = []

        if self.streaming:
            # Initialize video server
            self.video_server = StreamServer(
                int(args[0].video_port), self.stream_list, "image/jpeg")
            self.video_server.activate()

            # Initialize stream server
            self.stream_server = StreamServer(
                int(args[0].stream_port), self.js_list, "application/json")
            self.stream_server.activate()

            # Initialize json server
            self.js_server = ResponseServer(
                int(args[0].js_port), "application/json")
            self.js_server.activate()

        # turbo jpeg initialization
        self.jpeg = TurboJPEG()

        # Json recorder
        self.js_recording = False
        if args[0].js_out != "":
            self.js_recording = True
            self.js_out = open(args[0].js_out, "w")

        # Mjpeg recorder
        self.mjpeg_recorder = False
        if args[0].mjpeg_out != "":
            self.mjpeg_recorder = True
            self.mjpeg_out = open(args[0].mjpeg_out, "wb")

        # Json version
        self.dt_vector["vs"] = 1

        # Fps evaluation init
        self.millis = 0
        self.frames = 0

    '''
        Process source and save on image/video/js file, distribuite on network
    '''

    def process_source(self, source, background):
        start = round(time.time()*1000)

        if self.mask_in is not None:
            source = cv2.bitwise_and(source, self.mask_in)

        # Check if pre-processed json is used
        if not self.use_js:
            # Assign input image to openpose
            self.datum.cvInputData = source

            # Use Openpose to extract poses
            self.opWrapper.emplaceAndPop([self.datum])

            # Get openpose coordinates (rounding values)
            skeletals = np.around(
                np.array(self.datum.poseKeypoints).tolist(), 2).tolist()
        else:
            # Copy json data
            skeletals = source

        # Trace on background
        if self.background_masked:
            source = background

        if type(skeletals) is not list:
            return background

        # Evaluate distances, draw body and ellipses and get json bodies and ellipses list
        image, bodies, ellipses = self.social_distancing.distances_calculate(
            source, skeletals, [1 for k in range(len(skeletals))]) 

        # Save data to json vector
        self.dt_vector["bodies"] = bodies
        self.dt_vector["ellipses"] = ellipses

        if self.streaming:
            # Send video to client queues
            self.send_image(self.stream_list, image, int(self.dt_vector['ts']))

            # Put json vector availble to rest requests
            self.js_server.put(bytes(json.dumps(self.dt_vector), "UTF-8"))

            # Send json vestor available to streaming
            self.send_js(self.js_list, bytes(json.dumps(
                self.dt_vector), "UTF-8"), int(self.dt_vector['ts']))

        # Write json data
        if self.js_recording:
            self.js_out.write(json.dumps(self.dt_vector)+"\n")

        stop = round(time.time()*1000)

        if self.millis > 1000:
            print("Analyzing at {0} Fps".format(self.frames), end="\r", flush=True)
            self.millis = 0
            self.frames = 0

        self.millis += stop - start
        self.frames += 1

        return image

    '''
        Send image over queue list and then over http mjpeg stream
    '''

    def send_image(self, queue_list, image, ts):

        encoded_image = self.jpeg.encode(image, quality=80)
        # Put image into queue for each server thread
        for q in queue_list:
            try:
                block = (ts, encoded_image)
                q.put(block, True, 0.02)
            except queue.Full:
                pass
    '''
        Send json over queue list and then over http multipart stream
    '''

    def send_js(self, queue_list, js, ts):

        # Put json into queue for each server thread
        for q in queue_list:
            try:
                block = (ts, js)
                q.put(block, True, 0.02)
            except queue.Full:
                pass

    '''
        Analyze video
    '''

    def analyze_video(self):
        first_frame = True
        counter = 0
        while self.cap.isOpened():
            # Get a global image ts
            self.dt_vector['ts'] = int(round(time.time() * 1000))

            # Capture from image/video
            ret, image = self.cap.read()

            # Check image
            if image is None or not ret:
                os._exit(0)

            # Record image
            if self.mjpeg_recorder:
                encoded_image = self.jpeg.encode(image, quality=80)

                header = "--myboundary\r\n" \
                    "X-TimeStamp: " + str(self.dt_vector['ts']) + "\r\n" \
                    "Content-Type: image/jpeg\r\n" \
                    "Content-Length: " + \
                    str(len(encoded_image)) + "\r\n\r\n"

                self.mjpeg_out.write(bytes(header, "UTF-8"))
                self.mjpeg_out.write(encoded_image)

            # Get openpose output
            if self.background_masked:
                background = self.background_image.copy()
            else:
                background = image

            image = self.process_source(image, background)

            # Write image
            if not self.streaming:
                self.out.write(image)

            # Show image and wait some time
            if self.use_preview:
                cv2.imshow('Social Distance', image)
                cv2.waitKey(1)

            #print(counter, end="\r", flush=True)
            counter += 1

    '''
        Analyze image
    '''

    def analyze_image(self):

        # Get openpose output
        if self.background_masked:
            background = self.background_image.copy()
        else:
            background = self.image

        self.image = self.process_source(self.image, background)

        # Write image
        cv2.imwrite(self.image_out, self.image)

        # Show image and wait some time
        if self.use_preview:
            cv2.imshow('Social Distance', self.image)
            cv2.waitKey(1000)

    '''
        Analyze json data
    '''

    def analyze_js(self):
        # Read json files
        lines = open(self.js_in, "r").read().split("\n")

        # While there are lines
        for line in lines[:-1]:
            js_line = json.loads(line)

            # Create
            background = self.background_image.copy()

            if 'vs' in js_line.keys():
                self.image = self.process_source(
                    js_line['bodies'], background)
            else:
                self.image = self.process_source(
                    js_line['bodys'], background)

            # Write image
            if not self.streaming:
                self.out.write(self.image)

            # Show image and wait some time
            if self.use_preview:
                cv2.imshow('Social Distance', self.image)
                cv2.waitKey(1)

    '''
        Analyze mjpeg (timestamped jpeg sequence)
    '''

    def analyze_mjpeg(self):
        first_frame = True
        counter = 0

        old_timestamp = self.mjpeg_reader.get_ts()
        while True:
            # Capture from image/video
            image = self.mjpeg_reader.get_image()

            # Get a global image ts
            self.dt_vector['ts'] = self.mjpeg_reader.get_ts()

            # Check image
            if image is None:
                os._exit(0)

            # Record image
            if self.mjpeg_recorder:
                encoded_image = self.jpeg.encode(image, quality=80)

                header = "--myboundary\r\n" \
                    "X-TimeStamp: " + str(self.dt_vector['ts']) + "\r\n" \
                    "Content-Type: image/jpeg\r\n" \
                    "Content-Length: " + \
                    str(len(encoded_image)) + "\r\n\r\n"

                self.mjpeg_out.write(bytes(header, "UTF-8"))
                self.mjpeg_out.write(encoded_image)

            # Get openpose output
            if self.background_masked:
                background = self.background_image.copy()
            else:
                background = image

            image = self.process_source(image, background)

            # Write image
            if not self.streaming:
                self.out.write(image)

            # Show image and wait some time
            if self.use_preview:
                cv2.imshow('Social Distance', image)
                cv2.waitKey(1)

            # Wait timestamp difference
            time.sleep((self.mjpeg_reader.get_ts() - old_timestamp)/1000)

            # print(counter, end = "\n", flush=True)

            # Store old timestamp
            old_timestamp = self.mjpeg_reader.get_ts()

            counter += 1

    '''
        Analyze image/video/json/mjpeg
    '''

    def analyze(self):
        if self.use_image:
            self.analyze_image()

        if self.use_video:
            self.analyze_video()

        if self.use_js:
            self.analyze_js()

        if self.use_mjpeg:
            self.analyze_mjpeg()


'''
    Main Entry
'''
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--mjpeg_out", default="",
                        help="write on mjpeg multipart file, if defined")

    parser.add_argument("--use_mjpeg", default="disabled",
                        help="select mjpeg multipart recorded file")

    parser.add_argument("--video", default="disabled",
                        help="select video mode, if defined")

    parser.add_argument("--image", default="enabled",
                        help="select image mode, if defined")

    parser.add_argument("--masked", default="enabled",
                        help="mask to blur visual appearance of people")

    parser.add_argument("--image_in", default="./imput image.jpg",
                        help="Process an image. Read all standard image formats")

    parser.add_argument("--image_out", default="./output_image.jpg",
                        help="Image output")

    parser.add_argument("--background_in", default="./background.jpg",
                        help="Process an image, read all standard formats (jpg, png, bmp, etc.).")

    parser.add_argument("--stream_in", default="./input_stream.avi",
                        help="Process an image ora a video stream. Read all standard formats and connect to live stream")

    parser.add_argument("--stream_out", default="./output_stream.avi",
                        help="Image/video output")

    parser.add_argument("--net_size", default="512x384",
                        help="Openpose network size")

    parser.add_argument("--horizontal_ratio", default="0.7",
                        help="Ratio between the closest horizotal line of the scene to the furthest visible. It must be a float value in (0,1)")

    parser.add_argument("--vertical_ratio", default="0.7",
                        help="Ratio between the height of the trapezoid wrt the rectangular bird’s view scene (image height). It must be a float value in (0,1)")

    parser.add_argument("--openpose_folder", default="/home/dexmac/openpose/models/",
                        help="Path to the local OpenPose installation directory")

    parser.add_argument("--preview", default="enabled",
                        help="Enable video out")

    parser.add_argument("--streaming", default="disabled",
                        help="Enable video streaming")

    parser.add_argument("--video_port", default="5002",
                        help="video streaming port")

    parser.add_argument("--js_port", default="5005",
                        help="json rest port")

    parser.add_argument("--stream_port", default="5007",
                        help="json streaming port")

    parser.add_argument("--calibration", default="1.0",
                        help="calibrate each point of view with this value")

    parser.add_argument("--body_threshold", default="0.2",
                        help="remove too low confidential body")

    parser.add_argument("--show_confidence", default="disabled",
                        help="show confidence value")

    parser.add_argument("--show_sketch", default="enabled",
                        help="show body/ellipse scketch")

    parser.add_argument("--encoding_codec", default="XVID",
                        help="change output video encoding mode")

    parser.add_argument("--use_js", default="disabled",
                        help="change output video encoding mode")

    parser.add_argument("--js_in", default="./input.json",
                        help="change output video encoding mode")

    parser.add_argument("--js_out", default="./output.json",
                        help="change output video encoding mode")

    parser.add_argument("--jetson_video", default="disabled",
                        help="use jetson video")

    parser.add_argument("--dummy_fps", default="25",
                        help="use it if input stream frame rate is not knowed")

    parser.add_argument("--mask_in", default="./mask.jpg",
                        help="mask in (jpg, png, bmp, etc.).")
 
    # Parsing arguments
    args = parser.parse_known_args()

    # Create social_distance object
    process_source = ProcessSource(args)

    # Do hard work
    process_source.analyze()
