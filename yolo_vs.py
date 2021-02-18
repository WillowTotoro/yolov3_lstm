import math
import cv2
import numpy as np
from visual_servo import cal_robot_vel
import pyrealsense2 as rs
import darknet
import sys
sys.path.append('/home/sysadmin/darknet/')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 920, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 920, rs.format.bgr8, 30)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)
# Start streaming
profile = pipeline.start(config)

# converting BB coordinates in yolo txt format in pixels


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# drawing bounding boxes in image from detections

def cvDrawBoxes(detections, img):

    if len(detections) == 0:
        return(img, 320, 240)
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 1)
        cv2.putText(img, detection[0].decode(
        ), (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
    return(img, int(x), int(y))


cx = 637.992919921875
cy = 361.408325195312
fx = 927.108520507812
fy = 927.300964355469

#  PPX:        	637.992919921875
#   PPY:        	361.408325195312
#   Fx:         	927.108520507812
#   Fy:         	927.300964355469

IP_configPath = "/home/sysadmin/darknet/cfg/yolov3.cfg"
IP_weightPath = "/home/sysadmin/darknet/yolov3.weights"
IP_metaPath = "/home/sysadmin/darknet/cfg/coco.data"
darknet.set_gpu(1)
IP_netMain = darknet.load_net_custom(IP_configPath.encode(
    "ascii"), IP_weightPath.encode("ascii"), 0, 1)  # batch size = 1
IP_metaMain = darknet.load_meta(IP_metaPath.encode("ascii"))

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        bgr_color_image = np.asanyarray(color_frame.get_data())
        #bgr_color_image = cv2.resize(bgr_color_image,(416,416))
        rgb_color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)
        #rgb_color_image = cv2.resize(rgb_color_image,(416,416))

        colorized_depth_image = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())
        # colorized_depth_image = cv2.resize(colorized_depth_image,(416,416))

        darknet_image = darknet.make_image(darknet.network_width(
            IP_netMain), darknet.network_height(IP_netMain), 3)
        # resized = cv2.resize(image,(darknet.network_width(IP_netMain),darknet.network_height(IP_netMain)),interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, rgb_color_image.tobytes())
        detections = darknet.detect_image(
            IP_netMain, IP_metaMain, darknet_image, thresh=0.8)

        bgr_color_image, x, y = cvDrawBoxes(detections, bgr_color_image)

        pt1 = (x-30, y-30)
        pt2 = (x+30, y+30)

        colorized_depth_image = cv2.rectangle(
            colorized_depth_image, pt1, pt2, (255, 255, 255), 1)

        # Stack both images horizontally
        images = np.hstack((bgr_color_image, colorized_depth_image))

        # Crop depth data:
        depth = np.asanyarray(depth_frame.get_data())
        # depth = cv2.resize(depth,(416,416))
        depth = depth[x-30:x+30, y-30:y+30].astype(float)

        # Get data scale from the device and convert to meters
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        #dist = np.mean(np.asarray(depth))
        dist, _, _, _ = cv2.mean(depth)
        print("\n\n\nDetected {0} meters away...".format(dist))

        robot_vel = cal_robot_vel(x, y, dist, gain=0.1)
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1000)

finally:
    # Stop streaming
    pipeline.stop()
