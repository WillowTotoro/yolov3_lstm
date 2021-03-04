import math
# import sys
# sys.path.append('/home/sysadmin/darknet/')
import cv2
import numpy as np
# from visual_servo import cal_robot_vel
import pyrealsense2 as rs
# import darknet

# sys.path.append('/home/sysadmin/darknet/')


img_height = 1280
img_width = 720
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, img_height,img_width, rs.format.z16, 15)
config.enable_stream(rs.stream.color, img_height,img_width, rs.format.bgr8, 15)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)
# Start streaming
profile = pipeline.start(config)
# dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter()   # Temporal   - reduces temporal noise
hole_fill = rs.hole_filling_filter()
# frames = pipe.wait_for_frames()


# try:
while True:
    frames = pipeline.wait_for_frames()
    # frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue
    
    # depth_frame = frames.get_depth_frame()
    # filtered = dec_filter.process(depth_frame)
    filtered = spat_filter.process(depth_frame)
    filtered = temp_filter.process(filtered)
    filtered = hole_fill.process(filtered)
    # Convert images to numpy arrays
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    bgr_color_image = np.asanyarray(color_frame.get_data())
    #bgr_color_image = cv2.resize(bgr_color_image,(416,416))
    rgb_color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)
    #rgb_color_image = cv2.resize(rgb_color_image,(416,416))

    colorized_depth_image = np.asanyarray(colorizer.colorize(filtered).get_data())
    
    x = 1000
    y = 350
    pt1 = (int(x-3), int(y-3))
    pt2 = (int(x+3), int(y+3))
    colorized_depth_image = cv2.rectangle(colorized_depth_image, pt1, pt2, (0,0,0), -1)
    
    depth = np.asanyarray(depth_frame.get_data())
    
    depth = depth[pt1[1]:pt2[1], pt1[0]:pt2[0]].astype(float)
    
    print(depth)

    # Get data scale from the device and convert to meters
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth = depth * depth_scale
    # dist = np.mean(np.asarray(depth))
    dist, _, _, _ = cv2.mean(depth)
    print("\n\n\nDetected {0} meters away...".format(round(dist,2)))

    # robot_vel = cal_robot_vel(x, y, dist, gain=0.1)
    # Show images
    images = np.hstack((bgr_color_image, colorized_depth_image))
    cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1000)
# except:
#     print('Problem occurs')
#     pipeline.stop()
