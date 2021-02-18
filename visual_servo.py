import numpy as np


def cal_robot_vel(bbcx, bbcy, dist, gain=0.1):

    cx = 637.992919921875
    cy = 361.408325195312
    fx = 927.108520507812
    fy = 927.300964355469

    img_jacobian = np.array(
        [[-fx/dist, bbcx/dist, -(fx+bbcx**2/fx)], [0, -fy/dist, -bbcx*bbcy/fy]])
    pixel_diff = [bbcx-cx, bbcy-cy]

    vel_matrix = np.linalg.inv(img_jacobian)*gain*pixel_diff

    return vel_matrix
