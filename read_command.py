#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import os.path
import time


class GoForward():
    def __init__(self):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)

        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10)
        # Twist is a datatype for velocity
        move_cmd = Twist()

        previous_time_stamp = None

        file = 'speed.txt'
        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            # publish the velocity
            with open(file, 'r') as f:
                line = f.readline()
                x, y, z = line.split(' ')
                current_time_stamp = time.ctime(os.path.getmtime(file))
                if current_time_stamp != previous_time_stamp:
                    move_cmd.linear.x = x
                    move_cmd.angular.z = z
                    move_cmd.linear.y = y
                else:
                    move_cmd.linear.x = 0
                    move_cmd.angular.z = 0
                    move_cmd.linear.y = 0
            previous_time_stamp = current_time_stamp
            self.cmd_vel.publish(move_cmd)
            # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        GoForward()
    except:
        rospy.loginfo("GoForward node terminated.")
