#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage

# segmentation.py
# Author: tvoje mama
# This ros node is used to subscribe images from camera
# and publish a semantic segmentation with three types
# of labels:
#   -feasible      i.e. road     (you want to ride here)
#   -unfeasible    i.e. non-road (you dont want to ride here)
#   -non-important i.e  objects  (you have no information what is behind (e.g people))


def segmentation_callback(msg):
    rospy.loginfo("Segmentation in process")



def start_seg_node():
    rospy.init_node('segmentation_node')#, anonymous=True)
    rospy.loginfo("Starting Segmentation node")

    # img_sub = rospy.Subscriber(
    #     '/camera_front/image_color/compressed', 
    #     CompressedImage, 
    #     segmentation_callback)
    img_sub = rospy.Subscriber(
        '/camera_front/image_color/compressed', 
        CompressedImage, 
        segmentation_callback)

    rospy.spin()

if __name__ == '__main__':
    start_seg_node()


