#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from PIL import Image
import os


from src import RoadDataModule, RoadModel, LogPredictionsCallback, val_checkpoint, regular_checkpoint, rgb_to_label


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

def str_callback(msg):
    rospy.loginfo("jnsdcjncasdj")
    rospy.loginfo(msg.data)



def start_seg_node():
    rospy.init_node('segmentation_node')#, anonymous=True)
    rospy.loginfo("Starting Segmentation node")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #cfg = OmegaConf.load("../conf/config.yaml")
    #model = RoadModel(cfg, device)
    # img_sub = rospy.Subscriber(
    #     '/camera_front/image_color/compressed', 
    #     CompressedImage, 
    #     segmentation_callback)

    current_directory = os.getcwd()

    # Print the current working directory
    rospy.loginfo(current_directory)

    img_sub = rospy.Subscriber(
        '/camera_front/image_color/compressed', 
        CompressedImage, 
        segmentation_callback)
    str_sub = rospy.Subscriber(
        'chatter', 
        String,
        str_callback)
    rospy.spin()

if __name__ == '__main__':
    start_seg_node()


