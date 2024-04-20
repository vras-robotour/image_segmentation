#import sklearn  # scikit-learn hack to fix the error on jetson

#import torch
#import hydra
#import numpy as np
#from PIL import Image
#import albumentations as A
#import pytorch_lightning as L
#import matplotlib.pyplot as plt
#from omegaconf import DictConfig, OmegaConf
#from albumentations.pytorch import ToTensorV2
#from pytorch_lightning.loggers import WandbLogger

import rospy
#from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

#from src import RoadDataModule, RoadModel, LogPredictionsCallback, val_checkpoint, regular_checkpoint, rgb_to_label

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


    img_sub = rospy.Subscriber(
        '/camera_front/image_color/compressed', 
        CompressedImage, 
        segmentation_callback)

    rospy.spin()

if __name__ == '__main__':
    start_seg_node()


