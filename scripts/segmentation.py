#!/usr/bin/env python
import sklearn  # scikit-learn hack to fix the error on jetson

import os
import sys
import io

import torch
import rospy
import numpy as np
from PIL import Image
import albumentations as A
import pytorch_lightning as L
from albumentations.pytorch import ToTensorV2
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from omegaconf import DictConfig, OmegaConf

root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(root_dir)

from src import RoadDataModule, RoadModel, LogPredictionsCallback, val_checkpoint, regular_checkpoint, rgb_to_label


# segmentation.py
# Author: tvoje mama
# This ros node is used to subscribe images from camera
# and publish a semantic segmentation with three types
# of labels:
#   -feasible      i.e. road     (you want to ride here)
#   -unfeasible    i.e. non-road (you dont want to ride here)
#   -non-important i.e  objects  (you have no information what is behind (e.g people))

global model
global device
global cfg
#global transform


def segmentation_callback(msg:CompressedImage):
    rospy.loginfo("Segmentation in process")
    rospy.loginfo(msg.format)
    compressed_data = bytes(msg.data)
    np_image = np.array(Image.open(io.BytesIO(compressed_data)))
    transform = A.Compose([
        A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
        A.Resize(550, 688),
        ToTensorV2()
    ])
    sample = transform(image=np_image)
    np_image = sample['image'].float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(np_image)
    prediction = logits.argmax(1).squeeze(0).cpu().numpy()
    rospy.loginfo("Segmentation processed")



def start_seg_node():
    global model
    global device
    global cfg
    rospy.init_node('segmentation_node')#, anonymous=True)
    rospy.loginfo("Starting Segmentation node")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load("conf/config.yaml")
    model = RoadModel.load_from_checkpoint(
        cfg.ckpt_path, 
        cfg=cfg, 
        device=device).to(device)
    model.eval()

    rospy.loginfo(cfg.ckpt_path)
    
    current_directory = os.getcwd()

    # Print the current working directory
    rospy.loginfo(current_directory)

    img_sub = rospy.Subscriber(
        '/camera_front/image_color/compressed', 
        CompressedImage, 
        segmentation_callback)

    seg_pub = rospy.Publisher(
        '/camera_front/image_segmentation/compressed',
        CompressedImage, 
        queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    start_seg_node()


