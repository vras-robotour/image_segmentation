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
from hydra import compose, initialize

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

#global transform
class segmentation_node():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        rospy.init_node('segmentation_node')#, anonymous=True)
        rospy.loginfo("Starting Segmentation node")
        #cfg = OmegaConf.load("conf/config.yaml")

        #print(self.cfg)
        rospy.loginfo(self.cfg)
        #rospy.loginfo(self.cfg.model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = RoadModel(self.cfg, self.device)

        self.model = RoadModel.load_from_checkpoint(
            self.cfg.ckpt_path, 
            cfg=self.cfg, 
            device=self.device).to(self.device)
        self.model.eval()

        rospy.loginfo(self.cfg.ckpt_path)
        
        current_directory = os.getcwd()

        # Print the current working directory
        rospy.loginfo(current_directory)

        self.img_sub = rospy.Subscriber(
            '/camera_front/image_color/compressed', 
            CompressedImage, 
            self.segmentation_cb)

        self.seg_pub = rospy.Publisher(
            '/camera_front/image_segmentation/compressed',
            CompressedImage, 
            queue_size=10)


    def segmentation_cb(self, msg:CompressedImage):
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
        np_image = sample['image'].float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(np_image)
        prediction = logits.argmax(1).squeeze(0).cpu().numpy()
        rospy.loginfo("Segmentation processed")
        

if __name__ == '__main__':
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
        seg_node = segmentation_node(cfg)
        rospy.spin()
    


