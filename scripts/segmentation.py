#!/usr/bin/env python
import sklearn  # scikit-learn hack to fix the error on jetson

import os
import sys
import io
from datetime import datetime

import torch
import rospy
import numpy as np
from PIL import Image
import albumentations as A
import pytorch_lightning as L
from albumentations.pytorch import ToTensorV2
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize

root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(root_dir)

from src import RoadDataModule, RoadModel, LogPredictionsCallback, val_checkpoint, regular_checkpoint, rgb_to_label


# segmentation.py
# Author: Filip Dasek
# This ros node is used to subscribe images from camera
# and publish a semantic segmentation with three types
# of labels:
#   -feasible      i.e. road     (you want to ride here)
#   -unfeasible    i.e. non-road (you dont want to ride here)
#   -non-important i.e  objects  (you have no information what is behind (e.g people))

#global transform
class SegmentationNode():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.cfg.ckpt_path = rospy.get_param("ckpt_path", "/home/robot/robotour2024/workspace/src/image_segmentation/checkpoints/e51-iou0.60.ckpt")

        #cfg = OmegaConf.load("conf/config.yaml")

        #print(self.cfg)
        #rospy.loginfo(self.cfg)
        #rospy.loginfo(self.cfg.model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = RoadModel(self.cfg, self.device)

        self.model = RoadModel.load_from_checkpoint(
            self.cfg.ckpt_path, 
            cfg=self.cfg, 
            device=self.device).to(self.device)
        self.model.eval()

        rospy.loginfo(self.cfg.ckpt_path)
        rospy.loginfo(self.cfg)
        
        current_directory = os.getcwd()

        # Print the current working directory
        rospy.loginfo(current_directory)

        self.img_sub = rospy.Subscriber(
            '/image_to_segment/compressed', 
            CompressedImage, 
            self.segmentation_cb, queue_size=None)

        self.seg_pub = rospy.Publisher(
            '/segmented_image/compressed',
            CompressedImage, 
            queue_size=10)
        
        self.camera_height = rospy.get_param('camera_height', 550)
        self.camera_width = rospy.get_param('camera_width', 688)


    def segmentation_cb(self, msg:CompressedImage):
        header = msg.header
        time_delay = (rospy.Time.now() - header.stamp).to_sec()

        if time_delay > 0.3:
            rospy.logdebug(f"Threw away image with delay {time_delay}")
            return 
        
        msg_date = datetime.fromtimestamp(header.stamp.to_sec())
        rospy.logdebug(f"Successfully received image published at: {msg_date}")
        now_date = datetime.fromtimestamp(rospy.Time.now().to_sec())
        rospy.logdebug(f"Starting segmentation at: {now_date}")
        time_start = rospy.Time.now()
        
        compressed_data = bytes(msg.data)
        np_image = np.array(Image.open(io.BytesIO(compressed_data)))
        transform = A.Compose([
            A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
            A.Resize(550, 688),
            ToTensorV2()
        ])
        sample = transform(image=np_image)
        tensor_image = sample['image'].float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_image)
        prediction = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        feasible_label = (prediction[..., None] == 1).astype(np.uint8)
        feasible_label[feasible_label == 1] = 200

        infeasible_label = (prediction[..., None] == 2).astype(np.uint8)
        infeasible_label[infeasible_label == 1] = 200

        other_label = (prediction[..., None] == 3).astype(np.uint8)
        other_label[other_label == 1] = 200
        
        np_output_image=np.concatenate((
            infeasible_label, 
            feasible_label, 
            other_label), 
            axis=-1).astype(np.uint8)
        
        out_transform = A.Compose([
            A.Resize(self.camera_height, self.camera_width)
        ])

        np_output_image = out_transform(image=np_output_image)['image']

        pil_image = Image.fromarray(np_output_image)
        #pil_image.save("output.jpg")

        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='JPEG')
        self.seg_pub.publish(
            header=msg.header,
            format="bgr8; jpeg compressed bgr8",
            data=byte_io.getvalue())
        end_date = datetime.fromtimestamp(rospy.Time.now().to_sec())
        rospy.logdebug(f"Published segmentation at: {end_date}")
        rospy.loginfo(f"Segmentation delay: {(rospy.Time.now() - header.stamp).to_sec()}")
        

        

if __name__ == '__main__':
    rospy.init_node('segmentation_node', log_level=rospy.DEBUG)
    rospy.loginfo("Starting Segmentation node")
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
        seg_node = SegmentationNode(cfg)
        rospy.spin()
    


