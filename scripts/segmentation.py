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
        
        current_directory = os.getcwd()

        # Print the current working directory
        rospy.loginfo(current_directory)

        self.img_sub = rospy.Subscriber(
            '/image_to_segment/compressed', 
            CompressedImage, 
            self.segmentation_cb)

        self.seg_pub = rospy.Publisher(
            '/segmented_image/compressed',
            CompressedImage, 
            queue_size=10)


    def segmentation_cb(self, msg:CompressedImage):
        # rospy.loginfo("Segmentation in process")
        # rospy.loginfo(msg.format)
        # rospy.loginfo(msg.header)
        #rospy.loginfo("raw shape")
        #rospy.loginfo(msg.data)
        compressed_data = bytes(msg.data)
        np_image = np.array(Image.open(io.BytesIO(compressed_data)))
        # rospy.loginfo("image shape")
        # rospy.loginfo(np_image.shape)
        transform = A.Compose([
            A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
            A.Resize(550, 688),
            ToTensorV2()
        ])
        sample = transform(image=np_image)
        tensor_image = sample['image'].float().unsqueeze(0).to(self.device)
        # rospy.loginfo("tensor shape")
        # rospy.loginfo(tensor_image.shape)
        with torch.no_grad():
            logits = self.model(tensor_image)
        prediction = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        # rospy.loginfo("prediction shape")
        # rospy.loginfo(prediction.shape)
        rospy.loginfo("Segmentation processed")

        
        feasible_label = (prediction[..., None] == 1).astype(np.uint8)
        feasible_label[feasible_label == 1] = 200

        infeasible_label = (prediction[..., None] == 2).astype(np.uint8)
        infeasible_label[infeasible_label == 1] = 200

        other_label = (prediction[..., None] == 3).astype(np.uint8)
        other_label[other_label == 1] = 200
        # count = 0
        # for i in range(mask_1.shape[0]):
        #     for j in range(mask_1.shape[0]):
        #         if mask_1[i, j] == 1:
        #             mask_1[i, j] = 100
        #             count=count+1
        # rospy.loginfo(mask_1)
        # rospy.loginfo(count)
        # Combine masks along the last dimension to create the final array
        # np_output_image = np.zeros((550,688,3), np.uint8)
        # np_output_image[mask_1, 0] = 100
        # np_output_image[mask_2, 1] = 100
        # np_output_image[mask_3, 2] = 100
        np_output_image=np.concatenate((
            infeasible_label, 
            feasible_label, 
            other_label), 
            axis=-1).astype(np.uint8)

        print(np_output_image.shape)  # Output: (550, 688, 3)
        # Convert the image to bytes

        pil_image = Image.fromarray(np_output_image)
        #pil_image.save("output.jpg")

        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='JPEG')
        self.seg_pub.publish(
            format="bgr8; jpeg compressed bgr8",
            data=byte_io.getvalue())

        

if __name__ == '__main__':
    rospy.init_node('segmentation_node')#, anonymous=True)
    rospy.loginfo("Starting Segmentation node")
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
        seg_node = segmentation_node(cfg)
        rospy.spin()
    


