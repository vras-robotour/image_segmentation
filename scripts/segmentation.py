#!/usr/bin/env python

"""
This script is used to perform semantic segmentation on images
from the camera. The segmentation is performed using a trained
model and the segmented image is published to the topic
/segmented_image/compressed.

Author: Filip Dasek

Labels:
    - 0: void (not used for training)
    - 1: feasible (we want to ride here)
    - 2: infeasible (we don't want to ride here)
    - 3: other (non-important objects)
"""

import sklearn  # scikit-learn hack to fix the error on jetson

import os
import io
import sys
from datetime import datetime

import torch
import rospy
import numpy as np
from PIL import Image
import albumentations as A
from cv_bridge import CvBridge
from omegaconf import DictConfig
from hydra import compose, initialize
from sensor_msgs.msg import CompressedImage
from albumentations.pytorch import ToTensorV2

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.append(root_dir)

from src import RoadModel, label_to_rgb

CKPT_PATH = "/home/robot/robotour2024/workspace/src/image_segmentation/checkpoints/e51-iou0.60.ckpt"


class SegmentationNode:
    def __init__(self, cfg: DictConfig):

        # Load the parameters from the launch file
        self.pic_max_age = rospy.get_param('pic_max_age', 0.3)
        self.camera_width = rospy.get_param('camera_width', 688)
        self.camera_height = rospy.get_param('camera_height', 550)

        self.cfg = cfg
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RoadModel.load_from_checkpoint(
            self.cfg.ckpt_path, 
            cfg=self.cfg, 
            device=self.device).to(self.device)
        self.model.eval()

        self.img_sub = rospy.Subscriber(
            '/image_to_segment/compressed', 
            CompressedImage, 
            self.segmentation_cb,
            queue_size=None)

        self.seg_pub = rospy.Publisher(
            '/segmented_image/compressed',
            CompressedImage, 
            queue_size=10)

        rospy.logdebug(f"Camera height: {self.camera_height}")
        rospy.logdebug(f"Camera width: {self.camera_width}")

    def segmentation_cb(self, msg: CompressedImage):
        time_delay = msg_delay(msg)
        if time_delay > self.pic_max_age:
            rospy.logdebug(f"Threw away image with delay {time_delay}")
            return 

        rospy.logdebug(f"Successfully received image published at: " f"{msg_datetime(msg)}")
        rospy.logdebug(f"Starting segmentation at: {now_datetime()}")

        # Extract image from message
        msg_image = np.array(Image.open(io.BytesIO(bytes(msg.data))))

        # Apply transformations
        model_input = inference_transform(msg_image, self.device)

        # Perform inference
        with torch.no_grad():
            logits = self.model(model_input)

        # Select the most probable class
        prediction = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Calculate entropy
        prob = torch.nn.functional.softmax(logits, dim=0)
        entropy = -torch.sum(prob * torch.log(prob), dim=0)
        entropy = entropy.cpu().detach().numpy()

        # Create RGB segmentation image
        output_seg_image = label_to_rgb(prediction, self.cfg.ds.color_map)
        output_seg_image = A.Resize(height=self.camera_height,
                                    width=self.camera_width)(image=output_seg_image)['image']

        output_seg_image = self.bridge.cv2_to_compressed_imgmsg(output_seg_image)
        self.seg_pub.publish(output_seg_image)

        rospy.logdebug(f"Published segmentation at: {now_datetime()}")
        rospy.loginfo(f"Segmentation delay: {msg_delay(msg)}")
        

def msg_delay(msg: CompressedImage) -> float:
    return (rospy.Time.now() - msg.header.stamp).to_sec()


def msg_datetime(msg: CompressedImage) -> datetime:
    return datetime.fromtimestamp(msg.header.stamp.to_sec())


def now_datetime() -> datetime:
    return datetime.fromtimestamp(rospy.Time.now().to_sec())


def inference_transform(image: np.ndarray, device: torch.device) -> torch.Tensor:
    transform = A.Compose([
        A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
        A.Resize(550, 688),
        ToTensorV2()
    ])
    sample = transform(image=image)
    tensor_image = sample['image'].float().unsqueeze(0).to(device)
    return tensor_image


if __name__ == '__main__':
    rospy.init_node('segmentation_node', log_level=rospy.DEBUG)
    rospy.loginfo("Starting Segmentation node")
    ckpt_path = rospy.get_param("ckpt_path", CKPT_PATH)
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["ds=robotour", f"ckpt_path={ckpt_path}"])
        seg_node = SegmentationNode(cfg)
        rospy.spin()
    


