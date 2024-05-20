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
import cv2
import sklearn  # scikit-learn hack to fix the error on jetson

import os
import io
import sys
from typing import Tuple
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

        self.cost_pub = rospy.Publisher(
            '/cost_image/compressed',
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

        output_seg_image, cost = self._predict(msg)

        seg_msg = self.bridge.cv2_to_compressed_imgmsg(output_seg_image)
        seg_msg.header = msg.header
        self.seg_pub.publish(seg_msg)

        cost_msg = self.bridge.cv2_to_compressed_imgmsg(cost)
        cost_msg.header = msg.header
        self.cost_pub.publish(cost_msg)

        rospy.logdebug(f"Published segmentation at: {now_datetime()}")
        rospy.loginfo(f"Segmentation delay: {msg_delay(msg)}")

    def _predict(self, msg: CompressedImage) -> Tuple[np.ndarray, np.ndarray]:
        msg_image = np.array(Image.open(io.BytesIO(bytes(msg.data))))

        # Apply transformations
        model_input = inference_transform(msg_image, self.device)

        # Perform inference
        with torch.no_grad():
            logits = self.model(model_input)
            logits = logits.squeeze(0)

        # Select the most probable class
        prediction = logits.argmax(0).cpu().numpy().astype(np.uint8)

        # Calculate entropy
        entropy = compute_entropy(logits).cpu().detach().numpy()

        # Apply the uncertainty function
        cost = apply_uncertainty_function(prediction, entropy)

        # Create RGB segmentation image
        output_seg_image = label_to_rgb(prediction, self.cfg.ds.color_map)

        # Resize the outputs
        output_seg_image = cv2.resize(output_seg_image,
                                      (self.camera_width, self.camera_height),
                                      interpolation=cv2.INTER_NEAREST)
        cost = cv2.resize(cost,
                          (self.camera_width, self.camera_height),
                          interpolation=cv2.INTER_LINEAR)

        # Apply matplotlib colormap
        output_cost_image = np.zeros((cost.shape[0],cost.shape[1], 3), dtype=np.uint8)
        output_cost_image[:,:,0] = (cost * 255).astype(np.uint8)

        return output_seg_image, output_cost_image


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


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.nn.functional.softmax(logits, dim=0)
    entropy = -torch.sum(prob * torch.log(prob), dim=0)
    return entropy


def apply_uncertainty_function(image: np.ndarray, entropy: np.ndarray) -> np.ndarray:
    result_array = np.zeros_like(image, dtype=float)

    # Create boolean masks for each class
    mask0 = (image == 0)
    mask1 = (image == 1)
    mask2 = (image == 2)
    mask3 = (image == 3)

    # Apply the entropy function based on the masks
    result_array[mask0] = 0.5
    result_array[mask1] = 0.5 * entropy[mask1]
    result_array[mask2] = 1 - 0.5 * entropy[mask2]
    result_array[mask3] = 0.5

    return result_array


if __name__ == '__main__':
    rospy.init_node('segmentation_node', log_level=rospy.DEBUG)
    rospy.loginfo("Starting Segmentation node")
    ckpt_path = rospy.get_param("ckpt_path", CKPT_PATH)
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["ds=robotour", f"ckpt_path={ckpt_path}"])
        seg_node = SegmentationNode(cfg)
        rospy.spin()
    


