import sklearn  # scikit-learn hack to fix the error on jetson

import torch
#import hydra
import numpy as np
from PIL import Image
import albumentations as A
import pytorch_lightning as L
#import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch import ToTensorV2
#from pytorch_lightning.loggers import WandbLogger

import rospy
#from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

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



def start_seg_node():
    rospy.init_node('segmentation_node')#, anonymous=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = RoadModel(cfg, device)
    datamodule = RoadDataModule(cfg)

    #wandb_logger = WandbLogger(project="road-segmentation", name=cfg.run_name)

    # trainer = L.Trainer(max_epochs=cfg.train.max_epochs,
    #                     accelerator="gpu",
    #                     devices=1,
    #                     logger=wandb_logger,
    #                     callbacks=[
    #                         LogPredictionsCallback(),
    #                         val_checkpoint,
    #                         regular_checkpoint
    #                     ])

    # if cfg.action == "train":
    #     trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # elif cfg.action == "test":
    #     trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # elif cfg.action == "predict":
        # Load the trained model
    model = RoadModel.load_from_checkpoint(
        "/home/robot/robotour2024/road-segmentation/checkpoints/e51-iou0.60.ckpt", 
        cfg=cfg, device=device).to(device)
    model.eval()

    # Load an image and its label
    image_path = 'data/RUGD/Images/creek_00001.png'
    label_path = 'data/RUGD/Annotations/creek_00001.png'

    image = np.array(Image.open(image_path).convert('RGB')) / 255.0
    label = np.array(Image.open(label_path).convert('RGB'))

    # Process the label image
    label = rgb_to_label(label, cfg.ds.color_map)
    train_map = OmegaConf.to_container(cfg.ds.train_map)
    label = np.vectorize(train_map.get)(label)

    # Apply the same transformations as during training
    transform = A.Compose([
        A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
        A.Resize(550, 688),
        ToTensorV2()
    ])
    sample = transform(image=image, mask=label)
    image = sample['image'].float().unsqueeze(0).to(device)
    label = sample['mask'].long().unsqueeze(0).to(device)

    # Predict the label image
    with torch.no_grad():
        logits = model(image)
    prediction = logits.argmax(1).squeeze(0).cpu().numpy()

    # Plot the image, label, and prediction
    #fig = plt.figure(figsize=(12, 4))

    # Plot the prediction next to the label
    # plt.subplot(1, 3, 1)
    # plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    # plt.axis('off')  # Remove axes

    # plt.subplot(1, 3, 2)
    # plt.imshow(label[0].cpu().numpy())
    # plt.axis('off')  # Remove axes

    # plt.subplot(1, 3, 3)
    # plt.imshow(prediction)
    # plt.axis('off')  # Remove axes

    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()

    # # Save the plot
    # plt.savefig('prediction.png')
    #else:
    #    raise ValueError(f"Unknown action: {cfg.action}")




    img_sub = rospy.Subscriber(
        '/camera_front/image_color/compressed', 
        CompressedImage, 
        segmentation_callback)
    seg_pub = rospy.Publisher(
        '/segmentation/image', 
        CompressedImage, 
        queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    start_seg_node()


