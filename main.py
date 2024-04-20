import sklearn  # scikit-learn hack to fix the error on jetson

import torch
import hydra
import numpy as np
from PIL import Image
import albumentations as A
import pytorch_lightning as L
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import WandbLogger

from src import RoadDataModule, RoadModel, LogPredictionsCallback, val_checkpoint, regular_checkpoint, rgb_to_label


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadModel(cfg, device)
    datamodule = RoadDataModule(cfg)

    wandb_logger = WandbLogger(project="road-segmentation", name=cfg.run_name)

    trainer = L.Trainer(max_epochs=cfg.train.max_epochs,
                        accelerator="gpu",
                        devices=1,
                        logger=wandb_logger,
                        callbacks=[
                            LogPredictionsCallback(),
                            val_checkpoint,
                            regular_checkpoint
                        ])

    if cfg.action == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    elif cfg.action == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    elif cfg.action == "predict":
        # Load the trained model
        model = RoadModel.load_from_checkpoint(cfg.ckpt_path, cfg=cfg, device=device).to(device)
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
        fig = plt.figure(figsize=(12, 4))

        # Plot the prediction next to the label
        plt.subplot(1, 3, 1)
        plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')  # Remove axes

        plt.subplot(1, 3, 2)
        plt.imshow(label[0].cpu().numpy())
        plt.axis('off')  # Remove axes

        plt.subplot(1, 3, 3)
        plt.imshow(prediction)
        plt.axis('off')  # Remove axes

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()

        # Save the plot
        plt.savefig('prediction.png')
    else:
        raise ValueError(f"Unknown action: {cfg.action}")


if __name__ == "__main__":
    main()
