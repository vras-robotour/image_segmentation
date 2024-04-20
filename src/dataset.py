import os
from typing import Tuple

import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
import pytorch_lightning as L
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from src.utils import rgb_to_label


class RoadDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_transform = A.Compose([
            A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
            A.Resize(550, 688),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(always_apply=False,
                                       p=0.5,
                                       brightness_limit=(-0.1, 0.1),
                                       contrast_limit=(-0.1, 0.1),
                                       brightness_by_max=True),
            A.GaussNoise(always_apply=False,
                         p=0.5,
                         var_limit=(0.05, 0.2),
                         per_channel=True, mean=0.0),
            A.PixelDropout(p=0.01),
            ToTensorV2(),
        ])

        self.test_transform = A.Compose([
            A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
            A.Resize(550, 688),
            ToTensorV2(),
        ])

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.predict_ds = None

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_ds = RoadDataset(self.cfg, self.train_transform, split="train")
            self.val_ds = RoadDataset(self.cfg, self.test_transform, split="val")

        if stage == "test" or stage is None:
            self.test_ds = RoadDataset(self.cfg, self.test_transform, split="test")

        if stage == "predict" or stage is None:
            self.predict_ds = RoadDataset(self.cfg, self.test_transform, split="test")

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.train.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.train.num_workers)


class RoadDataset(Dataset):
    def __init__(self, cfg: DictConfig, transform: A.Compose = None, split: str = "train"):
        assert split in ["train", "val", "test"], f"Split {split} is not supported"
        self.cfg = cfg
        self.split = split
        self.path = cfg.ds.path
        self.transform = transform if transform else A.Compose([ToTensorV2()])
        self.color_map = cfg.ds.color_map if hasattr(cfg.ds, "color_map") else None
        self.train_map = OmegaConf.to_container(cfg.ds.train_map) if hasattr(cfg.ds, "train_map") else None

        self.images, self.labels = self._load_data()

    def _load_data(self) -> Tuple[list, list]:
        """Load the names of the images and labels from the split file.

        Returns:
            Tuple[list, list]: List of image and label paths.
        """

        images_dir = os.path.join(self.path, "Images")
        labels_dir = os.path.join(self.path, "Annotations")

        split_file = os.path.join(self.path, f"{self.split}.txt")

        with open(split_file, 'r') as f:
            samples = f.readlines()
            samples = [sample.strip() for sample in samples]

        images = [os.path.join(images_dir, sample) for sample in samples]
        labels = [os.path.join(labels_dir, sample) for sample in samples]

        # Sort the images and labels
        images.sort()
        labels.sort()

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and label at the specified index. This is done by following these steps:

        1. Load the image and label from the disk.
        2. If the image format is uint8, normalize it to [0, 1].
        3. If the label is an RGB image, convert it to a label image using the color map.
        4. If a train map is provided, map the labels to the desired classes.
        5. Apply the transformation to the image and label.
        6. Convert the image and label to tensors and permute the dimensions to (C, H, W).

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and label tensors.
        """

        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = np.array(Image.open(image_path).convert('RGB'))

        if os.path.exists(label_path):
            label = np.array(Image.open(label_path))
        else:
            label = np.zeros_like(image)

        if self.cfg.ds.name == "rugd":
            image = image / 255.0

        if len(label.shape) == 3:
            assert self.color_map is not None, "Color map must be provided for RGB labels"
            label = rgb_to_label(label, self.color_map)

        if self.train_map is not None:
            label = np.vectorize(self.train_map.get)(label)

        sample = self.transform(image=image, mask=label)

        image = sample['image'].float()
        label = sample['mask'].long()

        return image, label

    def compute_mean_and_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and standard deviation of the image samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the samples.
        """

        mean = torch.zeros(3)
        std = torch.zeros(3)
        nb_samples = 0
        for data, _ in tqdm(self):
            data = data.view(data.size(0), -1)
            mean += data.mean(1)
            std += data.std(1)
            nb_samples += 1

        mean /= nb_samples
        std /= nb_samples
        return mean, std


def visualize_samples(ds_name: str, num_samples: int = 3) -> None:
    """Visualize random samples from the dataset.

    Args:
        ds_name (str): Name of the dataset. (Should be in the conf/ds directory)
        num_samples (int): Number of samples to visualize. (default: 3)

    Returns:
        None
    """

    ds_file = f"conf/ds/{ds_name}.yaml"
    cfg = {"ds": {}}

    with open(ds_file, 'r') as f:
        cfg["ds"] = yaml.safe_load(f)

    cfg = OmegaConf.create(cfg)

    transform = A.Compose([
        A.Normalize(mean=cfg.ds.mean, std=cfg.ds.std, max_pixel_value=1.0),
        A.Resize(550, 688),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(always_apply=False,
                                   p=0.5,
                                   brightness_limit=(-0.1, 0.1),
                                   contrast_limit=(-0.1, 0.1),
                                   brightness_by_max=True),
        A.GaussNoise(always_apply=False,
                     p=0.5,
                     var_limit=(0.05, 0.2),
                     per_channel=True, mean=0.0),
        A.PixelDropout(p=0.01),
    ])

    ds = RoadDataset(cfg, transform)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))

    for i in range(num_samples):
        idx = np.random.randint(len(ds))
        image, label = ds[idx]

        # Convert tensors to numpy arrays
        image = image.permute(1, 2, 0).numpy()
        label = label.numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(label, cmap='gray')
        axes[i, 1].set_title('Label')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def rugd_preprocessing(path: str, new_path: str, train_ratio: float = 0.8) -> None:
    """Preprocess the RUGD dataset by moving all the samples
    and labels to a single directory and creating split files.

    Args:
        path (str): Path to the RUGD dataset.
        new_path (str): Path to save the preprocessed dataset.
        train_ratio (float): Ratio of the training set. (default: 0.8)

    Returns:
        None
    """

    dirs = ["creek", "park-1", "park-2", "park-8", "trail", "trail-3", "trail-4", "trail-5", "trail-6", "trail-7",
            "trail-9", "trail-10", "trail-11", "trail-12", "trail-13", "trail-14", "trail-15", "village"]
    images_dir = os.path.join(path, "RUGD_frames-with-annotations")
    labels_dir = os.path.join(path, "RUGD_annotations")

    # Create the new directories
    new_images_dir = os.path.join(new_path, "Images")
    new_labels_dir = os.path.join(new_path, "Annotations")

    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)

    train_file = os.path.join(new_path, "train.txt")
    val_file = os.path.join(new_path, "val.txt")
    test_file = os.path.join(new_path, "test.txt")

    # Clear the files
    open(train_file, 'w').close()
    open(val_file, 'w').close()
    open(test_file, 'w').close()

    for dir in dirs:
        # Get the samples in the directory (must end with .png)
        samples = [sample for sample in os.listdir(os.path.join(labels_dir, dir)) if sample.endswith(".png")]

        # Copy the images and labels to the new directories
        for sample in tqdm(samples):
            image_path = os.path.join(images_dir, dir, sample)
            label_path = os.path.join(labels_dir, dir, sample)

            new_image_path = os.path.join(new_images_dir, sample)
            new_label_path = os.path.join(new_labels_dir, sample)

            # Copy the images and labels
            os.system(f"cp {image_path} {new_image_path}")
            os.system(f"cp {label_path} {new_label_path}")

        # Create subsequences of the samples - each subsequence contains 50 samples
        subsequences = [samples[i:i + 50] for i in range(0, len(samples), 50)]

        # Shuffle the subsequences
        np.random.shuffle(subsequences)

        # Split the subsequences into train and validation
        split_idx = int(len(subsequences) * train_ratio)
        train_subsequences = subsequences[:split_idx]
        val_subsequences = subsequences[split_idx:]

        for subsequence in train_subsequences:
            with open(train_file, 'a') as f:
                f.writelines([f"{sample}\n" for sample in subsequence])

        for subsequence in val_subsequences:
            with open(val_file, 'a') as f:
                f.writelines([f"{sample}\n" for sample in subsequence])

        for subsequence in val_subsequences:
            with open(test_file, 'a') as f:
                f.writelines([f"{sample}\n" for sample in subsequence])


def compute_mean_and_std(ds_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean and standard deviation of the dataset.

    Args:
        ds_name (str): Name of the dataset. (Should be in the conf/ds directory)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the dataset.
    """

    ds_file = f"conf/ds/{ds_name}.yaml"
    cfg = {"ds": {}}

    with open(ds_file, 'r') as f:
        cfg["ds"] = yaml.safe_load(f)

    cfg = OmegaConf.create(cfg)

    ds = RoadDataset(cfg)

    mean, std = ds.compute_mean_and_std()

    return mean, std


if __name__ == "__main__":
    # rugd_path = "/home/ales/school/robolab/road-segmentation/data/RUGD_old"
    # new_rugd_path = "/home/ales/school/robolab/road-segmentation/data/RUGD"

    # rugd_preprocessing(rugd_path, new_rugd_path)

    # compute_mean_and_std("rugd")

    visualize_samples("rugd")
