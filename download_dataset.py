import os

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot
import matplotlib.pyplot as plt
from segments.typing import LabelStatus
from segments.utils import get_semantic_bitmap
from segments import SegmentsClient, SegmentsDataset

from src.utils import label_to_rgb

matplotlib.pyplot.switch_backend('TkAgg')

API_KEY = '4bacc032570420552ef6b038e1a1e8383ac372d9'

DATASET_NAME = 'aleskucera/robotour-tradr'
DATASET_RELEASE = 'v1.0'
FILTERS = [LabelStatus('REVIEWED')]
OUTPUT_DIR = './data/RoboTour'

color_map = {
    "0,0,0": 0,
    "0,255,0": 1,
    "255,0,0": 2,
    "0,0,255": 3
}


def initialize_dataset():
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient(API_KEY)
    release = client.get_release(DATASET_NAME, DATASET_RELEASE)
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=FILTERS)

    return dataset


def save_dataset(dataset):
    # Make sure that the output directory exists and remove all files in it
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, file))

    # Create the Images and Annotations directories
    images_dir = os.path.join(OUTPUT_DIR, 'Images')
    os.makedirs(images_dir, exist_ok=True)
    annotations_dir = os.path.join(OUTPUT_DIR, 'Annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    for sample in tqdm(dataset, desc='Exporting dataset'):
        # Save the image
        name = os.path.splitext(sample['name'])[0]
        image_path = os.path.join(images_dir, name + '.png')
        if isinstance(sample['image'], Image.Image):
            image = sample['image'].convert('RGB')
        else:
            image = Image.fromarray(sample['image'])
        image.save(image_path, format='PNG')

        if isinstance(sample['segmentation_bitmap'], Image.Image):
            semantic_image = sample['segmentation_bitmap'].convert('RGB')
        else:
            if sample['segmentation_bitmap'].dtype != np.uint8:
                semantic_bitmap = sample['segmentation_bitmap'].astype(np.uint8)
                semantic_image = Image.fromarray(semantic_bitmap)
            else:
                semantic_image = Image.fromarray(sample['segmentation_bitmap'])

        semantic_path = os.path.join(annotations_dir, name + '.png')

        # Convert the semantic bitmap array to an Image object and save it
        semantic_image.save(semantic_path, format='PNG')


def show_dataset(dataset):
    for sample in dataset:
        # Print the sample name and list of labeled objects
        print(sample['name'])
        print(sample['annotations'])

        # Show the image
        plt.imshow(sample['image'])
        plt.show()

        # Show the instance segmentation label
        plt.imshow(sample['segmentation_bitmap'])
        plt.show()

        # Show the semantic segmentation label
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        rgb_annotation = label_to_rgb(semantic_bitmap, color_map)
        plt.imshow(rgb_annotation)
        plt.show()


def main():
    dataset = initialize_dataset()
    save_dataset(dataset)
    # show_dataset(dataset)


if __name__ == '__main__':
    main()
