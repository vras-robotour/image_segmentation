import numpy as np


def rgb_to_label(rgb_image: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert an RGB image representing labels to a label image using a color map.

    Color map example:
        color_map = {
            "0,0,0": 0,
            "255,255,255": 1,
            "0,255,0": 2,
            "255,0,0": 3,
            "0,0,255": 4
        }

    Args:
        rgb_image (np.ndarray): RGB image with shape (H, W, 3).
        color_map (dict): Color map [RGB -> Label].

    Returns:
        np.ndarray: Label image.
    """

    rgb_to_label = np.zeros((256, 256, 256), dtype=np.uint8)
    for color_str, label in color_map.items():
        rgb = list(map(int, color_str.split(',')))
        rgb_to_label[rgb[0], rgb[1], rgb[2]] = label

    # Map each pixel's RGB value to its corresponding label using the lookup table
    label_image = rgb_to_label[rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]]
    return label_image
