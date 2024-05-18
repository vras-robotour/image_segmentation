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


def label_to_rgb(label_image: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert a label image to an RGB image using a color map.

    Args:
        label_image (np.ndarray): Label image with shape (H, W).
        color_map (dict): Color map [RGB -> Label].

    Returns:
        np.ndarray: RGB image with shape (H, W, 3).
    """

    # Create a dictionary to hold the inverse mapping from label to RGB
    label_to_rgb_dict = {}
    for rgb_str, label in color_map.items():
        rgb = list(map(int, rgb_str.split(',')))
        label_to_rgb_dict[label] = rgb

    # Initialize an empty array for the output RGB image
    rgb_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)

    # Use advanced indexing to fill the RGB image with colors from the color map
    for label in label_to_rgb_dict.keys():
        rgb_values = label_to_rgb_dict[label]
        mask = (label_image == label)
        rgb_image[mask] = rgb_values

    return rgb_image

if __name__ == "__main__":
    # Define a color map for the RUGD dataset
    color_map = {
        "0,0,0": 0,
        "0,255,0": 1,
        "255,0,0": 2,
        "0,0,255": 3
    }

    # Load an example label image
    rgb_image = np.array([
        [[0, 0, 0], [0, 0, 255]],
        [[0, 255, 0], [255, 0, 0]]
    ], dtype=np.uint8)

    label_image = np.array([
        [0, 3],
        [1, 2]
    ], dtype=np.uint8)

    # Convert the RGB image to a label image
    converted_label_image = rgb_to_label(rgb_image, color_map)
    print(converted_label_image)

    # Convert the label image back to an RGB image
    converted_rgb_image = label_to_rgb(label_image, color_map)
    print(converted_rgb_image)