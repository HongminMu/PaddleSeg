# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
from PIL import Image as PILImage


def visualize(image, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.

    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask

#
# def get_color_map_list(num_classes, custom_color=None):
#     """
#     Returns the color map for visualizing the segmentation mask,
#     which can support arbitrary number of classes.
#
#     Args:
#         num_classes (int): Number of classes.
#         custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
#
#     Returns:
#         (list). The color map.
#     """
#
#     num_classes += 1
#     color_map = num_classes * [0, 0, 0]
#     for i in range(0, num_classes):
#         j = 0
#         lab = i
#         while lab:
#             color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
#             color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
#             color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
#             j += 1
#             lab >>= 3
#     color_map = color_map[3:]
#
#     if custom_color:
#         color_map[:len(custom_color)] = custom_color
#     return color_map

def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    # Cityscapes predefined colors for 19 classes
    CITYSCAPES_PALETTE = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]

    color_map = []

    # First, use the custom color or Cityscapes palette
    if custom_color:
        color_map.extend(custom_color)
    else:
        color_map.extend(CITYSCAPES_PALETTE)

    # Then, use the default way to generate rest of the colors
    while len(color_map) < num_classes:
        idx = len(color_map)
        j = 0
        lab = idx
        r = 0
        g = 0
        b = 0
        while lab:
            r |= (((lab >> 0) & 1) << (7 - j))
            g |= (((lab >> 1) & 1) << (7 - j))
            b |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
        color_map.append((r, g, b))

    return [item for sublist in color_map for item in sublist][:num_classes * 3]

def paste_images(image_list):
    """
    Paste all image to a image.
    Args:
        image_list (List or Tuple): The images to be pasted and their size are the same.
    Returns:
        result_img (PIL.Image): The pasted image.
    """
    assert isinstance(image_list,
                      (list, tuple)), "image_list should be a list or tuple"
    assert len(
        image_list) > 1, "The length of image_list should be greater than 1"

    pil_img_list = []
    for img in image_list:
        if isinstance(img, str):
            assert os.path.exists(img), "The image is not existed: {}".format(
                img)
            img = PILImage.open(img)
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        pil_img_list.append(img)

    sample_img = pil_img_list[0]
    size = sample_img.size
    for img in pil_img_list:
        assert size == img.size, "The image size in image_list should be the same"

    width, height = sample_img.size
    result_img = PILImage.new(sample_img.mode,
                              (width * len(pil_img_list), height))
    for i, img in enumerate(pil_img_list):
        result_img.paste(img, box=(width * i, 0))

    return result_img
