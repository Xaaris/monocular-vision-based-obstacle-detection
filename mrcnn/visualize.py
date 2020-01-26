"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import colorsys
import random

import cv2
import numpy as np

from model.Box import Box
from model.DetectedObjects import DetectedObjects
from utils.timer import timing


############################################################
#  Visualization
############################################################


def random_colors(number_of_colors, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / number_of_colors, 1, brightness) for i in range(number_of_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    colors = [[int(channel * 255) for channel in color] for color in colors]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


@timing
def draw_instances(image,
                   detected_objects: DetectedObjects,
                   show_mask=True,
                   show_bbox=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    colors: (optional) An array or colors to use with each object
    """
    # Number of instances
    number_of_objects = len(detected_objects.objects)

    masked_image = image.copy()
    for i in range(number_of_objects):
        obj = detected_objects.objects[i]

        if obj.is_present():

            current_instance = obj.get_current_instance()

            color = static_colors[i % max_number_of_colors]

            # Bounding box
            if not np.any(current_instance.roi):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            box: Box = current_instance.roi
            pt1 = (box.x1, box.y1)
            pt2 = (box.x2, box.y2)
            if show_bbox:
                cv2.rectangle(masked_image, pt1=pt1, pt2=pt2, color=color, thickness=1)

            # Label
            class_name = current_instance.class_name
            confidence_score = current_instance.confidence_score
            approximate_distance_in_m = current_instance.approximate_distance() / 100
            obj_track_id = obj.id
            label_text = f"{class_name} {obj_track_id}: {approximate_distance_in_m :.3f}m"
            cv2.putText(masked_image, label_text, (box.x1, box.y1 - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            # Mask
            mask = current_instance.mask
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)

            # Trajectory
            trajectory = obj.get_trajectory()
            if trajectory:
                center = box.get_center()
                arrow_head = (int(center[0] + trajectory[0] * 100), int(center[1] + trajectory[1] * 100))
                cv2.arrowedLine(masked_image, center, arrow_head, (0, 0, 255), 2)

    return masked_image


max_number_of_colors = 50
static_colors = random_colors(max_number_of_colors)
