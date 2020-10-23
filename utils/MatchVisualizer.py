import os

import cv2

from Constants import MATCHER_TYPE, MatcherType
from Constants import ROOT_DIR

if MATCHER_TYPE == MatcherType.SIFT:
    pass
elif MATCHER_TYPE == MatcherType.SURF:
    pass
else:
    pass

from model.DetectedObjects import DetectedObjects
from model.ObjectInstance import create_objects
from mrcnn.Mask_R_CNN_COCO import detect
from utils.image_utils import show
import numpy as np

IMAGE_PATH_1 = os.path.join(ROOT_DIR, "data/imageSet/dynamic/MOT16-14-small/img1/000001.jpg")
IMAGE_PATH_2 = os.path.join(ROOT_DIR, "data/imageSet/dynamic/MOT16-14/img1/000001.jpg")

if __name__ == "__main__":

    detected_objects1 = DetectedObjects()
    detected_objects2 = DetectedObjects()

    image_1 = cv2.imread(IMAGE_PATH_1)
    image_2 = cv2.imread(IMAGE_PATH_2)

    print("detecting objects in image 1")
    result_1 = detect(image_1)
    print("detecting objects in image 2")
    result_2 = detect(image_2)

    objects_1 = create_objects(result_1, image_1)
    objects_2 = create_objects(result_2, image_2)

    if not objects_1 or not objects_2:
        print("Not enough objects found")

    detected_objects1.add_objects(objects_1)
    detected_objects2.add_objects(objects_2)

    for obj_id, detected_object in detected_objects1.objects.items():
        instance = detected_object.get_current_instance()
        image_1_with_kp = np.copy(image_1)
        cv2.drawKeypoints(image_1, instance.keypoints, image_1_with_kp)
        show(image_1_with_kp, await_keypress=True)

    for obj_id, detected_object in detected_objects2.objects.items():
        instance = detected_object.get_current_instance()
        image_2_with_kp = np.copy(image_2)
        cv2.drawKeypoints(image_2, instance.keypoints, image_2_with_kp)
        show(image_2_with_kp, await_keypress=True)
