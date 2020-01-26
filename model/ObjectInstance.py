from dataclasses import dataclass, field

import cv2
import numpy as np
from cv2.cv2 import KeyPoint

from mrcnn.Mask_R_CNN_COCO import get_class_name_for_id

from Constants import MATCHER_TYPE, MatcherType
if MATCHER_TYPE == MatcherType.SIFT:
    from matcher.SiftMatcher import average_descriptor_distance, get_keypoints_and_descriptors_for_object
elif MATCHER_TYPE == MatcherType.SURF:
    from matcher.SurfMatcher import average_descriptor_distance, get_keypoints_and_descriptors_for_object
else:
    from matcher.OrbMatcher import average_descriptor_distance, get_keypoints_and_descriptors_for_object

from model.Box import Box
from utils.timer import timing

LOCATION_VARIANCE_THRESHOLD = 0.1  # how far an object can move between two frames before it is recognized as a new obj
CONFIDENCE_SCORE_THRESHOLD = 0.8  # Only objects with higher confidence are taken into account


@dataclass
class ObjectInstance:
    class_name: str
    roi: Box
    confidence_score: float
    mask: [[]]
    keypoints: [KeyPoint] = field(default_factory=list)
    descriptors: np.ndarray = None

    def similarity_to(self, obj_instance) -> float:
        # Check if same class
        if not self.class_name == obj_instance.class_name:
            return 0
        # Check if both have descriptors
        if self.descriptors is None or obj_instance.descriptors is None:
            return 0
        # Check if instances are in similar location
        this_position_x, this_position_y = self.roi.get_position_in_image()
        other_position_x, other_position_y = obj_instance.roi.get_position_in_image()
        if abs(this_position_x - other_position_x) > LOCATION_VARIANCE_THRESHOLD or \
                abs(this_position_y - other_position_y) > LOCATION_VARIANCE_THRESHOLD:
            return 0
        # Check if descriptors match
        average_distance = average_descriptor_distance(self.descriptors, obj_instance.descriptors)
        normalized_distance = average_distance / 100  # normalize?
        return max(0.0, 1 - normalized_distance)


@timing
def create_objects(result, frame) -> [ObjectInstance]:
    objects = []
    number_of_results = result["class_ids"].shape[0]

    # Convert frame to grayscale for matchers
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(number_of_results):

        confidence_score = result["scores"][i]
        if confidence_score > CONFIDENCE_SCORE_THRESHOLD:  # filter objects
            class_name = get_class_name_for_id(result["class_ids"][i])

            roi = result["rois"][i]
            y1, x1, y2, x2 = roi
            box = Box(x1, y1, x2, y2)

            mask = result["masks"][:, :, i]

            keypoints, descriptors = get_keypoints_and_descriptors_for_object(frame_gray, mask)
            # show(drawKeypoints(frame, keypoints, None))
            detected_object = ObjectInstance(class_name, box, confidence_score, mask, keypoints, descriptors)
            objects.append(detected_object)

    return objects
