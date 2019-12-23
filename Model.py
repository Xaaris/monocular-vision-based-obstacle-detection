from dataclasses import dataclass, field

import numpy as np
from cv2 import cv2
from cv2.cv2 import KeyPoint
from collections import Counter

class DetectedObjects:
    pass
    # WIP: Build a structure like a Buffer with something like 5 spaces for every object.
    # It holds the occurrences of this object of the last 5 frames (None if it wasn't found).
    # If a buffer is only None, it gets dropped.
    # New detections are always checked against last (N?) occurrences to see if obj already exists
    # and then added to buffer or a new buffer is created.


    #
    # objects = {}
    # object_type_counter = Counter
    #
    # def add_object(self, obj):
    #
    #     obj_with_highest_similarity
    #     self.objects.values()



@dataclass
class Box:
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class ObjectInstance:
    class_name: str
    roi: Box
    confidence_score: float
    mask: [[]]
    keypoints: [KeyPoint] = field(default_factory=list)
    descriptors: np.ndarray = None

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    def similarity_to(self, obj_instance) -> float:
        if not isinstance(obj_instance, ObjectInstance):
            return 0
        if not self.class_name == obj_instance.class_name:
            return 0
        if self.descriptors is None or obj_instance.descriptors is None:
            return 0
        matches = self.matcher.match(self.descriptors, obj_instance.descriptors, None)
        # sum distances
        total_distance = sum([d.distance for d in matches])
        # divide by number of matches
        average_distance = total_distance / len(matches)
        # normalize?
        normalized_distance = average_distance / 100
        print(f"normalized_distance={normalized_distance}")
        return max(0, 1 - normalized_distance)
