from dataclasses import dataclass, field

import numpy as np
from cv2 import cv2
from cv2.cv2 import KeyPoint


class DetectedObjects:
    SAMENESS_THRESHHOLD = 0.5

    objects = []

    def add_objects(self, new_objects):

        touched_objects = []
        for new_obj in new_objects:
            touched_objects.append(self.add_object(new_obj))

        # add None to all obj_tracks that have not found a new instance
        for obj_track in self.objects:
            if obj_track not in touched_objects:
                obj_track.occurrences.append(None)

    def add_object(self, new_obj_instance):

        obj_with_highest_similarity = None
        highest_similarity = 0
        for obj_track in self.objects:
            similarity_to_current_obj = obj_track.similarity_to(new_obj_instance)
            if similarity_to_current_obj > highest_similarity:
                highest_similarity = similarity_to_current_obj
                obj_with_highest_similarity = obj_track

        if highest_similarity > self.SAMENESS_THRESHHOLD:
            obj_with_highest_similarity.occurrences.append(new_obj_instance)
            return obj_with_highest_similarity
        else:
            new_obj_track = ObjectTrack([new_obj_instance])
            self.objects.append(new_obj_track)
            return new_obj_track


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
        # print(f"normalized_distance={normalized_distance}")
        return max(0, 1 - normalized_distance)


@dataclass
class ObjectTrack:
    occurrences: [ObjectInstance] = field(default_factory=list)

    def similarity_to(self, obj_instance) -> float:
        last_n_occurrences = self.occurrences[-5:]
        for occurrence in reversed(last_n_occurrences):
            if occurrence is not None:
                return occurrence.similarity_to(obj_instance)
        # Object did not appear in last 5 frames
        return 0
