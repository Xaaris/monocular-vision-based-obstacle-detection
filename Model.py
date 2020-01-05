import operator
from dataclasses import dataclass, field

import numpy as np
from cv2 import cv2
from cv2.cv2 import KeyPoint


class DetectedObjects:
    SAMENESS_THRESHHOLD = 0.4

    objects = []

    def add_objects(self, new_objects):

        touched_objects = []
        for new_obj in new_objects:
            touched_objects.append(self.add_object(new_obj))

        # add None to all obj_tracks that have not found a new instance
        for obj_track in self.objects:
            if obj_track not in touched_objects:
                obj_track.occurrences.append(None)

    def add_object(self, new_obj_instance, verbose=False):

        if verbose:
            print(f"\nNew Object: {new_obj_instance.class_name}, {new_obj_instance.roi}")

        obj_with_highest_similarity = None
        highest_similarity = 0
        for i, obj_track in enumerate(self.objects):
            similarity_to_current_obj = obj_track.similarity_to(new_obj_instance)
            if verbose:
                if obj_track.is_present():
                    print(f"Similarity to obj {i}: {similarity_to_current_obj:.3f}, {obj_track.get_current_instance().class_name}, {obj_track.get_current_instance().roi}")
                else:
                    print(f"Similarity to obj {i}: {similarity_to_current_obj:.3f}, Not present in current frame")
            if similarity_to_current_obj > highest_similarity:
                highest_similarity = similarity_to_current_obj
                obj_with_highest_similarity = obj_track

        if highest_similarity > self.SAMENESS_THRESHHOLD:
            if verbose:
                if obj_with_highest_similarity.is_present():
                    print(f"Object existed before: {highest_similarity:.3f} {obj_with_highest_similarity.get_current_instance().class_name}, {obj_with_highest_similarity.get_current_instance().roi}")
                else:
                    print(f"Object existed before but was not present in this frame")
            obj_with_highest_similarity.occurrences.append(new_obj_instance)
            return obj_with_highest_similarity
        else:
            if verbose:
                print(f"New Object!")
            new_obj_track = ObjectTrack([new_obj_instance])
            self.objects.append(new_obj_track)
            return new_obj_track


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def get_width(self) -> int:
        return self.x2 - self.x1

    def get_height(self) -> int:
        return self.y2 - self.y1

    def get_center(self) -> tuple:
        return int(self.x1 + self.get_width() / 2), int(self.y1 + self.get_height() / 2)


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
        this_center_x, this_center_y = self.roi.get_center()
        other_center_x, other_center_y = obj_instance.roi.get_center()
        if abs(this_center_x - other_center_x) > 100 or abs(this_center_y - other_center_y) > 100:
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

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    GOOD_MATCH_PERCENT = 0.75

    def is_present(self) -> bool:
        return len(self.occurrences) > 0 and self.occurrences[-1] is not None

    def get_current_instance(self) -> ObjectInstance:
        return self.occurrences[-1] if self.is_present() else None

    def similarity_to(self, obj_instance, over_n_instances: int = 5) -> float:
        last_n_occurrences = self.occurrences[-over_n_instances:]
        for occurrence in reversed(last_n_occurrences):
            if occurrence is not None:
                return occurrence.similarity_to(obj_instance)
        # Object did not appear in last 5 frames
        return 0

    def get_trajectory(self, over_n_instances: int = 5) -> tuple:
        if len(self.occurrences) >= 2:
            last_n_instances = list(reversed(self.occurrences[-over_n_instances:]))
            smoothed_translation = (0, 0)
            for i in range(len(last_n_instances) - 1):
                current = last_n_instances[i]
                last = last_n_instances[i + 1]

                cumulative_translation = (0, 0)
                if current is not None and current.descriptors is not None and last is not None and last.descriptors is not None:
                    matches = self.matcher.match(current.descriptors, last.descriptors, mask=None)
                    # Sort matches by score
                    matches.sort(key=lambda x: x.distance, reverse=False)

                    # Remove not so good matches
                    num_good_matches = int(len(matches) * self.GOOD_MATCH_PERCENT)
                    matches = matches[:num_good_matches]
                    if matches:  # one or more matches
                        for match in matches:
                            kp_idx_current = match.queryIdx
                            kp_idx_last = match.trainIdx
                            key_point_current = current.keypoints[kp_idx_current].pt
                            key_point_last = last.keypoints[kp_idx_last].pt
                            translation = tuple(map(operator.sub, key_point_current, key_point_last))  # subtract current point from last one
                            cumulative_translation = tuple(map(operator.add, cumulative_translation, translation))  # add them up
                        cumulative_translation = tuple(map(lambda x: x/len(matches), cumulative_translation))  # div by length
                smoothed_translation = tuple(map(operator.add, smoothed_translation, cumulative_translation))  # add them up
            smoothed_translation = tuple(map(lambda x: x/over_n_instances, smoothed_translation))  # div by over_n_instances
            return smoothed_translation

