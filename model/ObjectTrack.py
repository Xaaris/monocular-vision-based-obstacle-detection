import math
import operator
from typing import Optional

from Constants import MATCHER_TYPE, MatcherType, INPUT_FPS
from matcher.KalmanTracker import KalmanTracker
from mrcnn.CocoClasses import is_static

if MATCHER_TYPE == MatcherType.SIFT:
    from matcher.SiftMatcher import get_matches
elif MATCHER_TYPE == MatcherType.SURF:
    from matcher.SurfMatcher import get_matches
else:
    from matcher.OrbMatcher import get_matches

from model.ObjectInstance import ObjectInstance


class ObjectTrack:

    def __init__(self, first_obj_occurrence: ObjectInstance):
        self.occurrences: [Optional[ObjectInstance]] = [first_obj_occurrence]
        x, y = first_obj_occurrence.roi.get_center()
        self.kalman_tracker: KalmanTracker = KalmanTracker(x, y)
        self.class_name: str = first_obj_occurrence.class_name
        self.active = True  # Boolean whether this object is considered for matching or not

    def add_occurrence(self, new_obj_instance: Optional[ObjectInstance]):
        self.occurrences.append(new_obj_instance)
        center_or_none = None if new_obj_instance is None else new_obj_instance.roi.get_center()
        self.kalman_tracker.update(center_or_none)
        if self.is_present():
            velocity = self.get_velocity()
            speed = self.calculate_spped_from_velocity(velocity)
            self.get_current_instance().velocity = velocity
            self.get_current_instance().speed = speed


    def get_next_position_prediction(self):
        return self.kalman_tracker.next_position_prediction()

    def get_current_position_prediction(self):
        return self.kalman_tracker.current_position_prediction()

    def get_next_position_uncertainty(self):
        return self.kalman_tracker.next_position_uncertainty()

    def get_current_position_uncertainty(self):
        return self.kalman_tracker.current_position_uncertainty()

    def is_present(self) -> bool:
        """Bool whether object is present in current frame"""
        return len(self.occurrences) > 0 and self.occurrences[-1] is not None

    def was_present_in_last_n_frames(self, n=5) -> bool:
        """Bool whether object was present in the last n frames at least once"""
        last_n_occurrences = self.occurrences[- n:]
        return any(last_n_occurrences)  # checks if any is not None

    def get_current_instance(self) -> ObjectInstance:
        return self.occurrences[-1] if self.is_present() else None

    def similarity_to(self, obj_instance: ObjectInstance, over_n_instances: int = 5) -> float:
        # Check if same class
        if not self.class_name == obj_instance.class_name:
            return 0
        # Check if location checks out
        if not self.kalman_tracker.is_point_in_predicted_area(obj_instance.roi.get_center()):
            return 0
        last_n_occurrences = self.occurrences[- over_n_instances:]
        for occurrence in reversed(last_n_occurrences):
            if occurrence is not None:
                return occurrence.similarity_to(obj_instance)
        # Object did not appear in last 5 frames
        return 0

    def get_velocity(self, over_n_instances: int = INPUT_FPS):
        if not self.active or not self.is_present():
            return None
        elif is_static(self.class_name):
            return 0, 0, 0
        else:
            last_n_occurrences = self.occurrences[- over_n_instances:]  # Getting last (max) n occurrences of this object
            last_n_occurrences_filtered = [x for x in last_n_occurrences if x is not None]  # Filtering for non None values
            last_positions = list(map(lambda x: x.get_3d_position(), last_n_occurrences_filtered))  # mapping to positions of this object
            differences = tuple(map(lambda p1, p2: tuple(map(operator.sub, p1, p2)), last_positions[:-1], last_positions[1:]))  # building pairwise differences
            cumulative_translation = tuple(map(sum, zip(*differences)))  # Adding up all the differences
            smoothed_translation = tuple(map(lambda x: x/len(last_n_occurrences), cumulative_translation))  # dividing by number of instances / frames since first appearance
            translation_in_meter_per_second = tuple(map(lambda x: x * INPUT_FPS, smoothed_translation))
            return translation_in_meter_per_second

    def calculate_spped_from_velocity(self, velocity) -> Optional[float]:
        """Returns the current estimated speed in km/h if a velocity could be calculated beforehand, else None"""
        return None if velocity is None else math.sqrt(sum([e ** 2 for e in velocity])) * 3.6

    def get_trajectory(self, over_n_instances: int = 5) -> tuple:
        """Returns tuple (x,y) of how the object (or rather its matched keypoints) moved on average over the last n frames"""
        if len(self.occurrences) >= 2:
            last_n_instances = list(reversed(self.occurrences[-over_n_instances:]))
            smoothed_translation = (0, 0)
            for i in range(len(last_n_instances) - 1):
                current = last_n_instances[i]
                last = last_n_instances[i + 1]

                cumulative_translation = (0, 0)
                if current is not None and current.descriptors is not None and last is not None and last.descriptors is not None:
                    matches = get_matches(current.descriptors, last.descriptors)
                    if matches:  # one or more matches
                        for match in matches:
                            kp_idx_current = match.queryIdx
                            kp_idx_last = match.trainIdx
                            key_point_current = current.keypoints[kp_idx_current].pt
                            key_point_last = last.keypoints[kp_idx_last].pt
                            translation = tuple(map(operator.sub, key_point_current, key_point_last))  # subtract current point from last one
                            cumulative_translation = tuple(map(operator.add, cumulative_translation, translation))  # add them up
                        cumulative_translation = tuple(map(lambda x: x/len(matches), cumulative_translation))  # div by length

                decayed_translation = tuple(map(lambda x: x/(i + 1), cumulative_translation))  # decay: less impact for older instances
                smoothed_translation = tuple(map(operator.add, smoothed_translation, decayed_translation))  # add them up

            smoothed_translation = tuple(map(lambda x: x/over_n_instances, smoothed_translation))  # div by over_n_instances
            return smoothed_translation

    def __str__(self):
        return f"{len(self.occurrences)} occurrences, " \
               f"current instance: {self.get_current_instance()}, " \
               f"present in last 5 frames: {self.was_present_in_last_n_frames(5)}, " \
               f"next predicted pos: {self.get_next_position_prediction()}, " \
               f"pos uncertainty: {self.get_next_position_uncertainty()}," \
               f"trajectory: {self.get_trajectory()}"
