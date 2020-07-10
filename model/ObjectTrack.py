import operator
from typing import Optional

from Constants import MATCHER_TYPE, MatcherType
from matcher.KalmanTracker import KalmanTracker

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

    def add_occurrence(self, new_obj_instance: Optional[ObjectInstance]):
        self.occurrences.append(new_obj_instance)
        center_or_none = None if new_obj_instance is None else new_obj_instance.roi.get_center()
        self.kalman_tracker.update(center_or_none)

    def get_next_position_prediction(self):
        return self.kalman_tracker.predict_next_position()

    def get_position_uncertainty(self):
        return self.kalman_tracker.get_uncertainty()

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
               f"pos uncertainty: {self.get_position_uncertainty()}," \
               f"trajectory: {self.get_trajectory()}"
