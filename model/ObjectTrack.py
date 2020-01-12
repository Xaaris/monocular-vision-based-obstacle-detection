import operator
from dataclasses import dataclass, field

from ORB import get_matches
from model.ObjectInstance import ObjectInstance


@dataclass
class ObjectTrack:
    id: int
    occurrences: [ObjectInstance] = field(default_factory=list)

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
                    matches = get_matches(current.descriptors, last.descriptors, max_distance=100)
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
