from typing import Dict

from model.ObjectTrack import ObjectTrack

SAMENESS_THRESHOLD = 0.3  # 0 = match all, 1 match basically none
KEEP_TRACK_OF_OBJS_FOR_N_FRAMES = 5


class DetectedObjects:

    def __init__(self):
        self.nextObjectID = 0
        self.objects: Dict[int, ObjectTrack] = dict()

    def get_next_id(self) -> int:
        self.nextObjectID += 1
        return self.nextObjectID

    def add_objects(self, new_objects):

        touched_object_ids = set()
        for new_obj in new_objects:
            new_or_added_to_obj_id = self._add_object(new_obj, touched_object_ids)
            touched_object_ids.add(new_or_added_to_obj_id)

        # add None to all obj_tracks that have not found a new instance
        for obj_id, obj_track in self.objects.items():
            if obj_id not in touched_object_ids:
                obj_track.add_occurrence(None)

        self._delete_old_object_tracks()

    def _add_object(self, new_obj_instance, already_touched_obj_ids, verbose=False):

        if verbose:
            print(f"\nNew Object: {new_obj_instance.class_name}, {new_obj_instance.roi}")

        obj_id_with_sufficient_similarity = self._find_existing_matching_obj_track(already_touched_obj_ids, new_obj_instance, verbose)

        if obj_id_with_sufficient_similarity:
            # Add to existing object
            obj_with_highest_similarity = self.objects[obj_id_with_sufficient_similarity]
            obj_with_highest_similarity.add_occurrence(new_obj_instance)
            return obj_id_with_sufficient_similarity
        else:
            # Add as new object
            new_obj_track = ObjectTrack(new_obj_instance)
            new_obj_id = self.get_next_id()
            self.objects[new_obj_id] = new_obj_track
            return new_obj_id

    def _find_existing_matching_obj_track(self, already_touched_obj_ids, new_obj_instance, verbose):
        obj_id_with_sufficient_similarity = None
        highest_similarity = 0
        for obj_id, obj_track in self.objects.items():
            if obj_id not in already_touched_obj_ids:
                similarity_to_current_obj = obj_track.similarity_to(new_obj_instance, over_n_instances=KEEP_TRACK_OF_OBJS_FOR_N_FRAMES)
                if similarity_to_current_obj > highest_similarity:
                    highest_similarity = similarity_to_current_obj
                    if similarity_to_current_obj > SAMENESS_THRESHOLD:
                        obj_id_with_sufficient_similarity = obj_id
        return obj_id_with_sufficient_similarity

    def _delete_old_object_tracks(self):
        ids_to_delete = [key for key, obj_track in self.objects.items() if not obj_track.was_present_in_last_n_frames(KEEP_TRACK_OF_OBJS_FOR_N_FRAMES)]
        for key in ids_to_delete:
            del self.objects[key]
