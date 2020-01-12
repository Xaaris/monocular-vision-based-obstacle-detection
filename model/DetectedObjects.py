from model.IdGenerator import get_next_id_for
from model.ObjectTrack import ObjectTrack

SAMENESS_THRESHOLD = 0.4


class DetectedObjects:
    objects: [ObjectTrack] = []

    def add_objects(self, new_objects):

        touched_objects = []
        for new_obj in new_objects:
            new_or_added_to_obj_track = self.add_object(new_obj, touched_objects)
            touched_objects.append(new_or_added_to_obj_track)

        # add None to all obj_tracks that have not found a new instance
        for obj_track in self.objects:
            if obj_track not in touched_objects:
                obj_track.occurrences.append(None)

    def add_object(self, new_obj_instance, already_touched_obj_tracks, verbose=False):

        if verbose:
            print(f"\nNew Object: {new_obj_instance.class_name}, {new_obj_instance.roi}")

        obj_with_highest_similarity = None
        highest_similarity = 0
        for i, obj_track in enumerate(self.objects):
            if obj_track not in already_touched_obj_tracks:
                similarity_to_current_obj = obj_track.similarity_to(new_obj_instance)
                if verbose:
                    if obj_track.is_present():
                        print(f"Similarity to obj {i}: {similarity_to_current_obj:.3f}, {obj_track.get_current_instance().class_name}, {obj_track.get_current_instance().roi}")
                    else:
                        print(f"Similarity to obj {i}: {similarity_to_current_obj:.3f}, Not present in current frame")
                if similarity_to_current_obj > highest_similarity:
                    highest_similarity = similarity_to_current_obj
                    obj_with_highest_similarity = obj_track

        if highest_similarity > SAMENESS_THRESHOLD:
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
            new_obj_track = ObjectTrack(get_next_id_for("object_id"), [new_obj_instance])
            self.objects.append(new_obj_track)
            return new_obj_track
