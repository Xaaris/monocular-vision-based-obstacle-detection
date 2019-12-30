from Mask_R_CNN_COCO import detect, draw, get_class_name_for_id
from Model import ObjectInstance, Box, DetectedObjects
from ORB import get_keypoints_and_descriptors_for_object
from utils.image_utils import get_frames, save_debug_image
from utils.timer import print_timing_results, timing

VIDEO_FILE = "data/IMG_5823.mov"


@timing
def create_objects(result) -> [ObjectInstance]:
    objects = []
    number_of_results = result["class_ids"].shape[0]
    for i in range(number_of_results):
        class_name = get_class_name_for_id(result["class_ids"][i])

        roi = result["rois"][i]
        y1, x1, y2, x2 = roi
        width_in_pixel = x2 - x1
        height_in_pixel = y2 - y1
        center_x_in_pixel = x1 + width_in_pixel / 2
        center_y_in_pixel = y1 + height_in_pixel / 2
        frame_height_in_pixel, frame_width_in_pixel, _ = frame.shape
        center_x_normalized = center_x_in_pixel / frame_width_in_pixel
        center_y_normalized = center_y_in_pixel / frame_height_in_pixel
        width_normalized = width_in_pixel / frame_width_in_pixel
        height_normalized = height_in_pixel / frame_height_in_pixel
        normalized_roi_box = Box(center_x_normalized, center_y_normalized, width_normalized, height_normalized)

        confidence_score = result["scores"][i]

        mask = result["masks"][:, :, i]

        keypoints, descriptors = get_keypoints_and_descriptors_for_object(frame, mask)

        detected_object = ObjectInstance(class_name, normalized_roi_box, confidence_score, mask, keypoints, descriptors)
        objects.append(detected_object)
    return objects


if __name__ == "__main__":

    detected_objects = DetectedObjects()

    for frame_number, frame in enumerate(get_frames(VIDEO_FILE, from_sec=1, to_sec=2)):
        result = detect(frame)
        result_frame = draw(frame, result)
        save_debug_image(result_frame, "frame_" + str(frame_number))

        newly_detected_objects = create_objects(result)
        detected_objects.add_objects(newly_detected_objects)

        print(f"Frame {frame_number}: detected {len(newly_detected_objects)} objects. {len(detected_objects.objects)} total objects")

    print_timing_results()
