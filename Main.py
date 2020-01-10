from Mask_R_CNN_COCO import detect, get_class_name_for_id
from Model import ObjectInstance, Box, DetectedObjects
from ORB import get_keypoints_and_descriptors_for_object
from mrcnn import visualize
from utils.image_utils import get_frames, save_debug_image, show, get_frames_from_image_directory
from utils.timer import print_timing_results, timing
import cv2


@timing
def create_objects(result) -> [ObjectInstance]:
    objects = []
    number_of_results = result["class_ids"].shape[0]
    for i in range(number_of_results):
        class_name = get_class_name_for_id(result["class_ids"][i])

        roi = result["rois"][i]
        y1, x1, y2, x2 = roi
        normalized_roi_box = Box(x1, y1, x2, y2)

        confidence_score = result["scores"][i]

        mask = result["masks"][:, :, i]

        keypoints, descriptors = get_keypoints_and_descriptors_for_object(frame, mask)

        detected_object = ObjectInstance(class_name, normalized_roi_box, confidence_score, mask, keypoints, descriptors)
        objects.append(detected_object)
    return objects


video_file = "IMG_5823.mov"
video_path = "data/" + video_file
INPUT_IMAGE_DIMENSIONS = (1242, 375)
output_video = cv2.VideoWriter("out/out_" + video_file, cv2.VideoWriter_fourcc("X", "V", "I", "D"), 10, INPUT_IMAGE_DIMENSIONS)

if __name__ == "__main__":

    detected_objects = DetectedObjects()

    for frame_number, frame in enumerate(get_frames_from_image_directory("data/Interesting Kitti frame sets/Mostly static camera/0010", from_image=0, to_image=100)):
        result = detect(frame)

        newly_detected_objects = create_objects(result)
        detected_objects.add_objects(newly_detected_objects)

        result_frame = visualize.draw_instances(frame, detected_objects)

        print(f"Frame {frame_number}: detected {len(newly_detected_objects)} objects. {len(detected_objects.objects)} total objects")
        # show(result_frame, "Frame", await_keypress=False)
        save_debug_image(result_frame, "frame_" + str(frame_number))
        output_video.write(result_frame)

    print_timing_results()
