from Mask_R_CNN_COCO import detect
from model.DetectedObjects import DetectedObjects
from model.ObjectInstance import create_objects
from mrcnn import visualize
from utils.image_utils import save_debug_image, get_frames_from_image_directory, show, get_frames
from utils.timer import print_timing_results
import cv2


video_file = "IMG_5823.mov"
video_path = "data/" + video_file
INPUT_IMAGE_DIMENSIONS = (640, 360)
output_video = cv2.VideoWriter("out/out_" + video_file, cv2.VideoWriter_fourcc("X", "V", "I", "D"), 10, INPUT_IMAGE_DIMENSIONS)

if __name__ == "__main__":

    detected_objects = DetectedObjects()

    # for frame_number, frame in enumerate(get_frames_from_image_directory("data/Interesting Kitti frame sets/Mostly static camera/0010", from_image=0, to_image=100)):
    for frame_number, frame in enumerate(get_frames(video_path, from_sec=1, to_sec=3)):
        result = detect(frame)

        newly_detected_objects = create_objects(result, frame)
        detected_objects.add_objects(newly_detected_objects)

        result_frame = visualize.draw_instances(frame, detected_objects)

        print(f"Frame {frame_number}: detected {len(newly_detected_objects)} objects. {len(detected_objects.objects)} total objects")
        show(result_frame, "Frame", await_keypress=False)
        save_debug_image(result_frame, "frame_" + str(frame_number))
        output_video.write(result_frame)

    print_timing_results()
