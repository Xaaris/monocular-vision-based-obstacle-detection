from Mask_R_CNN_COCO import detect
from model.DetectedObjects import DetectedObjects
from model.ObjectInstance import create_objects
from mrcnn import visualize
from utils.image_utils import save_debug_image, show, prepare_video_output, get_frames
from utils.timer import print_timing_results
import asyncio


VIDEO_FILE = "IMG_5823"
VIDEO_FORMAT = ".mov"
IMAGE_DIRECTORY = "static/0010"
VIDEO_PATH = "data/video/"
IMAGE_SET_PATH = "data/imageSet/"
INPUT_DATA_TYPE = "image"  # video or image
INPUT_DIMENSIONS = (1242, 375)
FPS = 10
FROM_SEC_OR_IMAGE = 0
TO_SEC_OR_IMAGE = 12

file_path = (VIDEO_PATH + VIDEO_FILE + VIDEO_FORMAT) if INPUT_DATA_TYPE == "video" else (IMAGE_SET_PATH + IMAGE_DIRECTORY)

output_video = prepare_video_output(INPUT_DATA_TYPE, VIDEO_FORMAT, VIDEO_FILE, IMAGE_DIRECTORY, INPUT_DIMENSIONS, FPS, FROM_SEC_OR_IMAGE, TO_SEC_OR_IMAGE)

if __name__ == "__main__":

    detected_objects = DetectedObjects()

    for frame_number, frame in enumerate(get_frames(INPUT_DATA_TYPE, file_path, FROM_SEC_OR_IMAGE, TO_SEC_OR_IMAGE)):
        result = detect(frame)

        newly_detected_objects = create_objects(result, frame)
        detected_objects.add_objects(newly_detected_objects)

        result_frame = visualize.draw_instances(frame, detected_objects)

        print(f"Frame {frame_number}: detected {len(newly_detected_objects)} objects. {len(detected_objects.objects)} total objects")
        show(result_frame, "Frame", await_keypress=False)
        asyncio.run(save_debug_image(result_frame, "frame_" + str(frame_number)))
        output_video.write(result_frame)

    print_timing_results()
