import asyncio

from Constants import *
from model.DetectedObjects import DetectedObjects
from model.ObjectInstance import create_objects
from mrcnn import visualize
from mrcnn.Mask_R_CNN_COCO import detect
from utils.image_utils import save_debug_image, show, prepare_video_output, get_frames
from utils.timer import print_timing_results

if __name__ == "__main__":

    output_video = prepare_video_output()
    detected_objects = DetectedObjects()

    for frame_number, frame in enumerate(get_frames(INPUT_DATA_TYPE, FILE_PATH, FROM_SEC_OR_IMAGE, TO_SEC_OR_IMAGE)):
        result = detect(frame)

        newly_detected_objects = create_objects(result, frame)
        detected_objects.add_objects(newly_detected_objects)

        result_frame = visualize.draw_instances(frame, detected_objects)

        print(f"Frame {frame_number}: detected {len(newly_detected_objects)} objects. {len(detected_objects.objects)} total objects")
        show(result_frame, "Frame", await_keypress=False)
        asyncio.run(save_debug_image(result_frame, "frame_" + str(frame_number)))
        output_video.write(result_frame)

    print_timing_results()
