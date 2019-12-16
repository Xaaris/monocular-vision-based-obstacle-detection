from Mask_R_CNN_COCO import detect, draw
from utils.image_utils import get_frames, save_debug_image
from utils.timer import print_timing_results

VIDEO_FILE = "data/IMG_2594.mov"

if __name__ == "__main__":

    for frame_number, frame in enumerate(get_frames(VIDEO_FILE, from_sec=1, to_sec=2)):
        result = detect(frame)
        result_frame = draw(frame, result)
        save_debug_image(result_frame, "frame_" + str(frame_number))

    print_timing_results()
