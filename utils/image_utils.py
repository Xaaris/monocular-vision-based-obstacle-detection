"""Miscellaneous utility functions for working with images"""
import os

import cv2.cv2 as cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def letterbox_image(image, desired_size):
    """resize image with unchanged aspect ratio using padding (width, height)"""

    ih, iw = image.shape[:2]
    w, h = desired_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized_image = cv2.resize(image, (nw, nh))

    delta_w = w - nw
    delta_h = h - nh
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [128, 128, 128]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image


def take_center_square(image):
    """take the center square of the original image and returns it"""

    height, width = image.shape[:2]
    min_dimension = min(height, width)
    center = (width / 2, height / 2)

    square_center_image = cv2.getRectSubPix(image, (min_dimension, min_dimension), center)

    return square_center_image


def resize_image(image, desired_size):
    """resize image to desired size (width, height)"""

    resized_image = cv2.resize(image, desired_size)

    return resized_image


def get_image_patch_from_rect(image, rect):
    """Returns the specified area from the image"""
    top, left, bottom, right = rect
    size = (right - left, bottom - top)
    center = (left + size[0] / 2, top + size[1] / 2)
    return cv2.getRectSubPix(image, size, center)


def get_image_patch_from_contour(image, contour):
    """Returns the specified area from the image"""
    x, y, w, h = cv2.boundingRect(contour)
    size = (int(w * 1.2), int(h * 1.5))
    center = (x + w / 2, y + h / 2)
    return cv2.getRectSubPix(image, size, center)


def show(image, label="image"):
    """Displays the image (optionally with a label) and halts the program until any key is pressed"""
    cv2.imshow(label, image)
    cv2.waitKey()


def save_debug_image(image, filename, folder=None, resize_to=None):
    """
    Saves a given image under a specified filename.
    Optionally within a folder and resizing it (width, height)
    """
    if resize_to is not None:
        image = cv2.resize(image, resize_to)
    if folder:
        path = "debugImages/" + folder + "/" + filename + ".png"
    else:
        path = "debugImages/" + filename + ".png"
    cv2.imwrite(path, image)


def get_frames(path_to_video, from_sec=0, to_sec=None):
    """
    Generator that reads a video file from disk and yields a color correct frame at a time
    """
    fullpath = os.path.abspath(path_to_video)
    video = VideoFileClip(fullpath, audio=False).subclip(from_sec, to_sec)
    for frame in video.iter_frames():
        # We have to switch the order of channels as opencv has a different order as they are coming from the camera
        color_corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield color_corrected_frame


def draw_rectangle(image, box, color=(0, 0, 255), thickness=2, offset=(0, 0)):
    """
    Draws a rectangle on the provided image
    :param image: Image to draw on
    :param box: Box specifying the upper left and lower right corner of the rectangle
    :param color: Color of the rectangle
    :param thickness: Thickness of the border
    :param offset: Offset towards the upper left corner of the image
    """
    top, left, bottom, right = [int(x) for x in box]
    offset_y, offset_x = [int(x) for x in offset]

    left += offset_x
    right += offset_x
    top += offset_y
    bottom += offset_y

    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)


def draw_processed_image(frame):
    """Draws information found in the frame onto the image and returns it:
    - Box around found vehicles
    - Valid plates marked in green, invalid ones in red
    """
    image_copy = frame.image
    for vehicle in frame.vehicles:
        draw_rectangle(image_copy, vehicle.box)
        for plate in vehicle.plates:
            v_top, v_left, _, _ = vehicle.box
            if plate.valid:
                color = (0, 255, 0)
            else:
                color = (0, 0, 200)
            draw_rectangle(image_copy, plate.box, color=color, offset=(v_top, v_left))
    return image_copy
