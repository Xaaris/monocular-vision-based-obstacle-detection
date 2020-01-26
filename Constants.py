import os
from enum import Enum


class InputDataType(Enum):
    IMAGE = 1
    VIDEO = 2


class MatcherType(Enum):
    SIFT = "Sift"
    ORB = "Orb"
    SURF = "Surf"

class CameraType(Enum):
    # Lens factor: image pixels per cm at 1m distance in real world
    IPHONE_8_PLUS_4K_60 = 32.8
    IPHONE_XR_4K_60 = 31.2


VIDEO_FILE = "IMG_5823"
VIDEO_FORMAT = ".mov"
IMAGE_DIRECTORY = "static/0010"
VIDEO_PATH = "data/video/"
IMAGE_SET_PATH = "data/imageSet/"
INPUT_DATA_TYPE = InputDataType.VIDEO
INPUT_DIMENSIONS = (640, 360)
FPS = 10
FROM_SEC_OR_IMAGE = 0
TO_SEC_OR_IMAGE = 2
MATCHER_TYPE = MatcherType.SIFT
VIDEO_SCALE = 1/6
CAMERA_TYPE = CameraType.IPHONE_XR_4K_60

FILE_PATH = (VIDEO_PATH + VIDEO_FILE + VIDEO_FORMAT) if INPUT_DATA_TYPE == InputDataType.VIDEO else (IMAGE_SET_PATH + IMAGE_DIRECTORY)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
