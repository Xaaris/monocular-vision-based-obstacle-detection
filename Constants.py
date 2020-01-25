from enum import Enum


class InputDataType(Enum):
    IMAGE = 1
    VIDEO = 2


class MatcherType(Enum):
    SIFT = "Sift"
    ORB = "Orb"


VIDEO_FILE = "IMG_5823"
VIDEO_FORMAT = ".mov"
IMAGE_DIRECTORY = "static/0010"
VIDEO_PATH = "data/video/"
IMAGE_SET_PATH = "data/imageSet/"
INPUT_DATA_TYPE = InputDataType.IMAGE
INPUT_DIMENSIONS = (1242, 375)
FPS = 10
FROM_SEC_OR_IMAGE = 0
TO_SEC_OR_IMAGE = 12
MATCHER_TYPE = MatcherType.ORB

FILE_PATH = (VIDEO_PATH + VIDEO_FILE + VIDEO_FORMAT) if INPUT_DATA_TYPE == InputDataType.VIDEO else (IMAGE_SET_PATH + IMAGE_DIRECTORY)

