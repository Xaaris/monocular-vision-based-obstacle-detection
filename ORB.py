import cv2
import numpy as np

MAX_FEATURES = 200
orb = cv2.ORB_create(MAX_FEATURES)

MATCHER = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
CROSS_CHECK_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Makes sure matches match both ways: min(desca, descb)


def get_matches(descriptor_a, descriptor_b, max_distance):
    matches = CROSS_CHECK_MATCHER.match(descriptor_a, descriptor_b, None)
    filtered_matches = list(filter(lambda m: m.distance <= max_distance, matches))
    return filtered_matches


def average_descriptor_distance(descriptor_a, descriptor_b) -> float:
    matches = MATCHER.match(descriptor_a, descriptor_b, None)
    # sum distances
    total_distance = sum([d.distance for d in matches])
    # divide by number of matches
    return total_distance / len(matches)


def get_keypoints_and_descriptors_for_object(image, mask):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_int = np.copy(mask).astype(np.uint8)

    # Detect ORB features and compute descriptors.
    return orb.detectAndCompute(img_gray, mask_int)
