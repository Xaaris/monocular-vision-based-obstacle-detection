import cv2
import numpy as np

SIFT = cv2.xfeatures2d.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)


def get_matches(descriptor_a, descriptor_b, max_distance=100):
    matches = flann.knnMatch(descriptor_a, descriptor_b, k=1)
    matches_with_at_least_one_hit = list(filter(lambda m: len(m) > 0, matches))
    unpacked_matches = [m[0] for m in matches_with_at_least_one_hit]
    filtered_matches = list(filter(lambda m: m.distance <= max_distance, unpacked_matches))
    return filtered_matches


def average_descriptor_distance(descriptor_a, descriptor_b) -> float:
    matches = flann.knnMatch(descriptor_a, descriptor_b, k=1)
    matches_with_at_least_one_hit = list(filter(lambda m: len(m) > 0, matches))
    # sum distances
    total_distance = sum([m[0].distance for m in matches_with_at_least_one_hit])
    # divide by number of matches
    return (total_distance / len(matches)) / 10


def get_keypoints_and_descriptors_for_object(graysclae_image, mask):

    mask_int = np.copy(mask).astype(np.uint8)

    # Detect Sift features and compute descriptors.
    return SIFT.detectAndCompute(graysclae_image, mask_int)
