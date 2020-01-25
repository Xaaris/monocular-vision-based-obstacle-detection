import cv2
import numpy as np

HESSIAN_THRESHOLD = 400
SURF = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD, upright=True)

KNN_DESCRIPTOR_MATCHER = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)


def get_matches(descriptor_a, descriptor_b, max_distance=0.3):
    matches = _get_matches(descriptor_a, descriptor_b)
    filtered_matches = list(filter(lambda m: m.distance <= max_distance, matches))
    return filtered_matches


def average_descriptor_distance(descriptor_a, descriptor_b) -> float:
    # make sure that number of features in both test and query image is greater than or equal to number of nearest neighbors in knn match.
    if len(descriptor_a) < 2 or len(descriptor_b) < 2:
        return 100

    matches = _get_matches(descriptor_a, descriptor_b)

    # sum distances
    total_distance = sum([d.distance for d in matches])
    # divide by number of matches
    return total_distance / len(matches) if len(matches) > 0 else 100


def get_keypoints_and_descriptors_for_object(graysclae_image, mask):

    mask_int = np.copy(mask).astype(np.uint8)

    # Detect Surf features and compute descriptors.
    return SURF.detectAndCompute(graysclae_image, mask_int)


def _get_matches(descriptor_a, descriptor_b):
    knn_matches = KNN_DESCRIPTOR_MATCHER.knnMatch(descriptor_a, descriptor_b, 2)

    # D. Lowe's ration test (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

