import cv2

from utils.timer import timing

SIFT = cv2.xfeatures2d.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)


@timing
def get_matches(descriptor_a, descriptor_b, max_distance=100):
    matches = _get_matches(descriptor_a, descriptor_b)
    filtered_matches = list(filter(lambda m: m.distance <= max_distance, matches))
    return filtered_matches


@timing
def average_descriptor_distance(descriptor_a, descriptor_b) -> float:
    # make sure that number of features in both test and query image is greater than or equal to number of nearest neighbors in knn match.
    if len(descriptor_a) < 2 or len(descriptor_b) < 2:
        return 100

    matches = _get_matches(descriptor_a, descriptor_b)
    # sum distances
    total_distance = sum([m.distance for m in matches])
    # divide by number of matches
    return (total_distance / len(matches)) / 10 if len(matches) > 0 else 100


@timing
def get_keypoints_and_descriptors_for_object(graysclae_image, mask):
    # Detect Sift features and compute descriptors.
    return SIFT.detectAndCompute(graysclae_image, mask)


def _get_matches(descriptor_a, descriptor_b):
    knn_matches = flann.knnMatch(descriptor_a, descriptor_b, 2)

    # D. Lowe's ration test (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches
