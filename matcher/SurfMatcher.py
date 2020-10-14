import cv2

from utils.timer import timing

HESSIAN_THRESHOLD = 400
MIN_NUMBER_OF_MATCHES = 2
SURF = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD, upright=True)

KNN_DESCRIPTOR_MATCHER = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)


@timing
def get_matches(descriptor_a, descriptor_b, max_distance=0.3):
    matches = _get_matches(descriptor_a, descriptor_b)
    filtered_matches = list(filter(lambda m: m.distance <= max_distance, matches))
    return filtered_matches


@timing
def average_descriptor_distance(descriptor_a, descriptor_b) -> float:
    # make sure that number of features in both test and query image is greater than or equal to number of nearest neighbors in knn match.
    if len(descriptor_a) < MIN_NUMBER_OF_MATCHES or len(descriptor_b) < MIN_NUMBER_OF_MATCHES:
        return 100

    matches = _get_matches(descriptor_a, descriptor_b)
    avg_number_of_descriptors = (len(descriptor_a) + len(descriptor_b)) / 2
    percent_of_matches = len(matches) / avg_number_of_descriptors
    if percent_of_matches < 0.1:
        return 100  # Less than 10% matches -> No Similarity

    # sum distances
    total_distance = sum([m.distance for m in matches])
    # divide by number of matches
    avg_distance = total_distance / len(matches) if len(matches) > 0 else 100
    return avg_distance


@timing
def get_keypoints_and_descriptors_for_object(graysclae_image, mask):
    # Detect Surf features and compute descriptors.
    return SURF.detectAndCompute(graysclae_image, mask)


def _get_matches(descriptor_a, descriptor_b):
    knn_matches = KNN_DESCRIPTOR_MATCHER.knnMatch(descriptor_a, descriptor_b, MIN_NUMBER_OF_MATCHES)

    # D. Lowe's ration test (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches
