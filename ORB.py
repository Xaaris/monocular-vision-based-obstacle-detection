import cv2
import numpy as np


#
#
# def simple():
#     img = cv2.imread('data/test.jpg', cv2.COLOR_BGR2BGRA)
#
#     # Initiate STAR detector
#     orb = cv2.ORB_create()
#
#     # find the keypoints with ORB
#     # kp = orb.detectAndCompute(img)
#
#     # compute the descriptors with ORB
#     kp, des = orb.detectAndCompute(img, None)
#
#     # draw only keypoints location,not size and orientation
#     img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
#     plt.imshow(img2)
#     plt.show()
#
#
# def show_keypoint_matches(im1, im2):
#     # Convert images to grayscale
#     im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#     im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
#
#     # Detect ORB features and compute descriptors.
#     orb = cv2.ORB_create(MAX_FEATURES)
#     keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)
#
#     # Match features.
#     matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#     matches = matcher.match(descriptors1, descriptors2, None)
#
#     # Sort matches by score
#     matches.sort(key=lambda x: x.distance, reverse=False)
#
#     # Remove not so good matches
#     num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
#     matches = matches[:num_good_matches]
#
#     # Draw top matches
#     im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
#     show(im_matches)
#
#
# def match_mask(im1, im2, mask1, mask2):
#     # Convert images to grayscale
#     im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#     im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
#
#     mask1_int = np.copy(mask1).astype(np.uint8)
#     mask2_int = np.copy(mask2).astype(np.uint8)
#
#     # Detect ORB features and compute descriptors.
#     orb = cv2.ORB_create(MAX_FEATURES)
#     keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, mask1_int)
#     keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, mask2_int)
#
#     # Match features.
#     if descriptors1 is not None and descriptors2 is not None:
#         matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#         matches = matcher.match(descriptors1, descriptors2, None)
#
#         # Sort matches by score
#         matches.sort(key=lambda x: x.distance, reverse=False)
#
#         # Remove not so good matches
#         num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
#         matches = matches[:num_good_matches]
#
#         # Draw top matches
#         im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
#         show(im_matches)
#
#
# def match_frames(frame1, frame2, masks1, masks2):
#     number_of_objects_detected = min(masks1.shape[2], masks2.shape[2])
#     for i in range(number_of_objects_detected):
#         match_mask(frame1, frame2, masks1[:, :, i], masks2[:, :, i])


def get_keypoints_and_descriptors_for_object(image, mask):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_int = np.copy(mask).astype(np.uint8)

    # Detect ORB features and compute descriptors.
    return orb.detectAndCompute(img_gray, mask_int)


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.75
VIDEO_FILE = "data/IMG_2594.mov"
orb = cv2.ORB_create(MAX_FEATURES)

# if __name__ == "__main__":

    # last_frame = None
    #
    # for frame_number, frame in enumerate(get_frames(VIDEO_FILE)):
    #
    #     if last_frame is not None:
    #         show_keypoint_matches(last_frame, frame)
    #
    #     last_frame = frame
