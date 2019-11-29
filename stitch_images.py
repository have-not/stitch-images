import numpy as np
import cv2

def warpTwoImages(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32)
    pts2 = np.array([[[0, 0], [0, h2], [w2, h2], [w2, 0]]], dtype=np.float32)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1[0,:,:], pts2_[0,:,:]), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)
    t = np.array([-xmin, -ymin], dtype=np.int)
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)  # translate
    result = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
    return result

def stitchTwoImages(img1, img2, good_match_rate, min_match):
    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # Compute matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # sort by good matches

    good = matches[:int(len(matches) * good_match_rate)]

    if len(good) > min_match:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Find homography
        h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
        
        img =  warpTwoImages(img1, img2, h)
        return img
    else:
        print("Fewer feature points!")    


if __name__=='__main__':
    good_match_rate = 0.1
    min_match = 10

    img1 = cv2.imread('kyoto01.jpg')  # Image 1
    img2 = cv2.imread('kyoto02.jpg')  # Image 2
    img3 = cv2.imread('kyoto03.jpg')  # Image 3

    img21 = stitchTwoImages(img2, img1, good_match_rate, min_match) 
    img23 = stitchTwoImages(img2, img3, good_match_rate, min_match)

    img123 = stitchTwoImages(img21, img23, good_match_rate, min_match)

    cv2.imshow('Matched Features', img123)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

