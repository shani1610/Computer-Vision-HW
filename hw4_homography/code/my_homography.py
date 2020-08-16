from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

# Add imports if needed:
from scipy.interpolate import interp2d
from skimage import color
from tqdm import tqdm
from numpy.linalg import pinv
# end imports

# Add extra functions here:
def FindCorners(im1, H):
     # This function used is function Translation that is used in HW function wrapH
    left_top_original = np.array([0, 0, 1]).reshape(-1, 1)
    left_bottom_original = np.array([0, im1.shape[0], 1]).reshape(-1, 1)
    right_top_original = np.array([im1.shape[1], 0, 1]).reshape(-1, 1)
    right_bottom_original = np.array([im1.shape[1], im1.shape[0], 1]).reshape(-1, 1)

    left_top = pinv(H) @ left_top_original
    left_bottom = pinv(H) @ left_bottom_original
    right_top = pinv(H) @ right_top_original
    right_bottom = pinv(H) @ right_bottom_original

    # normalized
    left_top /= left_top[2, :]
    left_bottom /= left_bottom[2, :]
    right_top /= right_top[2, :]
    right_bottom /= right_bottom[2, :]

    axis_top_button = [left_top, left_bottom, right_top, right_bottom]

    return axis_top_button

def Translation(im1, H):
    # This function used is in HW function wrapH
    axis_top_button = FindCorners(im1, H)
    left_top = axis_top_button[0]
    left_bottom = axis_top_button[1]
    right_top = axis_top_button[2]
    right_bottom = axis_top_button[3]

    axis_y_top = int(min(right_top[1], left_top[1]))
    axis_y_bottom = int(max(right_bottom[1], left_bottom[1]))
    axis_x_left = int(min(left_top[0], left_bottom[0]))
    axis_x_right = int(max(right_top[0], right_bottom[0]))

    axis_arr = [axis_y_top, axis_y_bottom, axis_x_left, axis_x_right]
    out_size = (abs(axis_y_bottom - axis_y_top), abs((axis_x_right - axis_x_left)))
    trans_mat = np.array([[1, 0, axis_x_left], [0, 1, axis_y_top], [0, 0, 1]])
    H_trans = H @ trans_mat  # h2to1
    return H_trans, out_size, axis_arr

def getScaled(im2, warp_im1, axis_arr, warp_is_left):
    axis_y_top = axis_arr[0]
    axis_y_bottom = axis_arr[1]
    axis_x_left = axis_arr[2]
    axis_x_right = axis_arr[3]

    shape_y = max(axis_y_bottom, im2.shape[0]) - min(axis_y_top, 0)
    shape_x = max(axis_x_right, im2.shape[1]) - min(axis_x_left, 0)

    warp_im1_scaled = np.zeros((shape_y, shape_x, 3))
    im2_scaled = np.zeros(warp_im1_scaled.shape)
    im2_mask = np.where(im2 > 0)
    if warp_is_left:
        im2_scaled[im2_mask[0] - min(axis_y_top, 0), im2_mask[1] - axis_x_left, im2_mask[2]] = im2[im2_mask]
    else:
        im2_scaled[im2_mask[0] - min(axis_y_top, 0), im2_mask[1], im2_mask[2]] = im2[im2_mask]
    # plt.figure(2)
    # plt.imshow(im2_scaled)
    # plt.show()
    im1_warp_mask = np.where(warp_im1 > 0)
    if warp_is_left:
        warp_im1_scaled[im1_warp_mask] = warp_im1[im1_warp_mask]
    else:
        warp_im1_scaled[im1_warp_mask[0], im1_warp_mask[1] + axis_x_left, im1_warp_mask[2]] = warp_im1[im1_warp_mask]
    # plt.figure(3)
    # plt.imshow(warp_im1_scaled)
    # plt.show()
    return warp_im1_scaled, im2_scaled

def prepareToMerge(xLeft, xRight, yTop, yBootom, warp_im1, im2):
    warp_im1_big = np.zeros((max(yBootom, im2.shape[0]) - min(yTop, 0), max(xRight, im2.shape[1]) - min(xLeft, 0), 3),
                            dtype='uint8')
    im1_warp_maskIdx = np.where(warp_im1 > 0)
    warp_im1_big[im1_warp_maskIdx[0] + max(yTop, 0), im1_warp_maskIdx[1], im1_warp_maskIdx[2]] = warp_im1[
        im1_warp_maskIdx]
    im2_big = np.zeros(warp_im1_big.shape, dtype='uint8')
    im2_maskIdx = np.where(im2 > 0)
    im2_big[im2_maskIdx[0] + max(-yTop, 0), im2_maskIdx[1] + max(xLeft, 0), im2_maskIdx[2]] = im2[im2_maskIdx]
    return warp_im1_big, im2_big

def panoramaTwoImg(im1, im2, warp_is_left, getPointMethod, useRANSAC):
    if getPointMethod == 'Manual':
        p1, p2 = getPoints(im1, im2, N=8)
    else:  # getPointMethod=='SIFT':
        p1, p2 = getPoints_SIFT(im1, im2)
    if useRANSAC:
        nIter = 4000
        tol = 5
        H2to1 = ransacH(p1, p2, nIter, tol)
    else:
        H2to1 = computeH(p1, p2)
    H_trans, out_size, axis_arr = Translation(im1, H2to1)
    warp_im1 = warpH(im1, H_trans, out_size)
    warp_im1_scaled, im2_scaled = getScaled(im2, warp_im1, axis_arr, warp_is_left)
    panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
    return panoramaTest

def beachTest(getPointMethod, useRANSAC):
    downSampleRate = 1
    # images beach
    beach1 = cv2.imread('data/beach1.jpg')
    beach2 = cv2.imread('data/beach2.jpg')
    beach3 = cv2.imread('data/beach3.jpg')
    beach4 = cv2.imread('data/beach4.jpg')
    beach5 = cv2.imread('data/beach5.jpg')

    im_beach1 = cv2.cvtColor(beach1, cv2.COLOR_BGR2RGB)
    im_beach2 = cv2.cvtColor(beach2, cv2.COLOR_BGR2RGB)
    im_beach3 = cv2.cvtColor(beach3, cv2.COLOR_BGR2RGB)
    im_beach4 = cv2.cvtColor(beach4, cv2.COLOR_BGR2RGB)
    im_beach5 = cv2.cvtColor(beach5, cv2.COLOR_BGR2RGB)

    im_beach1 = cv2.rotate(im_beach1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im_beach2 = cv2.rotate(im_beach2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im_beach3 = cv2.rotate(im_beach3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im_beach4 = cv2.rotate(im_beach4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im_beach5 = cv2.rotate(im_beach5, cv2.ROTATE_90_COUNTERCLOCKWISE)

    im_beach1 = cv2.resize(im_beach1, (im_beach1.shape[0] // downSampleRate,
                                       im_beach1.shape[1] // downSampleRate))
    im_beach2 = cv2.resize(im_beach2, (im_beach2.shape[0] // downSampleRate,
                                       im_beach2.shape[1] // downSampleRate))
    im_beach3 = cv2.resize(im_beach3, (im_beach3.shape[0] // downSampleRate,
                                       im_beach3.shape[1] // downSampleRate))
    im_beach4 = cv2.resize(im_beach4, (im_beach4.shape[0] // downSampleRate,
                                       im_beach4.shape[1] // downSampleRate))
    im_beach5 = cv2.resize(im_beach5, (im_beach5.shape[0] // downSampleRate,
                                       im_beach5.shape[1] // downSampleRate))

    # 2+3
    panorama23 = panoramaTwoImg(im_beach2, im_beach3, False, getPointMethod, useRANSAC)
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/beach_panorama23_SIFT.jpg', panorama23)

    # 2+3+4
    panorama23 = cv2.imread('./my_data/beach_panorama23_SIFT.jpg')
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    panorama234 = panoramaTwoImg(im_beach4, panorama23, True, getPointMethod, useRANSAC)
    panorama234 = cv2.cvtColor(panorama234, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/beach_panorama234_SIFT.jpg', panorama234)

    # 1+2+3+4
    panorama234 = cv2.imread('./my_data/beach_panorama234_SIFT.jpg')
    panorama234 = cv2.cvtColor(panorama234, cv2.COLOR_BGR2RGB)
    panorama1234 = panoramaTwoImg(im_beach1, panorama234, False, getPointMethod, useRANSAC)
    panorama1234 = cv2.cvtColor(panorama1234, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/beach_panorama1234_SIFT.jpg', panorama1234)

    # 1+2+3+4+5
    panorama1234 = cv2.imread('./my_data/beach_panorama1234_SIFT.jpg')
    panorama1234 = cv2.cvtColor(panorama1234, cv2.COLOR_BGR2RGB)
    panorama_final_beach = panoramaTwoImg(im_beach5, panorama1234, True, getPointMethod, useRANSAC)
    panorama_final_beach = cv2.cvtColor(panorama_final_beach, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/beach_panorama_final_beach_SIFT.jpg', panorama_final_beach)

    return panorama_final_beach

def sintraTest(getPointMethod, useRANSAC):
    downSampleRate = 4

    sintra1 = cv2.imread('data/sintra1.JPG')
    sintra2 = cv2.imread('data/sintra2.JPG')
    sintra3 = cv2.imread('data/sintra3.JPG')
    sintra4 = cv2.imread('data/sintra4.JPG')
    sintra5 = cv2.imread('data/sintra5.JPG')

    im_sintra1 = cv2.cvtColor(sintra1, cv2.COLOR_BGR2RGB)
    im_sintra2 = cv2.cvtColor(sintra2, cv2.COLOR_BGR2RGB)
    im_sintra3 = cv2.cvtColor(sintra3, cv2.COLOR_BGR2RGB)
    im_sintra4 = cv2.cvtColor(sintra4, cv2.COLOR_BGR2RGB)
    im_sintra5 = cv2.cvtColor(sintra5, cv2.COLOR_BGR2RGB)

    im_sintra1 = cv2.resize(im_sintra1, (im_sintra1.shape[0] // downSampleRate,
                                         im_sintra1.shape[1] // downSampleRate))
    im_sintra2 = cv2.resize(im_sintra2, (im_sintra2.shape[0] // downSampleRate,
                                         im_sintra2.shape[1] // downSampleRate))
    im_sintra3 = cv2.resize(im_sintra3, (im_sintra3.shape[0] // downSampleRate,
                                         im_sintra3.shape[1] // downSampleRate))
    im_sintra4 = cv2.resize(im_sintra4, (im_sintra4.shape[0] // downSampleRate,
                                         im_sintra4.shape[1] // downSampleRate))
    im_sintra5 = cv2.resize(im_sintra5, (im_sintra5.shape[0] // downSampleRate,
                                         im_sintra5.shape[1] // downSampleRate))

    # 2+3
    panorama23 = panoramaTwoImg(im_sintra2, im_sintra3, False, getPointMethod, useRANSAC)
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/sintra_panorama23_SIFT.jpg', panorama23)

    # 2+3+4
    panorama23 = cv2.imread('./my_data/sintra_panorama23_SIFT.jpg')
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    panorama234 = panoramaTwoImg(im_sintra4, panorama23, True, getPointMethod, useRANSAC)
    panorama234 = cv2.cvtColor(panorama234, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/sintra_panorama234_SIFT.jpg', panorama234)

    # 1+2+3+4
    panorama234 = cv2.imread('./my_data/sintra_panorama234_SIFT.jpg')
    panorama234 = cv2.cvtColor(panorama234, cv2.COLOR_BGR2RGB)
    panorama1234 = panoramaTwoImg(im_sintra1, panorama234, False, getPointMethod, useRANSAC)
    panorama1234 = cv2.cvtColor(panorama1234, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/sintra_panorama1234_SIFT.jpg', panorama1234)

    # 1+2+3+4+5
    panorama1234 = cv2.imread('./my_data/sintra_panorama1234_SIFT.jpg')
    panorama1234 = cv2.cvtColor(panorama1234, cv2.COLOR_BGR2RGB)
    panorama_final_sintra = panoramaTwoImg(im_sintra5, panorama1234, True, getPointMethod, useRANSAC)
    panorama_final_sintra = cv2.cvtColor(panorama_final_sintra, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/sintra_panorama_final_sintra_SIFT.jpg', panorama_final_sintra)

    return panorama_final_sintra

def buildingTest(getPointMethod, useRANSAC):
    downSampleRate = 1

    building1 = cv2.imread('my_data/buliding1.jpeg')
    building2 = cv2.imread('my_data/buliding2.jpeg')
    building3 = cv2.imread('my_data/buliding3.jpeg')

    im_building1 = cv2.cvtColor(building1, cv2.COLOR_BGR2RGB)
    im_building2 = cv2.cvtColor(building2, cv2.COLOR_BGR2RGB)
    im_building3 = cv2.cvtColor(building3, cv2.COLOR_BGR2RGB)

    im_building1 = cv2.resize(im_building1, (im_building1.shape[0] // downSampleRate,
                                         im_building1.shape[1] // downSampleRate))
    im_building2 = cv2.resize(im_building2, (im_building2.shape[0] // downSampleRate,
                                         im_building2.shape[1] // downSampleRate))
    im_building3 = cv2.resize(im_building3, (im_building3.shape[0] // downSampleRate,
                                         im_building3.shape[1] // downSampleRate))

    # 2+3
    panorama23 = panoramaTwoImg(im_building3, im_building2, False, getPointMethod, useRANSAC)
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/building_panorama23_SIFT.jpg', panorama23)

    # 2+3+1
    panorama23 = cv2.imread('./my_data/building_panorama23_SIFT.jpg')
    panorama23 = cv2.cvtColor(panorama23, cv2.COLOR_BGR2RGB)
    panorama_final_building = panoramaTwoImg(im_building1, panorama23, True, getPointMethod, useRANSAC)
    panorama_final_building = cv2.cvtColor(panorama_final_building, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/building_panorama_final_SIFT.jpg', panorama_final_building)

    return panorama_final_building

def q2_1():
    # manual finding corresponding points

    # uploading images:
    image1 = cv2.imread('data/incline_L.png')
    image2 = cv2.imread('data/incline_R.png')
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # get points:
    N = 10  # number of corresponding points
    p1, p2 = getPoints(im1, im2, N)

def q2_2():
    # calculate transformation
    # uploading images:
    image1 = cv2.imread('data/incline_L.png')
    image2 = cv2.imread('data/incline_R.png')
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # First Test: projecting arbitrary points from im1 to im2
    N = 10
    p1, p2 = getPoints(im1, im2, N)
    H2to1 = computeH(p1, p2)
    p2_homo = np.concatenate((p2, np.ones([1, p2.shape[1]])), axis=0)
    p2_mu = H2to1 @ p2_homo
    p2_mu /= p2_mu[2, :]
    H = H2to1 / H2to1[2, 2]
    p2_projected = p2_mu[:2, :]

    # Second Test: homography of an image with itself
    N = 10
    p1, p2 = getPoints(im1, im1, N)
    H2to1 = computeH(p1, p2)

def q2_3():
    # image wraping

    # uploading images:
    image1 = cv2.imread('data/incline_L.png')
    image2 = cv2.imread('data/incline_R.png')
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # get points:
    N = 10  # number of corresponding points
    p1, p2 = getPoints(im1, im2, N)
    H2to1 = computeH(p1, p2)
    H_trans, out_size, axis_arr = Translation(im1, H2to1)
    wrap_im1 = warpH(im1, H_trans, out_size)
    wrap_im1 = cv2.cvtColor(wrap_im1, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/wrap_im1_manual.jpg', wrap_im1)

def q2_4():
    # Panorama stitching

    # uploading images:
    image1 = cv2.imread('data/incline_L.png')
    image2 = cv2.imread('data/incline_R.png')
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # get points:
    N = 10  # number of corresponding points
    p1, p2 = getPoints(im1, im2, N)
    H2to1 = computeH(p1, p2)
    H_trans, out_size, axis_arr = Translation(im1, H2to1)
    wrap_im1 = warpH(im1, H_trans, out_size)
    warp_is_left = True
    warp_im1_scaled, im2_scaled = getScaled(im2, wrap_im1, axis_arr, warp_is_left)
    panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
    panoramaTest = cv2.cvtColor(panoramaTest, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/panoramaTest_incline_manual.jpg', panoramaTest)

def q2_5():
    # autonomous panorama stitching using SIFT
    # uploading images:
    image1 = cv2.imread('data/incline_L.png')
    image2 = cv2.imread('data/incline_R.png')
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # get points:
    p1, p2 = getPoints_SIFT(im1, im2)
    H2to1 = computeH(p1, p2)
    H_trans, out_size, axis_arr = Translation(im1, H2to1)
    wrap_im1 = warpH(im1, H_trans, out_size)
    cv2.imwrite('./my_data/wrap_imTest_incline_SIFT.jpg', wrap_im1)
    warp_is_left = True
    warp_im1_scaled, im2_scaled = getScaled(im2, wrap_im1, axis_arr, warp_is_left)
    panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
    panoramaTest = cv2.cvtColor(panoramaTest, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./my_data/panoramaTest_incline_SIFT.jpg', panoramaTest)

def q2_7():
    # compare SIFT and Manual image selection
    # beach
    beachTest(getPointMethod='Manual', useRANSAC=False)
    beachTest(getPointMethod='SIFT', useRANSAC=False)
    # portugal
    sintraTest(getPointMethod='Manual', useRANSAC=False)
    sintraTest(getPointMethod='SIFT', useRANSAC=False)

def q2_8():
    beachTest(getPointMethod='SIFT', useRANSAC=True)
    beachTest(getPointMethod='Manual', useRANSAC=True)
    sintraTest(getPointMethod='Manual', useRANSAC=True)
    sintraTest(getPointMethod='SIFT', useRANSAC=True)

def q2_10():
    #Be Creative
    buildingTest(getPointMethod='SIFT', useRANSAC=True)

# Homework function:
def getPoints(im1, im2, N):
    fig = plt.figure(figsize=(9, 13))
    fig.add_subplot(1, 2, 1)
    plt.imshow(im1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(im2)
    x = plt.ginput(N + 1, show_clicks=True, mouse_add=1)
    p1: List[Any] = []
    p2: List[Any] = []
    for i in range(N):
        if i % 2 == 0:  # even = image 1 = left
            p1.append(x[i])
        else:  # odd = image 2 = right
            p2.append(x[i])
    p1 = np.array(p1).T
    p2 = np.array(p2).T
    return p1, p2

def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    A = []
    N = p1.shape[1]
    for i in range(N):
        u = p2[0, i]
        v = p2[1, i]
        x = p1[0, i]
        y = p1[1, i]
        A.append(np.array([-u, -v, -1, 0, 0, 0, u * x, v * x, x]))
        A.append(np.array([0, 0, 0, -u, -v, -1, u * y, v * y, y]))
    A = np.array(A)
    U, D, Vt = np.linalg.svd(A)
    H2to1 = np.reshape(Vt.T[:, -1], [3, 3])
    return H2to1

def warpH(im1, H, out_size):
    lab_image = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB)  # LAB
    warp_im1 = np.zeros((out_size[0], out_size[1], 3), dtype="uint8")
    x_range = np.arange(0, lab_image.shape[1])
    y_range = np.arange(0, lab_image.shape[0])
    zero_val = cv2.cvtColor(np.array([0, 0, 0], dtype="uint8").reshape(1, 1, 3), cv2.COLOR_RGB2LAB)
    f = {}

    for idx, ch in enumerate(["L", "A", "B"]):
        z_range = lab_image[:, :, idx]
        f[ch] = interp2d(x_range, y_range, z_range, copy="False", kind='linear')
    # H_inverse = np.linalg.inv(H)

    for x in tqdm(range(warp_im1.shape[1])):  # x
        for y in range(warp_im1.shape[0]):  # y
            p2 = np.array([x, y, 1]).reshape(-1, 1)  # indexs of wrap_im1
            p1 = H @ p2
            p1 = p1 / p1[2, 0]  # normalized the third index
            if p1[0] > 0 and p1[1] > 0 and p1[0] < im1.shape[1] and p1[1] < im1.shape[0]:
                for idx, ch in enumerate(["L", "A", "B"]):
                    warp_im1[y, x, idx] = int(round(f[ch](p1[0, 0], p1[1, 0])[0]))
                continue
            warp_im1[y, x, :] = zero_val

    warp_im1 = cv2.cvtColor(warp_im1.astype("uint8"), cv2.COLOR_LAB2RGB)
    return warp_im1

def imageStitching(img1, wrap_img2):
    panoImg = np.maximum(img1, wrap_img2)
    panoImg = np.uint8(panoImg)
    return panoImg

def getPoints_SIFT(im1, im2):
    gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, dest1 = sift.detectAndCompute(gray, None)

    gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, dest2 = sift.detectAndCompute(gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(dest1, dest2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append([m])

    p1 = []
    p2 = []
    for i in good_matches:
        p1.append(kp1[i[0].queryIdx].pt)
        p2.append(kp2[i[0].trainIdx].pt)

    p1 = np.asarray(p1).T
    p2 = np.asarray(p2).T
    return p1, p2

def ransacH(p1, p2, nIter, tol):
    best_score = 0
    bestH = np.zeros([3, 3])
    number_to_choose = 4
    matches_Number = p1.shape[1]
    for i in range(nIter):
        points_group = np.random.choice(matches_Number, number_to_choose, replace=False)
        H = computeH(p1[:, points_group], p2[:, points_group])
        p1_samp = H @ np.vstack((p2, np.ones([1, p2.shape[1]])))
        p1_samp /= p1_samp[2, :]
        p1_samp = p1_samp[:2, :]
        dist = np.linalg.norm(p1 - p1_samp, axis=0)
        current_score = np.sum(dist <= tol)
        if best_score < current_score:
            bestH = H
            best_score = current_score
    return bestH

if __name__ == '__main__':
    q2_1()  # Manual finding corresponding points
    q2_2()  # Calculate transformation
    q2_3()  # Image warping
    q2_4()  # Panorama stitching
    q2_5()  # Autonomous panorama stitching using SIFT
    q2_7()  # compare SIFT and Manual image selection
    q2_8()  # RANSAC
    q2_10() # Be Creative
