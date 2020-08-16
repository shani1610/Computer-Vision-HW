import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import PIL.Image
from PIL import Image
from matplotlib import pyplot as plt
import my_homography as mh

#Add functions here:
def q3_1():
    # create reference model
    im_path = './data/pf_floor.jpg'
    ref_image1 = create_ref(im_path)
    cv2.imwrite('./my_data/ref_image1_final.jpg', ref_image1)

def q3_2():
    # implant an image inside another image

    # example 1:
    im_path = './data/pf_floor.jpg'
    scene_path = './data/pf_desk.jpg'
    scene_im = im2im(scene_path, im_path)
    scene_im = cv2.cvtColor(scene_im, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./output/im2im1.jpg', scene_im)

    # example 2:
    im_path2 = './my_data/star_west.jpg'
    scene_path2 = './my_data/bigbangcomicbookshop.jpg'
    scene_im2 = im2im(scene_path2, im_path2)
    scene_im2 = cv2.cvtColor(scene_im2, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./output/im2im2.jpg', scene_im2)

    # example 3:
    im_path3 = './my_data/friends_episode.jpg'
    scene_path3 = './my_data/bill_gates.jpg'
    scene_im3 = im2im(scene_path3, im_path3)
    cv2.imwrite('./output/im2im3.jpg', scene_im3)
#Functions end

# HW functions:
def create_ref(im_path):
    image1 = cv2.imread(im_path)
    im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    N = 8  # number of corresponding points
    p1, _ = mh.getPoints(im1, im1, N)
    out_size = np.array([300, 150])
    p2 = np.array([[0, out_size[1], out_size[1], 0],
                   [0, 0, out_size[0], out_size[0]]])
    H2to1 = mh.computeH(p1, p2)
    H_trans, out_size, axis_arr = mh.Translation(im1, H2to1)
    wrap_im1 = mh.warpH(im1, H_trans, out_size)
    ref_image1 = wrap_im1[(-axis_arr[0]):(300-axis_arr[0]), (-axis_arr[2]):(150-axis_arr[2])]
    return ref_image1

def im2im(scene_path, im_path):
    ref_image1 = create_ref(im_path)
    scene_image = cv2.imread(scene_path)
    scene_im = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
    N = 8  # number of corresponding points
    _, p2 = mh.getPoints(scene_im, scene_im, N)
    out_size_ref = np.array([300, 150])
    p1 = np.array([[0, out_size_ref[1], out_size_ref[1], 0],
                   [0, 0, out_size_ref[0], out_size_ref[0]]])
    H2to1 = mh.computeH(p1, p2)
    H_trans, out_size, axis_arr = mh.Translation(ref_image1, H2to1)
    wrap_ref_im1 = mh.warpH(ref_image1, H_trans, out_size)
    wrap_idx = np.argwhere(wrap_ref_im1)
    scaled_wrap_idx = wrap_idx + np.array([axis_arr[0], axis_arr[2], 0])
    scene_im[scaled_wrap_idx[:,0], scaled_wrap_idx[:,1], scaled_wrap_idx[:,2]] = wrap_ref_im1[wrap_idx[:,0], wrap_idx[:,1], wrap_idx[:,2]]
    return scene_im

if __name__ == '__main__':
    q3_1()  # Create reference model
    q3_2()  # Implant an image inside another image

