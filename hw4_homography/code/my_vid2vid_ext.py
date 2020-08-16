import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import PIL.Image
from PIL import Image
from matplotlib import pyplot as plt
import my_homography as mh
import os
import time
import imutils
import glob

def image_seq_to_video(imgs_path, output_path='./video.mp4', fps=15.0):
    output = output_path
    img_array = []
    for filename in glob.glob(os.path.join(imgs_path, '*.jpg')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        # img = cv2.resize(img, (width // 2, height // 2))
        img = cv2.resize(img, (width, height))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    print(size)
    print("writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, size)
    # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("saved video @ ", output)

def video_to_image_seq(vid_path, output_path='./datasets/OTB/img/Custom/'):
    os.makedirs(output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    print("converting video to frames...")
    while success:
        fname = str(count).zfill(4)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(output_path, fname + ".jpg"), image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print("total frames: ", count)

def neural_style_transfer(path, model):
    net = cv2.dnn.readNetFromTorch(model)
    # load the input image, resize it to have a width of 600 pixels, and
    # then grab the image dimensions
    image = cv2.imread(path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                 (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()
    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    return image, output

def artistic_video(video_path, output_path):
    model1 = './my_data/models/candy.t7 '
    model2 = './my_data/models/la_muse.t7 '
    model3 = './my_data/models/the_wave.t7 '
    model4 = './my_data/models/feathers.t7 '
    dir_path_org = './output/video_before/'
    dir_merged_img = './output/video_after/'
    os.makedirs(dir_merged_img, exist_ok=True)
    video_to_image_seq(video_path, dir_path_org)
    images = []
    count = 0
    for file1 in sorted(os.listdir(dir_path_org)):
        images.append(file1)
    model = model1
    for image1 in images:
        input_image_path1 = os.path.join(dir_path_org, image1)
        image, output = neural_style_transfer(input_image_path1, model)
        im_out = cv2.convertScaleAbs(output, alpha=(255.0))
        fname = str(count).zfill(4)
        count = count + 1
        new_img_path = os.path.join(dir_merged_img, fname + ".jpg")
        cv2.imwrite(new_img_path, im_out)
        if (count > 20):
            model = model2
        if (count > 45):
            model = model3
        if (count > 60):
            model = model4
    image_seq_to_video(dir_merged_img, output_path, fps=10.0)

if __name__ == '__main__':
    video_path = './my_data/ofek_is_fooling_around.mp4'
    output_path = './output/ofek_is_fooling_around_in_style.mp4'
    artistic_video(video_path, output_path)

# credits for the Johnson neural network:
# @inproceedings{Johnson2016Perceptual,
#   title={Perceptual losses for real-time style transfer and super-resolution},
#   author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
#   booktitle={European Conference on Computer Vision},
#   year={2016}
# }