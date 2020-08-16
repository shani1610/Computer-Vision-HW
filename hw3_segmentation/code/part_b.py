# -*- coding: utf-8 -*-
"""Part_B.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17ERxXge9HLdbwosWLTFV3oYPcaZ-rRCX

**Part 2 - Jurrasic Fishbach**

In this part you are going to apply segmentation on a video, and integrate with other elements.
"""

#0 imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import signal
import time
import os

#pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# import datasets in torchvision
import torchvision.datasets as datasets
# import model zoo in torchvision
import torchvision.models as models
from torchvision import utils

"""1. Film a short video of yourself (you can use your phone/webcam for that), but without too much camera movement. You on the other hand, can
move -- walk/move you hands... (we expect you to). Convert the video to frames and resize the images for a reasonable not too high resolution
(lower than 720p ~ 1280x720 pixles). You can use the function in frame_video_convert.py to help you. Display 2 frames in the report.
"""

#1
#helper function provided by the course staff
#Author: Tal Daniel

import cv2
import numpy as np
import glob
import os


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


def video_to_image_seq(vid_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    print("converting video to frames...")
    while success:
        fname = str(count).zfill(4)
        cv2.imwrite(os.path.join(output_path, fname + ".jpg"), image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print("total frames: ", count)

output_path='./datasets/OTB/img/Custom/shani_gym'
video_to_image_seq("./my_data/shani_in_gym.mp4",output_path)


items_shani = os.listdir('./datasets/OTB/img/Custom/shani_gym')
i = 0
for  each_image in items_shani:
  if(i<4):
    if each_image.endswith(".jpg"):
      full_path = "./datasets/OTB/img/Custom/shani_gym/" + each_image
      im = cv2.imread(full_path)
      img_rotate= cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
      image = cv2.cvtColor(img_rotate,cv2.COLOR_BGR2RGB)  
      plt.figure()
      plt.imshow(image)
      plt.show()
      plt.title(each_image)
      plt.axis(False)
      plt.grid(False)
      i = i + 1
  else:
    break

"""2. Segment yourself out of the video (frame-by-frame) using one of the methods (classic or deep). Display 2 frames in the report."""

#2

# download and load the pre-trained model
model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
# put in inference mode
model.eval();
# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# deep learning-based method:
def deeplabv3_segmentation(filename, label_idx, output_path, model ,count, is_rotate):
    # load an image
    input_image = Image.open(filename)
    # define the pre-processing steps
    # image->tensor, normalization
    preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # perform pre-processing
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch of size 1 as expected by the model
    # send to device
    model = model.to(device)
    input_batch = input_batch.to(device)
    # forward pass
    with torch.no_grad():
      output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    print("output shape: ", output.shape)
    print("output_predictions shape: ", output_predictions.shape)
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    # plot
    #fig = plt.figure(figsize=(15,15))
    #ax = fig.add_subplot(111)
    #
    #ax.imshow(r)
    #ax.set_axis_off()
    labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable','dog', 'horse', 'motorbike', 'person',
    'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
    #print(["{}: {}".format(i + 1, labels[i]) for i in range(len(labels))])
    # what labels were recognized?
    np.unique(output_predictions.cpu().numpy())
    # create a mask
    mask = torch.zeros_like(output_predictions).float().to(device)
    mask[output_predictions == label_idx] = 1 
    masked_img = input_image * mask.unsqueeze(2).byte().cpu().numpy()
    if(is_rotate == 1):
      masked_img = cv2.rotate(masked_img, cv2.ROTATE_90_CLOCKWISE)
    masked_img_rotate_90 = cv2.cvtColor(masked_img,cv2.COLOR_BGR2RGB)
    fname = str(count).zfill(4)
    cv2.imwrite(os.path.join(output_path, fname + ".jpg" ), masked_img_rotate_90)
    mask[output_predictions == label_idx] = 255
    mask_squeez =  mask.unsqueeze(2).byte().cpu().numpy()
    if(is_rotate == 1):
      mask_squeez = cv2.rotate(mask_squeez, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_path, fname + "_01" + ".jpg" ),mask_squeez)
    #fig = plt.figure(figsize=(15,15))
    #ax2 = fig.add_subplot(111)
    #ax2.imshow(masked_img_rotate_90)
    #ax2.set_axis_off()

output_path_after_segmention_shani_gym = './output_path_after_segmention/shani_gym'
os.makedirs(output_path_after_segmention_shani_gym , exist_ok=True)

count = 0
for each_image in items_shani:
  full_path = "./datasets/OTB/img/Custom/shani_gym/" + each_image
  is_rotate = 1
  deeplabv3_segmentation(full_path, 15, output_path_after_segmention_shani_gym , model, count, is_rotate)
  count =  count + 1

items_shani_mask = os.listdir('./output_path_after_segmention/shani_gym')
full_path_shani = "./output_path_after_segmention/shani_gym/0000.jpg"
input_image_shani = Image.open(full_path_shani)
im_shani = cv2.imread(full_path_shani)
image_shani = cv2.cvtColor(im_shani,cv2.COLOR_BGR2RGB)  
plt.figure()
plt.imshow(image_shani)
plt.show()
plt.title('0000')
plt.axis(False)
plt.grid(False)

full_path_shani2 = "./output_path_after_segmention/shani_gym/0021.jpg"
input_image_shani2 = Image.open(full_path_shani2)
im_shani2 = cv2.imread(full_path_shani2)
image_shani2 = cv2.cvtColor(im_shani2,cv2.COLOR_BGR2RGB)  
plt.figure()
plt.imshow(image_shani2)
plt.show()
plt.title('0021')
plt.axis(False)
plt.grid(False)

"""3. Pick one of the objects in the supplied videos ( ./data/dancing_man_model.mp4 , ./data/dinosaur_model.mp4 ,
./data/jet_model.mp4 ), convert it to images and segement it out using one of the methods (classic or deep). Display 2 frames in the report.
You can choose another object from: https://pixabay.com/videos/search/green%20screen/
(https://pixabay.com/videos/search/green%20screen/).
Explain how you performed the sementation for this specific type of video (i.e., green-screen videos). Did you use a simple/classic
method? Deep method? Combined both?
"""

#3
output_path='./datasets/OTB/img/Custom/dancing_man'
video_to_image_seq("./data/dancing_man_model.mp4", output_path)


items_dancing_man = os.listdir('./datasets/OTB/img/Custom/dancing_man')
i = 0
for  each_image in items_dancing_man:
  if(i<4):
    if each_image.endswith(".jpg"):
      full_path = "./datasets/OTB/img/Custom/dancing_man/" + each_image
      im = cv2.imread(full_path)
      image = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  
      plt.figure()
      plt.imshow(image)
      plt.show()
      plt.title(each_image)
      plt.axis(False)
      plt.grid(False)
      i = i + 1
  else:
    break

output_path_after_segmention_dancing_man = './output_path_after_segmention/dancing_man'
os.makedirs(output_path_after_segmention_dancing_man , exist_ok=True)


count = 0
for each_image in items_dancing_man:
  full_path = "./datasets/OTB/img/Custom/dancing_man/" + each_image
  is_rotate = 0
  deeplabv3_segmentation(full_path, 15, output_path_after_segmention_dancing_man , model, count, is_rotate)
  count =  count + 1

#display 

full_path_man = "./output_path_after_segmention/dancing_man/0000.jpg"
input_image_man = Image.open(full_path_man)
im_man = cv2.imread(full_path_man)
image_man = cv2.cvtColor(im_man,cv2.COLOR_BGR2RGB)  
plt.figure()
plt.imshow(image_man)
plt.show()
plt.title('0000')
plt.axis(False)
plt.grid(False)

full_path_man2 = "./output_path_after_segmention/dancing_man/0021.jpg"
input_image_man2 = Image.open(full_path_man2)
im_man2 = cv2.imread(full_path_man2)
image_man2 = cv2.cvtColor(im_man2,cv2.COLOR_BGR2RGB)  
plt.figure()
plt.imshow(image_man2)
plt.show()
plt.title('0021')
plt.axis(False)
plt.grid(False)

"""4. Put it all together - pick a background (can be a video or static image), put yourself (the segmented-self) and the segemented object on the
background. Stitch it frame-by-frame (don't make the video too long or it will take a lot of time, 10secs maximum). Display 2 frames of the
result. Convert the frames back to video. You can use the function in frame_video_convert.py to help you.
Tip: To make it look good, you can resize the images, create a mapping from pixel locations in the original image to pixels locations in the
new image.
You should submit the final video in the output folder, and upload it to YouTube as instructed above.
We expect some creative results, this can benefit you a lot when you want to demonstrate your Computer Vision abilities
"""

#4 

temp_output_path = './output/'
os.makedirs(temp_output_path , exist_ok=True)


num_frame = 200
num = 0;
num_save = 0
for i in range (num_frame):
  if (num > 92):
    num = 0 
  if (num < 10): 
    im1_shani_path = "./output_path_after_segmention/shani_gym/000" + str(num) + ".jpg"
    im1_shani_mask_path = "./output_path_after_segmention/shani_gym/000" + str(num) + "_01.jpg"
  else : 
    im1_shani_path = "./output_path_after_segmention/shani_gym/00" + str(num) + ".jpg"
    im1_shani_mask_path = "./output_path_after_segmention/shani_gym/00" + str(num) + "_01.jpg"

  if (i < 10):
    im2_man_path = "./output_path_after_segmention/dancing_man/000" + str(i) + ".jpg"  
    im2_man_mask_path = "./output_path_after_segmention/dancing_man/000" + str(i) + "_01.jpg"
  elif ((i < 100) and (i >= 10)):
    im2_man_path = "./output_path_after_segmention/dancing_man/00" + str(i) + ".jpg"  
    im2_man_mask_path = "./output_path_after_segmention/dancing_man/00" + str(i) + "_01.jpg"
  else : 
    im2_man_path = "./output_path_after_segmention/dancing_man/0" + str(i) + ".jpg"  
    im2_man_mask_path = "./output_path_after_segmention/dancing_man/0" + str(i) + "_01.jpg"


  num = num + 1
  # open images
  im1_shani = Image.open(im1_shani_path)
  im1_shani_mask = Image.open(im1_shani_mask_path)
  im2_man = Image.open(im2_man_path)
  im2_man_mask = Image.open(im2_man_mask_path)
  back_im = Image.open('./my_data/gym.jpg')


  #resize
  width, height = im2_man.size 
  newsize = (int(width/2),int(height/2))
  im2_man = im2_man.resize(newsize) 
  im2_man_mask = im2_man_mask.resize(newsize) 

  back_im.paste(im1_shani, (0,-100),im1_shani_mask)
  back_im.paste(im2_man, (320,120),im2_man_mask)

  num_save = 3*i
  if(num_save < 10):
    out_path = "./output/000" + str(num_save) + ".jpg"
    back_im.save(out_path, quality=95)
    out_path = "./output/000" + str(num_save +1) + ".jpg"
    back_im.save(out_path, quality=95)
    out_path = "./output/000" + str(num_save +2) + ".jpg"
    back_im.save(out_path, quality=95)
  if(num_save >= 10 and num_save<100 ):
    out_path = "./output/00" + str(num_save) + ".jpg"
    back_im.save(out_path, quality=95)
    out_path = "./output/00" + str(num_save +1) + ".jpg"
    back_im.save(out_path, quality=95)
    out_path = "./output/00" + str(num_save +2) + ".jpg"
    back_im.save(out_path, quality=95)
  if(num_save >= 100):
    out_path = "./output/0" + str(num_save) + ".jpg"
    back_im.save(out_path, quality=95)
    out_path = "./output/0" + str(num_save +1) + ".jpg"
    back_im.save(out_path, quality=95)
    out_path = "./output/0" + str(num_save +2) + ".jpg"
    back_im.save(out_path, quality=95)

items_back = os.listdir('./output')
i = 0
for  each_image in items_back:
  if(i<2):
    if each_image.endswith(".jpg"):
      full_path = "./output/" + each_image
      im = cv2.imread(full_path)
      image = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  
      plt.figure()
      plt.imshow(image)
      plt.show()
      plt.title(each_image)
      plt.axis(False)
      plt.grid(False)
      i = i + 1
  else:
    break

imgs_path = "./output"
image_seq_to_video(imgs_path, output_path='./shani_in_the_gym.mp4', fps=5)

"""end of part 2"""