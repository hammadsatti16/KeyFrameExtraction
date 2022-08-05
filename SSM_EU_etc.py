#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import os


# In[2]:


get_ipython().system('nvidia-smi')


# In[45]:


path ="C:/Users/Darvis AN3/Downloads/Frames/"
photos=[]
for fold_name in os.listdir(path):
    for i,photo in enumerate(os.listdir(os.path.join(path,fold_name))):
        if photo.endswith(".jpg"):
            if photo not in photos:
                photos.append(os.path.join(path+fold_name+"/"+photo))
         


# In[46]:


len(photos)


# In[64]:


len(lis)


# In[63]:


len(scores)


# In[61]:


lis


# In[48]:


photos


# In[59]:


lis = []
scores = []


# In[60]:


for i in range(0,len(photos)):
                    x = photos[i].split("/")[-1]
                    imageA = cv2.imread(photos[i])
                    imageB = cv2.imread(photos[i+1])
                    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                    (score, diff) = compare_ssim(grayA, grayB, full=True)
                    diff = (diff * 255).astype("uint8")
                    print("SSIM: {}".format(score))
                    thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    imageA = cv2.resize(imageA, (960, 540))
                    imageB = cv2.resize(imageB, (960, 540))
                    diff = cv2.resize(diff, (960, 540))
                    thresh = cv2.resize(thresh, (960, 960))
                    edges = cv2.Canny(diff,100,200)
                    laplacian = cv2.Laplacian(diff,cv2.CV_64F)
                    sobelx = cv2.Sobel(grayA, ddepth=cv2.CV_32F, dx=1, dy=0)  # x
                    sobely =  cv2.Sobel(grayA, ddepth=cv2.CV_32F, dx=0, dy=1)  # y
                    sobelx = cv2.resize(sobelx, (960, 960))
                    sobely = cv2.resize(sobely, (960, 960))
                    #cv2.imshow("Original Image 1", imageA)
                    #cv2.imshow("Original Image 2", imageB)
                    #cv2.imshow("Diff", diff)
                    #cv2.imshow("Thresh", thresh)
                    #cv2.imshow("edges", edges)
                    #cv2.imshow("laplacian", laplacian)
                    #cv2.imshow("sobelx", sobelx)
                    #cv2.imshow("sobely", sobely)
                    #cv2.waitKey(0)
                    score = score * 100
                    if score < 94:
                        cv2.imwrite("C:/Users/Darvis AN3/Downloads/SSM_frames1/"+x,imageA)
                    if score >= 94:
                        lis.append(photos[i])
                    scores.append(score)


# In[5]:


import cv2
import imutils
imageA = cv2.imread("C:/Users/Darvis AN3/Downloads/Frames_1/1653817051.12069.jpg")
imageB = cv2.imread("C:/Users/Darvis AN3/Downloads/Frames_1/1653817049.0472517.jpg")
#frame = cv2.imread("C:/Users/Darvis AN3/Downloads/Frames_1/1653817049.0472517.jpg")
#frame = cv2.resize(frame, (960, 540))
#cv2.imshow('image',frame)
#cv2.waitKey(0)


# In[8]:


gray_1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#gray_1 = cv2.GaussianBlur(imageA, (21, 21), 0)
#cv2.imshow('image-1',gray)
#cv2.waitKey(0)


# In[9]:


gray_2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#gray_2 = cv2.GaussianBlur(imageB, (21, 21), 0)


# In[10]:


(score, diff) = compare_ssim(gray_1, gray_2, full=True)
print(score)


# In[ ]:


diff = (diff * 255).astype("uint8")
diff = cv2.resize(diff, (960, 540))
cv2.imshow("Diff", diff)
cv2.waitKey(0)


# In[11]:


frameDelta = cv2.absdiff(gray_1, gray_2)
frame = cv2.resize(frameDelta, (960, 540))
cv2.imshow('image-2',frame)
cv2.waitKey(0)


# In[41]:


thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
# dilate the thresholded image to fill in holes, then find contours
# on thresholded image
thresh = cv2.dilate(thresh, None, iterations=2)
#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
frame = cv2.resize(thresh, (960, 540))
cv2.imshow('image-3',frame)
cv2.waitKey(0)


# In[2]:


import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre

test_img = cv2.imread("C:/Users/Darvis AN3/Downloads/Frames_1/1653817049.0472517.jpg")

ssim_measures = {}
rmse_measures = {}
sre_measures = {}

scale_percent = 100 # percent of original img size
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)

data_dir = 'C:/Users/Darvis AN3/Downloads/Frames_1/'
   
for file in os.listdir(data_dir):
	img_path = os.path.join(data_dir, file)
	data_img = cv2.imread(img_path)
	resized_img = cv2.resize(data_img, dim, interpolation = cv2.INTER_AREA)
	ssim_measures[img_path]= ssim(test_img, resized_img)
	rmse_measures[img_path]= rmse(test_img, resized_img)
	sre_measures[img_path]= sre(test_img, resized_img)

def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
    	closest = max(dict.values())
    else:
    	closest = min(dict.values())
    		
    for key, value in dict.items():
    	print("The difference between ", key ," and the original image is : \n", value)
    	if (value == closest):
    	    result[key] = closest
    	    
    print("The closest value: ", closest)	    
    print("######################################################################")
    return result
    
ssim = calc_closest_val(ssim_measures, True)
rmse = calc_closest_val(rmse_measures, False)
sre = calc_closest_val(sre_measures, True)

print("The most similar according to SSIM: " , ssim)
print("The most similar according to RMSE: " , rmse)
print("The most similar according to SRE: " , sre)


# In[15]:


from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2


# In[27]:


vs = cv2.VideoCapture("C:/Users/Darvis AN3/Downloads/T systems/10_ Full/89-TSystems-10_Full_05.09.2022_Session1.mp4")
firstFrame = None
while True:
    frame = vs.read()
    text = "Unoccupied"
    if frame is None:
        break
    frame = imutils.resize(frame, width=500,height = 960)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"


# In[19]:


from torch import nn
from PIL import Image
from torchvision import transforms

image1 = Image.open("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906845.797826.jpg")
image2 = Image.open("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906849.0548751.jpg")
# Note: images have to be of equal size

# Transform the images to tensors then flatten them to 1D tensors
image1_tensor = transforms.ToTensor()(image1).reshape(1, -1).squeeze() 
image2_tensor = transforms.ToTensor()(image2).reshape(1, -1).squeeze()

cos = nn.CosineSimilarity(dim=-1) # dim=0 -> dimension where cosine similarity is computed
value = float(cos(image1_tensor, image2_tensor)) # a value of 1 indicates strong similarity


# In[22]:


print(value)


# In[30]:


import numpy as np
from scipy.spatial import distance
from PIL import Image
from SSIM_PIL import compare_ssim

image1 = Image.open("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906845.797826.jpg")
image2 = Image.open("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906846.700652.jpg")
# Note: images have to be of equal size

# we need to flatten the image to a 1D vector
value1 = np.linalg.norm(np.array(image1) - np.array(image2))
value = distance.euclidean(np.array(image1).flatten(), np.array(image2).flatten())
value2 = compare_ssim(image1, image2, GPU=True)


# In[32]:


print(value1)
print(value)
print(value2)


# In[42]:


import imagehash
hash1 = imagehash.average_hash(image1)
hash2 = imagehash.average_hash(image2)

# Calculate the hamming distance
value3 = hash1-hash2


# In[43]:


print(value3)


# In[44]:


errorL2 = cv2.norm( image1, image2, cv2.NORM_L2 )
similarity = 1 - errorL2 / ( height * width )
print('Similarity = ',similarity)


# In[48]:


import cv2 as cv
import numpy as np

base = cv.imread("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906845.797826.jpg")
test = cv.imread("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906846.700652.jpg")
test2 = cv.imread("C:/Users/Darvis AN3/Downloads/Frames/frames1/1653906847.5825076.jpg")

hsv_base = cv.cvtColor(base, cv.COLOR_BGR2HSV)
hsv_test = cv.cvtColor(test, cv.COLOR_BGR2HSV)
hsv_test2 = cv.cvtColor(test2, cv.COLOR_BGR2HSV)

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]

hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
hist_test = cv.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
hist_test2 = cv.calcHist([hsv_test2], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

compare_method = cv.HISTCMP_CORREL

base_base = cv.compareHist(hist_base, hist_base, compare_method)
base_test = cv.compareHist(hist_base, hist_test, compare_method)
base_test2 = cv.compareHist(hist_base, hist_test2, compare_method)
base_test3 = cv.compareHist(hist_test, hist_test2, compare_method)
print('base_base Similarity = ', base_base)
print('base_test Similarity = ', base_test)
print('base_test2 Similarity = ', base_test2)
print('base_test3 Similarity = ', base_test3)


# In[ ]:




