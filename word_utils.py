import os
import natsort
import cv2
import Benchmark
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pytesseract
import nltk
import helpers
import CV_operations as cvop

def load_lined_images(root_folder,gt_folder):
    list_of_folders = natsort.natsorted(os.listdir(root_folder))

    all_images = []
    for i in range(len(list_of_folders)):
        image_folder = os.path.join(root_folder,list_of_folders[i],"image_crops")
        image_files = natsort.natsorted(os.listdir(image_folder))
        images=[]
        ground_truth = []
        for i in range(len(image_files)):
            image_path = os.path.join(image_folder,image_files[i])
            image = cv2.imread(image_path)
            images.append(image)
        all_images.append(images)
    # ground_truth_folder = os.path.join("arab_gt",gtFile)
    ground_truth_files = natsort.natsorted(os.listdir(gt_folder))
    print("____________Fetching_GT______________")
    for i in range(len(ground_truth_files)):
        gt_path = os.path.join(gt_folder,ground_truth_files[i])
        with open(gt_path, "r", encoding="utf8") as file:
                gt_text = file.read()
                ground_truth.append(gt_text)
    return all_images,ground_truth,list_of_folders,ground_truth_files

def line_to_word_image(image,PrePorcessedImage):
    img_row_sum = np.sum(cvop.BB(image),axis=0).tolist()
    # plt.plot(img_row_sum)
    # plt.show()
    img_row_sum1=np.convolve(img_row_sum,np.ones(1),mode='same')
    img_row_sum2=np.array(img_row_sum1)
    # print(img_row_sum2)
    list=[]
    super_threshold_indices = img_row_sum2 < 500
    img_row_sum2[super_threshold_indices] = 0
    super_threshold_indices = img_row_sum2 > max(img_row_sum2)/2
    img_row_sum2[super_threshold_indices] = max(img_row_sum2)
    for i in range(len(img_row_sum2)-1):
        # print(i,img_row_sum2[i],img_row_sum2[i+1])
        if (img_row_sum2[i]>img_row_sum2[i+1]):
            list.append(i)
    # plt.plot(img_row_sum2)
    # plt.show()
    if len(list)==0:
        return []
    sub_image= PrePorcessedImage[:,0:list[0]]
    word_images=[]
    word_images.append(sub_image)
    # cv2.imwrite("bignady/Test_" + str(len(list)) + ".jpg", sub_image)
    for i in range(len(list)-1):
        sub_image= PrePorcessedImage[:,list[i]:list[i+1]]
        word_images.insert(0,sub_image)
        # cv2.imwrite("bignady/Test_" + str(len(list)-1-i) + ".jpg", sub_image)
    return word_images

def tokenize_text(text):
    return text.split()

def OCRING(images):
    OCRResults = []
    for i in range(len(images)):
        OCRResults.append(pytesseract.image_to_string(images[i], lang='ara',config='--psm 8'))
    return OCRResults

def word_accuracy(ocr_result,grt):
    ocr_words = []
    for i in range(len(ocr_result)):
        ocr_words.append(tokenize_text(ocr_result[i].strip()))
    # for i in range(len(grt)):
    gt_words = tokenize_text(grt.strip())
    ocr_words = helpers.flatten(ocr_words)
    ocr_words=' '.join(ocr_words)
    gt_words=' '.join(gt_words)
    # Calculate the Levenshtein distance (edit distance) between the recognized words and ground truth words
    distance = nltk.edit_distance(ocr_words, gt_words)
    # Calculate WER by normalizing the distance by the number of words in ground truth
    wer = distance / max(len(ocr_words), len(gt_words))
    # Calculate accuracy as 1 - WER (lower WER is better, so higher accuracy)
    return 1 - wer

###########################################################
################### Vertical Histogram ####################
###########################################################

def resize(images):
    # Define scale factor
    scale_factor = 0.68
    
    resized_images=[]
    for singleImage in range(len(images)):
        height, width = images[singleImage].shape[:2]

        # Calculate new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Resize image
        resized_images.append(cv2.resize(images[singleImage], (new_width, new_height),interpolation=cv2.INTER_LINEAR))
    
    return resized_images

def vertical_histogram(img):
    img = 255-img
    gray_Image=cvop.greyScale(img)
    dilate = cv2.dilate(gray_Image, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3)), iterations=9)
    plt.imshow(dilate)
    plt.show()
    img_row_sum11 = np.sum(dilate,axis=1).tolist()
    # img_row_sum22=np.convolve(img_row_sum11,np.ones(20)/11,mode='same')
    img_row_sum33=np.array(img_row_sum11)
    # print(argrelextrema(img_row_sum33, np.greater))
    img44=argrelextrema(img_row_sum33, np.greater)
    print(img44[0].size)
    plt.plot(img_row_sum33)
    plt.show()