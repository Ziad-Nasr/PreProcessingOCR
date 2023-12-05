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

# To be Moved to cvOperation.py

def greyScale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def BB(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)[1]
    noN= noise_removal(thresh)
    noN=thin_font(noN)
    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    dilate = cv2.dilate(noN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 60)), iterations=3)
    # dilate = cv2.dilate(thresh, kernel, iterations=2)
    cv2.imwrite("box1.jpg", (dilate))
    return dilate

# To be Moved to cvOperation.py

def load_lined_images(imageFile,gtFile):
    gt_folder = "arab_gt"
    image_folder = os.path.join("oneLine",imageFile,"image_crops")
    # ground_truth_folder = os.path.join("arab_gt",gtFile)
    image_files = natsort.natsorted(os.listdir(image_folder))
    ground_truth_files = natsort.natsorted(os.listdir(gt_folder))
    images=[]
    ground_truth = []
    for i in range(len(image_files)):
        image_path = os.path.join(image_folder,image_files[i])
        image = cv2.imread(image_path)
        images.append(image)
    for i in range(len(ground_truth_files)):
        if(ground_truth_files[i] == "14766.txt"):
            print("Found")
            gt_path = os.path.join(gt_folder,ground_truth_files[i])
            with open(gt_path, "r", encoding="utf8") as file:
                    gt_text = file.read()
                    ground_truth.append(gt_text)
    return images,ground_truth

def line_to_word_image(image):
    # gray_Image=greyScale(BB(image))
    img_row_sum = np.sum(BB(image),axis=0).tolist()
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
    sub_image= image[:,0:list[0]]
    word_images=[]
    word_images.append(sub_image)
    # cv2.imwrite("bignady/Test_" + str(len(list)) + ".jpg", sub_image)
    for i in range(len(list)-1):
        sub_image= image[:,list[i]:list[i+1]]
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

def acc(ocr_result,grt):
    ocr_words = []
    for i in range(len(ocr_result)):
        ocr_words.append(tokenize_text(ocr_result[i].strip()))
    # for i in range(len(grt)):
    gt_words = tokenize_text(grt.strip())
    ocr_words = helpers.flatten(ocr_words)
    print(ocr_words)
    print("Break")
    ocr_words=' '.join(ocr_words)
    print(ocr_words)
    print("Break")
    print(gt_words)
    print("Break")
    gt_words=" ".join(gt_words)
    print(gt_words)
    # Calculate the Levenshtein distance (edit distance) between the recognized words and ground truth words
    distance = nltk.edit_distance(ocr_words, gt_words)
    # Calculate WER by normalizing the distance by the number of words in ground truth
    wer = distance / max(len(ocr_words), len(gt_words))
    # Calculate accuracy as 1 - WER (lower WER is better, so higher accuracy)
    return 1 - wer