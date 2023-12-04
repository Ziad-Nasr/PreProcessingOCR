import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
import numpy as np
import nltk
import csv
import helpers
# from craft_text_detector import (
#     Craft, read_image, 
#     load_craftnet_model, 
#     load_refinenet_model, 
#     get_prediction, 
#     empty_cuda_cache,
# )

def acc(ocr_result, grt):
    ocr_words = nltk.word_tokenize(ocr_result.strip())
    gt_words = nltk.word_tokenize(grt.strip())

    # Calculate the Levenshtein distance (edit distance) between the recognized words and ground truth words
    distance = nltk.edit_distance(ocr_words, gt_words)

    # Calculate WER by normalizing the distance by the number of words in ground truth
    wer = distance / max(len(ocr_words), len(gt_words))

    # Calculate accuracy as 1 - WER (lower WER is better, so higher accuracy)
    return 1 - wer

def ocring(image,grt):
    ocr = pytesseract.image_to_boxes(image, lang="ara")

    # accuracy = acc(ocr, grt)
    # print(accuracy)
    return ocr

def psm6OCR(image, singleImage=False):
    if (singleImage == True):
        OCRText = pytesseract.image_to_string(image, lang="ara",config="--psm 6")
    else:
        OCRText = []
        for i in range(len(image)):
            OCRText.append(pytesseract.image_to_string(image[i], lang="ara", config="--psm 6"))
    return OCRText

def AbstractOCR(image):
    OCRText = []
    for i in range(len(image)):
        OCRText.append(pytesseract.image_to_string(image[i], lang="ara"))
    return OCRText

# Resizing the image to 70%(Random Value) of the original size
def thresholding(images):
    results=[]
    for singleImageIdx in range(len(images)):
        greyscale=cv2.cvtColor(images[singleImageIdx], cv2.COLOR_BGR2GRAY)
        results.append(cv2.threshold(greyscale, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+ cv2.THRESH_OTSU)[1])
    return results
def resize(images):
    # Define scale factor
    scale_factor = 0.7
    
    resized_images=[]
    for singleImage in range(len(images)):
        height, width = images[singleImage].shape[:2]

        # Calculate new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Resize image
        resized_images.append(cv2.resize(images[singleImage], (new_width, new_height),interpolation=cv2.INTER_LINEAR))
    
    return resized_images

def noise_removal(images):

    resized_images=[]
    for singleImage in range(len(images)):
        resized_images.append(cv2.fastNlMeansDenoisingColored(images[singleImage], None, 15, 15, 7, 15))
    return resized_images

# def extract_lines(image,path):
#     output_dir = 'oneLine/' + path
#     craft = Craft(output_dir=output_dir, crop_type="box", cuda=False)
#     craft.detect_text(image)
#     read_image(image)

def one_line_extraction(images, filenames):

    for singleImageIdx in range(len(images)):
        print(f"____________{singleImageIdx}______________")
        extract_lines(images[singleImageIdx], filenames[singleImageIdx])


def create_csv(filename, data):
    print("__________________________")
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        print("__________________________")

        # write header
        writer.writerow(["File Name", "Adaptive Thresholding"])
        for i in range(len(data[0])):
            writer.writerow([data[0][i], data[1][i]])
        file.close()

def noNoise(img):
    kernal = np.ones((1,1), np.uint8)
    image = cv2.dilate(img,kernal,iterations=1)
    kernal= np.ones((1,1), np.uint8)
    image = cv2.erode(img,kernal,iterations=1)
    image = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernal)
    image = cv2.medianBlur(image,1)
    return(image) 

def Skewing(images,degree):
    results=[]
    for singleImageIdx in range(len(images)):
        image_center = tuple(np.array(images[singleImageIdx].shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -degree, 1.0)
        results.append(cv2.warpAffine(images[singleImageIdx],
                                 rot_mat, images[singleImageIdx].shape[1::-1],
                                   flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)))
    return results

def SkewingAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def DeSkewing(images,degree):
    results = []
    for singleImageIdx in range(len(images)):
        angle = SkewingAngle(images[singleImageIdx])
        results.append(rotateImage(images[singleImageIdx], -1.0 * angle))
    return results


if __name__ == "__main__":
    # Loading the Dataset and ground truth from the full path
    # Arabian Data Frame, Arabian Ground Truth
    arabDF, arabGT , arabFileNames, gtFileNames = helpers.load_images_and_ground_truth(
        "arab\\hundred",
          "ground_truth\\arab_gt")
    
    filename = 'resizing.csv'
    arabOCR = thresholding(arabDF)
    
    print("__________________________")
    arabOCR = psm6OCR(arabOCR)
    print("__________________________")
    original = []
    for i in range(len(arabOCR)):
        original.append(acc(arabOCR[i],arabGT[i]))

    # arabOCR = psm6OCR(arabDF[85:100])

    # psm6 = []
    # for i in range(len(arabDF[85:100])):
    #     psm6.append(acc(arabOCR[i],arabGT[i+85]))

    # print("__________________________")
    # resized=noise_removal(arabDF[85:100])

    print("__________________________")
    # arabOCR = psm6OCR(resized)
    # resized = []
    # for i in range(len(arabOCR)):
    #     resized.append(acc(arabOCR[i],arabGT[i+85]))
    
    # # print("__________________________")
    # # arabOCR = one_line_OCR(arabDF[:5])
    
    create_csv(filename, [arabFileNames,original])

    # results = Skewing(arabDF[55:57], 5)
    # # for i in range(len(results)):
    # #     print("__________________________")
    # #     plt.imshow(results[i])
    # #     plt.show()

    # print("__________________________")
    # results = DeSkewing(results, 5)
    # for i in range(len(results)):
    #     print("__________________________")
    #     plt.imshow(results[i])
    #     plt.show()
    # Lines Extractions can not be perfect without some preprocessing
    # soraa = greyscale(arabDF[:5])
    # sora = one_line_extraction(arabDF[5:15],arabFileNames[5:15])
    # print("__________________________")
    # # for i in range(5):
    # print(arabFileNames[0])
    # sarabDF, sarabGT , sarabFileNames, sgtFileNames = helpers.load_images_and_ground_truth(
    #     f"oneLine\\{arabFileNames[0]}",
    #     "ground_truth\\arab_gt")
    # arabOCR = AbstractOCR(sarabDF)
    # original = []
    # print(arabOCR[0])
    # for i in range(len(sarabDF)):
    #     original.append(acc(arabOCR[i],sarabGT[i]))
    # thresh, im_bw = cv2.threshold(sora, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
    # extract_lines(sora)

## Next : Search how to extend the csv file to include extra columns
## 3ayz a3ml column gded w a3ml append lel csv file

