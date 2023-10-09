import cv2
import pytesseract
import helpers
import matplotlib.pyplot as plt
import os
import numpy as np
import nltk


def OCR(image, language):
    OCRText = pytesseract.image_to_string(image, lang=language)
    return OCRText


def skewImageToOCR(images, language, singlePointFlag=False):
    # Images ba3dha Kol el skews ?
    # Walla Skew wa7ed le kol el images then iterate ?
    OCRResults = []
    for image in images:
        imageSkew = []
        Size = image.shape
        maximumVerticalSkew = Size[0]/3
        maximumHorizontalSkew = Size[1]/3
        # y-axis = Vertical Axis , x-axis = Horizontal Axis
        verticalSkew = Size[0]/35
        horizontalSkew = Size[1]/35
        while verticalSkew <= maximumVerticalSkew and horizontalSkew <= maximumHorizontalSkew:
            # skewingValue = []
            source = np.float32(
                [[0, Size[0]], [Size[1], Size[0]], [Size[1], 0], [0, 0]])
            if (singlePointFlag == True):
                destination = np.float32(
                    [[0, Size[0]],
                     [Size[1], Size[0]], [Size[1], 0], [horizontalSkew, verticalSkew]])
            else:
                destination = np.float32(
                    [[horizontalSkew, Size[0]-verticalSkew],
                        [Size[1], Size[0]], [Size[1], 0], [horizontalSkew, verticalSkew]])
            skewingKernel = cv2.getPerspectiveTransform(source, destination)
            skewedImage = cv2.warpPerspective(
                image, skewingKernel, (Size[1], Size[0]
                                       ), borderValue=(255, 255, 255))
            if (verticalSkew == 300):
                plt.imshow(skewedImage)
                plt.show()
            imageSkew.append(OCR(skewedImage, language))
            # skewingValue.append([horizontalSkew, verticalSkew])
            verticalSkew += Size[0]/35
            horizontalSkew += Size[1]/35
        OCRResults.append(imageSkew)
    return OCRResults


def singlePointSkew(images, language):
    for image in images:
        OCRResults = []
        Size = image.shape
        maximumVerticalSkew = Size[0]/3
        maximumHorizontalSkew = Size[1]/3
        # y-axis = Vertical Axis , x-axis = Horizontal Axis
        verticalSkew = Size[0]/20
        horizontalSkew = Size[1]/20
        while verticalSkew <= maximumVerticalSkew and horizontalSkew <= maximumHorizontalSkew:
            imageSkew = []
            skewingValue = []
            source = np.float32(
                [[0, Size[0]], [Size[1], Size[0]], [Size[1], 0], [0, 0]])
            destination = np.float32(
                [[0, Size[0]],
                    [Size[1], Size[0]], [Size[1], 0], [horizontalSkew, verticalSkew]])
            skewingKernel = cv2.getPerspectiveTransform(source, destination)
            skewedImage = cv2.warpPerspective(
                image, skewingKernel, (Size[1], Size[0]
                                       ), borderValue=(255, 255, 255))
            plt.imshow(skewedImage)
            plt.show()
            imageSkew.append(OCR(skewedImage, language))
            # skewingValue.append([horizontalSkew, verticalSkew])
            verticalSkew += Size[0]/15
            horizontalSkew += Size[1]/15
        OCRResults.append(imageSkew)
    return OCRResults


def calculate_accuracy(ocr_results, ground_truth):
    accuracies = []

    for ocr_result, gt in zip(ocr_results, ground_truth):
        ocr_words = nltk.word_tokenize(ocr_result.strip())
        gt_words = nltk.word_tokenize(gt.strip())

        # Calculate the Levenshtein distance (edit distance) between the recognized words and ground truth words
        distance = nltk.edit_distance(ocr_words, gt_words)

        # Calculate WER by normalizing the distance by the number of words in ground truth
        wer = distance / max(len(ocr_words), len(gt_words))

        # Calculate accuracy as 1 - WER (lower WER is better, so higher accuracy)
        true_accuracy = 1 - wer
        accuracies.append(true_accuracy)

    total_images = len(ocr_results)
    correct_ocr = sum(accuracies)
    accuracy = (correct_ocr / total_images) * 100.0

    return accuracy


def skewImage(images, labels, language):
    # Initial OCR
    resultsPreSkew = OCR(images, language)
    accuracies = [calculate_accuracy(resultsPreSkew, labels)]
    # Skew the images from 2 points to the left
    resultsSkewImageLeft = skewImageToOCR(images, language, False)
    accuracies.append(calculate_accuracy(resultsSkewImageLeft, labels))
    
    # Skew the images from 1 point
    # resultsSinglePointSkew = skewImage(
    #     images[2], language=language, singlePointFlag=True)
    # print(resultsSinglePointSkew[0][0])
    # Compare the results


def scalingImage(images, labels, landuage):
    # Initial OCR
    # resultsPreScale = OCR(images, language)
    # Scale the images
    for scale in range(1, 20, 2):
        pass


if __name__ == "__main__":
    englishImages, englishLabels = helpers.load_images_and_ground_truth(
        "Testing", "ground_truth/old_books_gt")

    # arabicImages, arabicLabels = helpers.load_images_and_ground_truth(
    #     "yarmouk/01_col", "ground_truth/yarmouk_gt")

    skewImage(englishImages, englishLabels, "eng")
    # scalingImage(englishImages, englishLabels, "ara")
