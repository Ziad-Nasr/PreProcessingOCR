import cv2
import pytesseract
import helpers
import matplotlib.pyplot as plt
import os
import numpy as np
import nltk

Skewing_Value = ["Size/35", "2(Size)/35", "3(Size)/35", "4(Size)/35", "5(Size)/35",
                 "6(Size)/35", "7(Size)/35", "8(Size)/35", "9(Size)/35", "10(Size)/35", "11(Size)/35",]


def OCR(image, language, singleImage=False):
    if (singleImage == True):
        OCRText = pytesseract.image_to_string(image, lang=language)
    else:
        OCRText = []
        for i in range(len(image)):
            OCRText.append(pytesseract.image_to_string(
                image[i], lang=language))
    return OCRText


def skewImageToOCR(images, labels, language, singlePointFlag=False):
    OCRResults = []
    accuracies = []
    # y-axis = Vertical Axis , x-axis = Horizontal Axis
    for image in images:
        Size = image.shape
        # y_axis = Size[0]
        # x_axis = Size[1]
        imageSkew = []
        maximumVerticalSkew = Size[0]/3
        maximumHorizontalSkew = Size[1]/3
        verticalSkew = Size[0]/35
        horizontalSkew = Size[1]/35
        while verticalSkew <= maximumVerticalSkew and horizontalSkew <= maximumHorizontalSkew:
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
            skewingKernel = cv2.getPerspectiveTransform(
                source, destination)
            skewedImage = cv2.warpPerspective(
                image, skewingKernel, (Size[1], Size[0]), borderValue=(255, 255, 255))
            # plt.imshow(skewedImage)
            # plt.show()
            imageSkew.append(OCR(skewedImage, language, singleImage=True))
            # skewingValue.append([horizontalSkew, verticalSkew])
            verticalSkew += Size[0]/35
            horizontalSkew += Size[1]/35
        OCRResults.append(imageSkew)
        # print(OCRResults)
        OCRResults_Adjusted = helpers.AdjustOCRResults(OCRResults)
        accuracies.append(calculate_accuracy(
            helpers.flatten(OCRResults_Adjusted), labels))
    print(accuracies)
    return accuracies


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
    # accuracies.append(accuracyPreSkew)
    # Skew the images from 2 points to the left
    accuracies = skewImageToOCR(
        images, labels, language, False)
    resultsPreSkew = OCR(images, language)
    accuracyPreSkew = calculate_accuracy(resultsPreSkew, labels)
    accuracies.insert(0, accuracyPreSkew)
    # accuracies = np.array(accuracies)
    # accuracies.flatten()
    print(accuracies)
    plt.bar(Skewing_Value, accuracies, color='blue')
    plt.xlabel("Skewing Value")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy for Skewing Images by {language} Language")
    plt.show()
    # Skew the images from 1 point
    # resultsSinglePointSkew = skewImage(
    #     images[2], language=language, singlePointFlag=True)
    # print(resultsSinglePointSkew[0][0])
    # Compare the results


def scalingImage(images, labels, language):
    # Initial OCR
    resultsPreScale = OCR(images, language)
    accuracies = [calculate_accuracy(resultsPreScale, labels)]
    # Scale the images
    scaler = [0]
    for scale in range(1, 20, 2):
        scaler.append(scale)
        scaled_images = []
        for image in images:
            size = image.shape
            scaled_image = cv2.resize(image, (scale*size[1], scale*size[0]))
            scaled_images.append(scaled_image)
        results = OCR(scaled_images, language)
        accuracies.append(calculate_accuracy(results, labels))
        print(calculate_accuracy(results, labels))
    plt.bar(scaler, accuracies, color='blue')
    plt.xlabel("Scaling Value")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy for Skewing Images by {language} Language")
    plt.show()


if __name__ == "__main__":
    englishImages, englishLabels = helpers.load_images_and_ground_truth(
        "Testing", "ground_truth/old_books_gt")

    # arabicImages, arabicLabels = helpers.load_images_and_ground_truth(
    #     "yarmouk/01_col", "ground_truth/yarmouk_gt")

    skewImage(englishImages, englishLabels[:2], "eng")
    # skewImage(arabicImages, arabicLabels, "ara")

    # scalingImage(englishImages, englishLabels[:2], "eng")
    # scalingImage(arabicImages, arabicLabels, "ara")
