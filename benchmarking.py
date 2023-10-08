import cv2
import pytesseract
import helpers
import matplotlib.pyplot as plt
import os
import numpy as np


def OCR(images, language):
    results = []
    for image in images:
        OCRText = pytesseract.image_to_string(image, lang=language)
        results.append(OCRText)
    return results


def skewImageLeft(images):
    for image in images:
        Size = image.shape
        maximumVerticalSkew = Size[0]/3
        maximumHorizontalSkew = Size[1]/3
        # y-axis = Vertical Axis , x-axis = Horizontal Axis
        verticalSkew = Size[0]/15
        horizontalSkew = Size[0]/15
        while verticalSkew < maximumVerticalSkew and horizontalSkew < maximumHorizontalSkew:
            source = np.float32(
                [[0, Size[0]], [Size[1], Size[0]], [Size[1], 0], [0, 0]])
            destination = np.float32(
                [[horizontalSkew, Size[0]-verticalSkew],
                 [Size[1], Size[0]], [Size[1], 0], [horizontalSkew, verticalSkew]])
            skewingKernel = cv2.getPerspectiveTransform(source, destination)
            skewedImage = cv2.warpPerspective(
                image, skewingKernel, (Size[1], Size[0]))
            plt.imshow(skewedImage)
            plt.show()
            verticalSkew += 75
            horizontalSkew += 75
            # resultsPostSkewLeft = OCR(images, language)
    # return resultsPostSkewLeft


def singlePointSkew(images):
    for image in images:
        Size = image.shape
        maximumVerticalSkew = Size[0]/3
        maximumHorizontalSkew = Size[1]/3
        # y-axis = Vertical Axis , x-axis = Horizontal Axis
        verticalSkew = Size[0]/15
        horizontalSkew = Size[0]/15
        while verticalSkew < maximumVerticalSkew and horizontalSkew < maximumHorizontalSkew:
            source = np.float32(
                [[0, Size[0]], [Size[1], Size[0]], [Size[1], 0], [0, 0]])
            destination = np.float32(
                [[0, Size[0]],
                 [Size[1], Size[0]], [Size[1], 0], [horizontalSkew, verticalSkew]])
            skewingKernel = cv2.getPerspectiveTransform(source, destination)
            skewedImage = cv2.warpPerspective(
                image, skewingKernel, (Size[1], Size[0]))
            plt.imshow(skewedImage)
            plt.show()
            verticalSkew += 75
            horizontalSkew += 75
            # resultsPostSkewLeft = OCR(images, language)
    # return resultsPostSkewLeft


def skewImage(images, labels, language):
    # Initial OCR
    # resultsPreSkew = OCR(images, language)
    # Skew the images from 2 points to the left
    resultsSkewImageLeft = skewImageLeft(images)
    # resultsPostSkewLeft = OCR(images, language)
    # Skew the images from 1 point
    resultsSinglePointSkew = singlePointSkew(images)

    # Compare the results


if __name__ == "__main__":
    englishImages, englishLabels = helpers.load_images_and_ground_truth(
        "Testing", "ground_truth/old_books_gt")

    # arabicImages, arabicLabels = helpers.load_images_and_ground_truth(
    #     "yarmouk/01_col", "ground_truth/yarmouk_gt")

    skewImage(englishImages, englishLabels, "ara")
