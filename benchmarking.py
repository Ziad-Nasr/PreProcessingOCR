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


def skewImageLeft(images, language):
    # Images ba3dha Kol el skews ?
    # Walla Skew wa7ed le kol el images then iterate ?
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
            resultsSinglePointSkew = OCR(images, language)
    return resultsSinglePointSkew


def singlePointSkew(images, language):
    for image in images:
        results = []
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
            results.append(OCR(images, language))

    return results


def skewImage(images, labels, language):
    # Initial OCR
    # resultsPreSkew = OCR(images, language)
    # Skew the images from 2 points to the left
    resultsSkewImageLeft = skewImageLeft(images, language)
    # Skew the images from 1 point
    resultsSinglePointSkew = singlePointSkew(images, language)

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

    skewImage(englishImages, englishLabels, "ara")
    scalingImage(englishImages, englishLabels, "ara")
