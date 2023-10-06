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


def skewImage(images, labels, language):
    # Initial OCR
    resultsPreSkew = OCR(images, language)
    # Skew the images
    # for image in images:
    #   Shwayt Testing keda 3al mashy
    print(images[265].shape)
    plt.imshow(images[265])
    plt.show()
    # OCR after skewing
    resultsPostSkew = OCR(images, language)


if __name__ == "__main__":
    englishImages, englishLabels = helpers.load_images_and_ground_truth(
        "old_books/02_bin", "ground_truth/old_books_gt")

    # arabicImages, arabicLabels = helpers.load_images_and_ground_truth(
    #     "yarmouk/01_col", "ground_truth/yarmouk_gt")

    skewImage(englishImages, englishLabels, "ara")
