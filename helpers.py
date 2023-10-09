import cv2
import os
import matplotlib.pyplot as plt


# read images and ground truth data , return 2 lists : images , ground_truth
def load_images_and_ground_truth(image_folder, ground_truth_folder):
    image_files = os.listdir(image_folder)
    ground_truth_files = os.listdir(ground_truth_folder)

    images = []
    ground_truth = []

    for i in range(len(image_files)):
        # joining full paths
        image_path = os.path.join(image_folder, image_files[i])
        gt_path = os.path.join(ground_truth_folder, ground_truth_files[i])

        # read text files and append it to the ground_truth list
        with open(gt_path, "r", encoding="utf8") as file:
            gt_text = file.read()
            ground_truth.append(gt_text)

        # read images and append it to images list
        image = cv2.imread(image_path)
        images.append(image)

    return images, ground_truth


def AdjustOCRResults(OCRResults):
    OCRResults_Adjusted = []
    OCRResults_Adjusted = [[OCRResults[j][i]
                            for j in range(len(OCRResults))] for i in range(len(OCRResults[0]))]
    print(OCRResults_Adjusted)
    return OCRResults_Adjusted


def flatten(l):
    return [item for sublist in l for item in sublist]

# example usage
# data = "old_books"
# labels_path = "old_books_gt"
# images, labels = load_images_and_ground_truth(data, labels_path)

# display image using opencv (bad displaying in that case)
# im = cv2.resize(images[1], (960, 540))     # Resize image
# cv2.imshow("image", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# displaying image using matplotlib
# plt.imshow(images[1])
# plt.show()

# print(labels[1])
