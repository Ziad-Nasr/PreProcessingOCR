import os
import natsort
import cv2

def load_images(imageFile):
    folder = os.path.join("oneLine",imageFile,"image_crops")
    image_files = natsort.natsorted(os.listdir(folder))
    images=[]
    for i in range(len(image_files)):
        image_path = os.path.join(folder,image_files[i])
        image = cv2.imread(image_path)
        images.append(image)
    return images

def line_to_word_image(image):
    pass