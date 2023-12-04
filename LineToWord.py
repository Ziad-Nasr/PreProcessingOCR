import os
import natsort
import cv2
import Benchmark
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

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
    # gray_Image=greyScale(BB(image))
    img_row_sum = np.sum(BB(image),axis=0).tolist()
    plt.plot(img_row_sum)
    plt.show()
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
    print((list))
    print(len(list))
    # print(argrelextrema(img_row_sum2, np.less)[0].size)
    # print(argrelextrema(img_row_sum2, np.less)[0])
    plt.plot(img_row_sum2)
    plt.show()
    sub_image= image[:,0:list[0]]
    cv2.imwrite("bignady/Test_" + str(len(list)) + ".jpg", sub_image)
    for i in range(len(list)-1):
        sub_image= image[:,list[i]:list[i+1]]
        cv2.imwrite("bignady/Test_" + str(len(list)-1-i) + ".jpg", sub_image)
