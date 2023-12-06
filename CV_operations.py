import cv2

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

def halfOp(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)[1]
    noN= noise_removal(thresh)
    noN=thin_font(noN)
    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    # dilate = cv2.dilate(noN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 60)), iterations=3)
    # dilate = cv2.dilate(thresh, kernel, iterations=2)
    # cv2.imwrite("box1.jpg", (dilate))
    return noN

# To be Moved to cvOperation.py
