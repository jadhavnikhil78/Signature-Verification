import cv2
import numpy as np

def preprocessing(input):
    # preprocessing the image input
    clean_image = cv2.fastNlMeansDenoising(input)
    # thresholding the image to 0 and 1
    __, tresholded = cv2.threshold(clean_image, 200, 1, cv2.THRESH_BINARY_INV)
    # just to convert the source to CV_8UC1, because the src for findNonZero function needs CV_8UC1 format
    imagegray = cv2.cvtColor(tresholded, cv2.COLOR_BGR2GRAY)
    # cropping the signature portion of the image
    img = signature_crop(imagegray)

    # 40x10 image as a flatten array
    raveled_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_AREA).flatten()

    # resize to 400x100
    increase_resolution = cv2.resize(img, (400, 100), interpolation=cv2.INTER_AREA)

    # Signature Features
    columns = np.sum(increase_resolution, axis=0)  # sum of all columns
    lines = np.sum(increase_resolution, axis=1)  # sum of all lines
    h, w = img.shape
    aspect_ratio = w / h

    return [*raveled_img, *columns, *lines, aspect_ratio]

def signature_crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]

