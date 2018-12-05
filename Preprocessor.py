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
    
    #calculate projections along the x and y axes
    xp = np.sum(increase_resolution,axis=0)
    yp = np.sum(increase_resolution,axis=1)
    
    height, width = increase_resolution.shape
    x = range(width)  # cols value
    y = range(height)  # rows value
    
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(increase_resolution))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(increase_resolution))
    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(increase_resolution) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(increase_resolution) * sy**3)
    
    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(increase_resolution) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(increase_resolution) * sy**4) - 3

    return [*raveled_img, *columns, *lines, skewx, skewy, kurtx, kurty, aspect_ratio]

def signature_crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]

