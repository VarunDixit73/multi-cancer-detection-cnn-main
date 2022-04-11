from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

IMG_WIDTH, IMG_HEIGHT = (240, 240)

def saved_model(filepath='models/cnn-parameters-improvement-23-0.91.model'):
    return load_model(filepath)

def crop_brain_contour(image):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]                
    return new_image

def predict(filepath):

    image = cv2.imread(filepath)
    # crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image)
    # resize image
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.
    xdata =  [image]
    # load model
    model = saved_model()
    # predict
    y_pred = model.predict(xdata)
    result = np.argmax(y_pred, axis=1)[0]
    return result
