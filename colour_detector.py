#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:34:44 2019

@author: alkesha
"""
import cv2
import numpy as np

#range of colour which we want to detect
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20,255,255], dtype = "uint8")

#capturaning of video 
cam = cv2.VideoCapture(0)

#resizing of image frame
def iresize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    h = image.shape[0]
    w=image.shape[1]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized 
#captuting of frames
while True:
    #frame capturing
    value,frame = cam.read()
    #frame resizing
    frame = iresize(frame, width = 500)
    #RBG TO HSV Conversion
    convert = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #masking
    mask = cv2.inRange(convert, lower, upper)
    #kernel declaration
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    #masking
    
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask=cv2.dilate(mask,kernel,iterations=2)
    #bluring with gaussian blur
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    #bitwise and
    skin = cv2.bitwise_and(frame, frame, mask = mask)
    #video display
    cv2.imshow("images",skin)
    # q press for stopping
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#frames release    
cam.release()
#destroying all windows
cv2.destroyAllWindows()