import cv2
import numpy as np

# 500 x 250


def adding():
    img1 = cv2.imread('images/pic3.png')
    img2 = cv2.imread('images/pic5.png')
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',img2gray)
    ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('mask',mask)
    cv2.imshow('ret',ret)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mask_inv',mask_inv)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst
    cv2.imshow('res',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def enhance():
    img = cv2.imread('images/bookpage.jpg')
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow('original',img)
    cv2.imshow('Adaptive threshold',th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def matching():
    img_rgb = cv2.imread('images/pic1.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('images/pic2.jpg',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.1
    loc = np.where( res >= threshold)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        cv2.imshow('Detected',img_rgb)

    cv2.waitKey(0)

matching()