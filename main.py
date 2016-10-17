import cv2 
import numpy as np

#-------------Thresholding Image--------------#

src_img = cv2.imread('./data/img_1.jpg', 0)
# blur = cv2.GaussianBlur(src_img,(3,3),0)
gud_img = cv2.adaptiveThreshold(src_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,73,2)
ret3,thr_img = cv2.threshold(gud_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# gud_img = cv2.fastNlMeansDenoising(gud_img, None,20,7,21)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
closing = cv2.morphologyEx(thr_img, cv2.MORPH_CLOSE, kernel)

# ret,thr_img = cv2.threshold(src_img,127,255,cv2.THRESH_BINARY)


#-------------Displaying Image----------------#

cv2.namedWindow('Source Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Threshold Image', cv2.WINDOW_NORMAL)
cv2.imshow("Source Image", src_img)
cv2.imshow("Threshold Image", closing)
cv2.waitKey(0)
