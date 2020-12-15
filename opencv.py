import cv2, sys
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('n02691156_196.jpg')
image_gray = cv2.imread('n02691156_196.jpg', cv2.IMREAD_GRAYSCALE)

b,g,r = cv2.split(image)
image2 = cv2.merge([r,g,b])

plt.imshow(image2)
plt.xticks([])
plt.yticks([])
plt.show()

blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127,255,cv2.THRESH_BINARY)

blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

edged = cv2.Canny(blur, 10, 250)
cv2.imshow('Edged', edged)
cv2.waitKey(0)

