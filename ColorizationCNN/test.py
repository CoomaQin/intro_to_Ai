import numpy as np
import skimage.color as color
from skimage import io
import matplotlib.pyplot as plt
import cv2

# rgb = io.imread("nasa.jpg")
# lab = color.rgb2lab(rgb)
# l_img = lab[:, :, 0]
# a_img = lab[:, :, 1]
# b_img = lab[:, :, 2]
# fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))
#
# ax0.imshow(rgb)
# ax0.set_title("RGB image")
# ax0.axis('off')
# ax1.imshow(a_img)
# ax1.set_title("a channel")
# ax1.axis('off')
# ax2.imshow(l_img)
# ax2.set_title("L channel")
# ax2.axis('off')
# ax3.imshow(b_img)
# ax3.set_title("b channel")
# ax3.axis('off')
# plt.show()

input = cv2.imread('nasa.jpg')
# cv2.imshow('Hello World', input)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

lab = cv2.cvtColor(input, cv2.COLOR_BGR2LAB)
# cv2.imshow("l*a*b", lab)

L, A, B = cv2.split(lab)
# cv2.imshow("L_Channel", L)  # For L Channel
# cv2.imshow("A_Channel", A)  # For A Channel (Here's what You need)
cv2.imshow("B_Channel", B)  # For B Channel

cv2.waitKey(0)
cv2.destroyAllWindows()
