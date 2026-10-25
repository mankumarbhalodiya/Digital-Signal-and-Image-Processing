import cv2
import numpy as np
image = cv2.imread('1.jpeg')
if image is None:
    print("Error: Could not load 1.jpeg")
    exit()
kernel_size = 5
smoothed_image = cv2.medianBlur(image, kernel_size)
cv2.imshow("Original Image", image)
cv2.imshow("Median Filtered Image", smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
