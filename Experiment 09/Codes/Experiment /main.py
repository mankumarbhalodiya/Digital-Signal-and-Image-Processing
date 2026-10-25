import cv2
import numpy as np

image = cv2.imread('1.jpeg')
if image is None:
    print("Error: Could not load 1.jpg")
    exit()

kernel_size = (5, 5)
sigma = 1.5

gaussian_1d = cv2.getGaussianKernel(5, sigma)
gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)
smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)

sharpening_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

cv2.imshow("Original Image", image)

cv2.imshow("Smoothed Image", smoothed_image)

cv2.imshow("Sharpened Image", sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
