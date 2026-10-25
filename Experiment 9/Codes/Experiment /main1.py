import cv2
import numpy as np
image = cv2.imread('1.jpeg')
if image is None:
    print("Error: Could not load 1.jpeg")
    exit()
kernel_size = (5, 5)
kernel = np.ones(kernel_size, dtype=np.float32) / (
    kernel_size[0] * kernel_size[1]
)
smoothed_image = cv2.filter2D(image, -1, kernel)
cv2.imshow("Original Image", image)
cv2.imshow("Smoothed Image", smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
