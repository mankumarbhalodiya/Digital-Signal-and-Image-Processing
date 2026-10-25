import cv2
import numpy as np
image = cv2.imread('1.jpeg')
if image is None:
    print("Error: Could not load 1.jpeg")
    exit()
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
laplacian_kernel = np.array([
    [0, -1,  0],
    [-1,  5, -1],
    [0, -1,  0]
], dtype=np.float32)
sharpened_image = cv2.filter2D(
    blurred_image, -1, laplacian_kernel
)
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
