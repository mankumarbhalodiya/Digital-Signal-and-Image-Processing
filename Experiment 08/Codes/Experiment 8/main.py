import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "4.jpg")
reference_path = os.path.join(script_dir, "3.jpg")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    plt.figure(figsize=(8, 6))
    plt.title("Original Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

    equalized_image = cv2.equalizeHist(image)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Equalized Image")
    plt.imshow(equalized_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    equalized_histogram = cv2.calcHist(
        [equalized_image], [0], None, [256], [0, 256]
    )

    plt.figure(figsize=(8, 6))
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(equalized_histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    if reference is None:
        print("Reference image not found")
    else:
        source_hist = cv2.calcHist(
            [image], [0], None, [256], [0, 256]
        )

        reference_hist = cv2.calcHist(
            [reference], [0], None, [256], [0, 256]
        )

        source_hist = source_hist / source_hist.sum()
        reference_hist = reference_hist / reference_hist.sum()

        source_cdf = source_hist.cumsum()
        reference_cdf = reference_hist.cumsum()

        mapping = np.interp(
            source_cdf,
            reference_cdf,
            np.arange(256)
        )

        matched_image = mapping[image].astype(np.uint8)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Source Image")
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Reference Image")
        plt.imshow(reference, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Matched Image")
        plt.imshow(matched_image, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()