import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_filter(image, sigma):
    image = np.float32(image)

    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    h, w = image.shape
    cy = h // 2
    cx = w // 2

    y, x = np.ogrid[:h, :w]

    distance = (x - cx) ** 2 + (y - cy) ** 2

    gaussian = np.exp(
        -distance / (2 * sigma * sigma)
    )

    mask = np.zeros((h, w, 2), np.float32)
    mask[:, :, 0] = gaussian
    mask[:, :, 1] = gaussian

    result = dft_shift * mask
    result = np.fft.ifftshift(result)

    output = cv2.idft(
        result,
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )

    output = cv2.normalize(
        output,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(output)


def high_pass_filter(image, strength):
    image = np.float32(image)

    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    h, w = image.shape
    cy = h // 2
    cx = w // 2

    mask = np.ones((h, w, 2), np.float32)

    mask[
        cy - strength:cy + strength,
        cx - strength:cx + strength
    ] = 0

    result = dft_shift * mask
    result = np.fft.ifftshift(result)

    output = cv2.idft(
        result,
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )

    output = cv2.normalize(
        output,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(output)


image_paths = [
    "smoothening data1.png",
    "smoothening data2.png",
    "smoothening data3.png",
    "smoothening data4.png",
    "smoothening data5.png"
]

sigma = 20
strength = 20


for i in range(5):

    image = cv2.imread(
        image_paths[i],
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    smooth = gaussian_filter(image, sigma)
    sharp = high_pass_filter(image, strength)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(smooth, cmap="gray")
    plt.title("Gaussian Smoothing")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sharp, cmap="gray")
    plt.title("High-Pass Sharpening")
    plt.axis("off")

    plt.suptitle("Experiment 9 - Image " + str(i + 1))
    plt.tight_layout()
    plt.show()

