import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Get the folder where the Python file is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs")
os.makedirs(output_dir, exist_ok=True)

# Degraded images
degraded_files = [
    "1.jpg",
    "2.jpg",
    "3.jpg",
    "4.jpg",
    "5.jpg"
]


reference_files = [
    "1_ref.jpg",
    "2_ref.jpg",
    "3_ref.jpg",
    "4_ref.jpg",
    "5_ref.jpg"
]

for degraded_file, reference_file in zip(degraded_files, reference_files):

    # Paths
    degraded_path = os.path.join(script_dir, degraded_file)
    reference_path = os.path.join(script_dir, reference_file)

    # Read images in grayscale
    image = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    # Check images
    if image is None:
        print("Degraded image not found:", degraded_file)
        continue

    if reference is None:
        print("Reference image not found:", reference_file)
        continue

    # --------------------------------------------------
    # Histogram Equalization
    # --------------------------------------------------

    equalized = cv2.equalizeHist(image)

    # --------------------------------------------------
    # Histogram Matching
    # --------------------------------------------------

    matched = match_histograms(image, reference)

    # Convert to uint8
    matched = np.clip(matched, 0, 255).astype(np.uint8)

    # --------------------------------------------------
    # Display Images & Histograms (2x4 Grid)
    # --------------------------------------------------

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Images
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("1. Degraded Image", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(reference, cmap="gray")
    axes[0, 1].set_title("2. Reference Image", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(equalized, cmap="gray")
    axes[0, 2].set_title("3. Histogram Equalized", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(matched, cmap="gray")
    axes[0, 3].set_title("4. Histogram Matched", fontsize=10, fontweight="bold")
    axes[0, 3].axis("off")

    # Row 2: Histograms
    axes[1, 0].hist(image.ravel(), bins=256, range=[0, 256], color="gray")
    axes[1, 0].set_title("Degraded Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(reference.ravel(), bins=256, range=[0, 256], color="green")
    axes[1, 1].set_title("Reference Histogram")
    axes[1, 1].set_xlabel("Intensity")

    axes[1, 2].hist(equalized.ravel(), bins=256, range=[0, 256], color="blue")
    axes[1, 2].set_title("Equalized Histogram")
    axes[1, 2].set_xlabel("Intensity")

    axes[1, 3].hist(matched.ravel(), bins=256, range=[0, 256], color="purple")
    axes[1, 3].set_title("Matched Histogram")
    axes[1, 3].set_xlabel("Intensity")

    plt.suptitle(f"Image Enhancement Comparison: {degraded_file} vs {reference_file}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save figure plot to Outputs folder
    base_name = os.path.splitext(degraded_file)[0]
    fig_path = os.path.join(output_dir, f"result_{base_name}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(degraded_file, "processed successfully. Output saved to Outputs/result_" + base_name + ".png")

print("All images processed successfully.")