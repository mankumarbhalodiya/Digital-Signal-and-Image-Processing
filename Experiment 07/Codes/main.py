import glob, os
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

files = [os.path.join(IMAGE_DIR, f"{i}.jpg") for i in range(1, 6)]
for f in files:
    if not os.path.exists(f):
        files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))[:5]
        break

images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]

def contrast_stretch(img):
    min_val, max_val = np.min(img), np.max(img)
    if max_val == min_val:
        return img
    return np.uint8((img.astype(np.float32) - min_val) * 255.0 / (max_val - min_val))

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def gamma_correction(img, gamma=0.4):
    norm = img / 255.0
    return np.uint8(np.power(norm, gamma) * 255.0)

def log_transform(img):
    norm = img.astype(np.float32)
    c = 255.0 / np.log(1.0 + np.max(norm))
    return np.uint8(c * np.log1p(norm))

def image_negative(img):
    return 255 - img

methods = [
    ("Contrast Stretching", contrast_stretch),
    ("Histogram Equalization", histogram_equalization),
    ("Gamma Transformation (γ=0.4)", gamma_correction),
    ("Log Transformation", log_transform),
    ("Image Negative", image_negative)
]

enhanced_images = []

for i, (name, func) in enumerate(methods):
    enh = func(images[i])
    enhanced_images.append(enh)
    cv2.imwrite(os.path.join(BASE_DIR, f"enhanced_image_{i+1}.png"), enh)

fig, axes = plt.subplots(5, 2, figsize=(10, 18))
for i in range(5):
    name, _ = methods[i]
    
    axes[i, 0].imshow(images[i], cmap="gray")
    axes[i, 0].set_title(f"Image {i+1} - Original", fontsize=11, fontweight="bold")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(enhanced_images[i], cmap="gray")
    axes[i, 1].set_title(f"Image {i+1} - {name}", fontsize=11, fontweight="bold")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "comparison_results.png"), dpi=300, bbox_inches="tight")
plt.close()

fig_hist, axes_hist = plt.subplots(5, 2, figsize=(12, 16))
for i in range(5):
    name, _ = methods[i]
    
    axes_hist[i, 0].hist(images[i].ravel(), bins=256, range=[0, 256], color="gray", alpha=0.8)
    axes_hist[i, 0].set_title(f"Image {i+1} Original Histogram", fontsize=10)
    axes_hist[i, 0].set_xlim([0, 256])

    axes_hist[i, 1].hist(enhanced_images[i].ravel(), bins=256, range=[0, 256], color="indigo", alpha=0.8)
    axes_hist[i, 1].set_title(f"Image {i+1} Enhanced Histogram ({name})", fontsize=10)
    axes_hist[i, 1].set_xlim([0, 256])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "histogram_analysis.png"), dpi=300, bbox_inches="tight")
plt.close()