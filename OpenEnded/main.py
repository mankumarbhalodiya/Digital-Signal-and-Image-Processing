
import numpy as np
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os

original_file = "Balam_Pichkari_Original.mp3"
karaoke_file = "Balam_Pichkari_karaoke.mp3"
different_file = "Aayi_Nai_-Stree_2.mp3"

for file in [original_file, karaoke_file, different_file]:
    if not os.path.exists(file):
        print("File not found:", file)
        exit()

sr = 22050

original, _ = librosa.load(original_file, sr=sr, mono=True)
karaoke, _ = librosa.load(karaoke_file, sr=sr, mono=True)
different, _ = librosa.load(different_file, sr=sr, mono=True)

length = min(len(original), len(karaoke), len(different))

original = original[:length]
karaoke = karaoke[:length]
different = different[:length]

original = original / np.max(np.abs(original))
karaoke = karaoke / np.max(np.abs(karaoke))
different = different / np.max(np.abs(different))

def normalized_correlation(x, y):
    corr = correlate(x, y, mode="full")
    value = np.max(np.abs(corr)) / (np.linalg.norm(x) * np.linalg.norm(y))
    return corr, value

corr1, val1 = normalized_correlation(original, karaoke)
corr2, val2 = normalized_correlation(original, different)
corr3, val3 = normalized_correlation(karaoke, different)

print("--- Normalized Correlation Results ---")
print(f"Original vs Karaoke   : {val1:.4f}")
print(f"Original vs Different : {val2:.4f}")
print(f"Karaoke vs Different  : {val3:.4f}")
print("--------------------------------------")

auto_original = correlate(original, original, mode="full")
auto_karaoke = correlate(karaoke, karaoke, mode="full")
auto_different = correlate(different, different, mode="full")

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(auto_original)
plt.title("Autocorrelation - Original")

plt.subplot(3, 1, 2)
plt.plot(auto_karaoke)
plt.title("Autocorrelation - Karaoke")

plt.subplot(3, 1, 3)
plt.plot(auto_different)
plt.title("Autocorrelation - Different")

plt.tight_layout()
plt.savefig("autocorrelation.png")
plt.show()

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(corr1)
plt.title("Original vs Karaoke")

plt.subplot(3, 1, 2)
plt.plot(corr2)
plt.title("Original vs Different")

plt.subplot(3, 1, 3)
plt.plot(corr3)
plt.title("Karaoke vs Different")

plt.tight_layout()
plt.savefig("cross_correlation.png")
plt.show()

pairs = [
    "Original\nKaraoke",
    "Original\nDifferent",
    "Karaoke\nDifferent"
]

values = [val1, val2, val3]

plt.figure(figsize=(6, 4))
plt.bar(pairs, values)
plt.title("Normalized Correlation Comparison")
plt.ylabel("Correlation Value")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.savefig("correlation_bar.png")
plt.show()
