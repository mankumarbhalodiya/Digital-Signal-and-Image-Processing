import librosa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load only first 5 seconds
original, sr = librosa.load(
    "Balam_Pichkari_Original.mp3",
    sr=None,
    mono=True,
    duration=5
)

karaoke, sr = librosa.load(
    "Balam_Pichkari_karaoke.mp3",
    sr=None,
    mono=True,
    duration=5
)

different, sr = librosa.load(
    "Aayi_Nai_-Stree_2.mp3",
    sr=None,
    mono=True,
    duration=5
)

# Make same length
min_len = min(len(original), len(karaoke), len(different))
original = original[:min_len]
karaoke = karaoke[:min_len]
different = different[:min_len]

# Pearson Correlation
print("Original vs Karaoke   :", np.corrcoef(original, karaoke)[0,1])
print("Original vs Different :", np.corrcoef(original, different)[0,1])
print("Karaoke vs Different  :", np.corrcoef(karaoke, different)[0,1])

# Cross Correlation
cross_ok = np.correlate(original, karaoke, mode="same")
cross_od = np.correlate(original, different, mode="same")
cross_kd = np.correlate(karaoke, different, mode="same")

# Plot
plt.figure(figsize=(10,7))

plt.subplot(311)
plt.plot(cross_ok)
plt.title("Original vs Karaoke")
plt.grid()

plt.subplot(312)
plt.plot(cross_od)
plt.title("Original vs Different")
plt.grid()

plt.subplot(313)
plt.plot(cross_kd)
plt.title("Karaoke vs Different")
plt.grid()

plt.tight_layout()
plt.show()
