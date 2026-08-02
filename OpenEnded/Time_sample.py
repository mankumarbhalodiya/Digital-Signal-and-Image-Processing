import librosa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

original, sr = librosa.load("Balam_Pichkari_Original.mp3", sr=None, mono=True)
karaoke, sr = librosa.load("Balam_Pichkari_karaoke.mp3", sr=None, mono=True)
different, sr = librosa.load("Aayi_Nai_-Stree_2.mp3", sr=None, mono=True)

#it take a 15 seconds to give response
duration = 15
samples = sr * duration

original = original[:samples]
karaoke = karaoke[:samples]
different = different[:samples]

# for the same length
min_len = min(len(original), len(karaoke), len(different))

original = original[:min_len]
karaoke = karaoke[:min_len]
different = different[:min_len]

corr_ok = np.corrcoef(original, karaoke)[0, 1]
corr_od = np.corrcoef(original, different)[0, 1]
corr_kd = np.corrcoef(karaoke, different)[0, 1]

print(f"Original vs Karaoke        : {corr_ok:.4f}")
print(f"Original vs Different Song : {corr_od:.4f}")
print(f"Karaoke vs Different Song  : {corr_kd:.4f}")

cross_ok = np.correlate(original, karaoke, mode='same')
cross_od = np.correlate(original, different, mode='same')
cross_kd = np.correlate(karaoke, different, mode='same')

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(cross_ok)
plt.title("Cross Correlation: Original vs Karaoke")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(cross_od)
plt.title("Cross Correlation: Original vs Different Song")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(cross_kd)
plt.title("Cross Correlation: Karaoke vs Different Song")
plt.grid(True)

plt.tight_layout()
plt.show()