import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob

# === CONFIGURATION ===
BASE_PATH = r"C:\Users\palak\Desktop\camera_characterization"
DARK_PATH = os.path.join(BASE_PATH, "dark")
RESULTS_DIR = os.path.join(BASE_PATH, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === LOAD DARK FRAMES ===
print("==== Analyzing Dark Current ====")
dark_files = sorted(glob(os.path.join(DARK_PATH, "*.fits")))
if not dark_files:
    print("❌ No dark frames found.")
    exit()

print(f"🟢 Found {len(dark_files)} dark frames.")
stacked = []
for f in dark_files:
    with fits.open(f) as hdul:
        stacked.append(hdul[0].data.astype(np.float32))

stacked = np.array(stacked)
master_dark = np.median(stacked, axis=0)

# === CALCULATE THRESHOLD ===
median_val = np.median(master_dark)
mad = np.median(np.abs(master_dark - median_val))
threshold = median_val + 5 * mad
print(f"🔍 Median: {median_val:.2f}, MAD: {mad:.2f}, Threshold: {threshold:.2f}")

# === HOT PIXEL DETECTION ===
hot_pixels = master_dark > threshold
num_hot_pixels = np.sum(hot_pixels)
print(f"🔥 Number of hot pixels: {num_hot_pixels}")

# === SAVE MASTER DARK ===
fits.writeto(os.path.join(RESULTS_DIR, "master_dark.fits"), master_dark, overwrite=True)

# === SAVE HOT PIXEL MAP ===
fits.writeto(os.path.join(RESULTS_DIR, "hot_pixel_map.fits"), hot_pixels.astype(np.uint8), overwrite=True)

# === SAVE CSV OF HOT PIXELS ===
coords = np.column_stack(np.where(hot_pixels))
csv_path = os.path.join(RESULTS_DIR, "hot_pixels.csv")
np.savetxt(csv_path, coords, fmt='%d', delimiter=",", header="y,x", comments='')
print(f"📄 Hot pixel coordinates saved to: {csv_path}")

# === SAVE TEXT REPORT ===
with open(os.path.join(RESULTS_DIR, "dark_report.txt"), "w") as f:
    f.write("===== Dark Current Report =====\n")
    f.write(f"Number of dark frames: {len(dark_files)}\n")
    f.write(f"Master dark shape: {master_dark.shape}\n")
    f.write(f"Median: {median_val:.2f}\n")
    f.write(f"MAD: {mad:.2f}\n")
    f.write(f"Threshold (median + 5×MAD): {threshold:.2f}\n")
    f.write(f"Hot pixels detected: {num_hot_pixels}\n")

# === PLOT 1: HOT PIXEL IMAGE ===
plt.figure(figsize=(10, 4))
plt.imshow(master_dark, cmap='hot')
plt.colorbar(label='Pixel Value')
plt.title('Master Dark Frame')
plt.savefig(os.path.join(RESULTS_DIR, "master_dark.png"), dpi=300)
plt.close()

# === PLOT 2: HOT PIXEL MAP ===
plt.figure(figsize=(10, 4))
plt.imshow(hot_pixels, cmap='gray')
plt.title('Hot Pixel Map')
plt.savefig(os.path.join(RESULTS_DIR, "hot_pixel_map.png"), dpi=300)
plt.close()

# === PLOT 3: HISTOGRAM ===
plt.figure(figsize=(8, 4))
plt.hist(master_dark.flatten(), bins=200, color='skyblue', alpha=0.9)
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Dark Frame Pixel Value Histogram')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dark_histogram.png"), dpi=300)
plt.close()

print("✅ Analysis complete. All outputs saved to:", RESULTS_DIR)
