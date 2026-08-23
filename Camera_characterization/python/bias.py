import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from glob import glob
from scipy.stats import linregress
from tqdm import tqdm

# Set paths
desktop_path = os.path.expanduser("~/Desktop")
ptc_folder = os.path.join(desktop_path, "camera_characterization", "PTC")

# Get all PTC frames
ptc_files = glob(os.path.join(ptc_folder, "*.fits"))  # assuming FITS files
if not ptc_files:
    raise FileNotFoundError("No PTC frames found in the specified folder")

# Organize files by exposure time
ptc_data = {}
for file in ptc_files:
    base = os.path.basename(file)
    parts = base.split('_')
    
    try:
        # Extract exposure time (handle both PTC_exp_30_n1 and similar formats)
        if parts[1] == 'exp':
            exp_time = int(parts[2])
        else:
            # Handle case where 'exp' might be merged with number
            exp_part = parts[1]
            if exp_part.startswith('exp'):
                exp_time = int(exp_part[3:])
            else:
                exp_time = int(exp_part)
                
        if exp_time not in ptc_data:
            ptc_data[exp_time] = []
        ptc_data[exp_time].append(file)
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse exposure time from filename {base}")
        continue

if not ptc_data:
    raise ValueError("No valid PTC files found with recognizable exposure times")

# Find the shortest exposure (should be bias frames)
bias_exposure = min(ptc_data.keys())
print(f"\nFound bias frames with exposure time: {bias_exposure}")

# Load all bias frames
bias_frames = []
for file in ptc_data[bias_exposure]:
    with fits.open(file) as hdul:
        data = hdul[0].data.astype(np.float32)
        bias_frames.append(data)

if not bias_frames:
    raise ValueError("No valid bias frames found")

# Create master bias
master_bias = np.median(bias_frames, axis=0)

# Calculate bias statistics
bias_mean = np.mean(master_bias)
bias_std = np.std(master_bias)
bias_median = np.median(master_bias)
bias_min = np.min(master_bias)
bias_max = np.max(master_bias)

# Plot bias frame
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(master_bias, cmap='gray')
plt.title(f'Master Bias (exp={bias_exposure})')
plt.colorbar(label='DN')

plt.subplot(1, 3, 2)
plt.hist(master_bias.ravel(), bins=100, log=True)
plt.title('Bias Value Distribution')
plt.xlabel('DN')
plt.ylabel('Pixel Count (log)')

plt.subplot(1, 3, 3)
row_means = np.mean(master_bias, axis=1)
plt.plot(row_means)
plt.title('Row-wise Bias Variation')
plt.xlabel('Row number')
plt.ylabel('Mean DN')

plt.tight_layout()
bias_plot_path = os.path.join(ptc_folder, "bias_analysis.png")
plt.savefig(bias_plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Save master bias
master_bias_path = os.path.join(ptc_folder, "master_bias.fits")
fits.writeto(master_bias_path, master_bias.astype(np.float32), overwrite=True)

print("\nBias Analysis Results:")
print("---------------------")
print(f"Mean bias level: {bias_mean:.2f} DN")
print(f"Bias standard deviation: {bias_std:.2f} DN")
print(f"Bias median: {bias_median:.2f} DN")
print(f"Bias range: {bias_min:.2f} to {bias_max:.2f} DN")
print(f"Master bias saved to: {master_bias_path}")
print(f"Bias plots saved to: {bias_plot_path}")

# Now proceed with PTC analysis using bias-subtracted frames if needed
# (Optional: subtract bias from all frames before PTC analysis)