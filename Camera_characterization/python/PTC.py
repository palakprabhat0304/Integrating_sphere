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
    # Extract exposure time from filename (assuming format PTC_exp_XXX_nY)
    base = os.path.basename(file)
    parts = base.split('_')
    
    try:
        # Handle both formats: PTC_exp_30_n1 and PTC_exp_8265_n1
        # The exposure time is the number after 'exp_'
        exp_part = parts[2]
        if exp_part.startswith('exp'):
            # Handle case where 'exp' might be separate
            exp_time = int(parts[3])
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

# Sort by exposure time
sorted_exposures = sorted(ptc_data.keys())

# Process each exposure group
means = []
variances = []
exposure_times = []

for exp_time in sorted_exposures:
    files = ptc_data[exp_time]
    frames = []
    
    # Load all frames for this exposure
    for file in files:
        with fits.open(file) as hdul:
            data = hdul[0].data.astype(np.float32)
            frames.append(data)
    
    if not frames:
        continue
    
    # Stack frames to compute statistics
    stack = np.stack(frames)
    
    # Calculate mean and variance across the stack
    mean_frame = np.mean(stack, axis=0)
    variance_frame = np.var(stack, axis=0, ddof=1)
    
    # Use central region to avoid edge effects (optional)
    h, w = mean_frame.shape
    roi = (slice(h//4, 3*h//4), slice(w//4, 3*w//4))
    
    # Store results
    means.append(np.mean(mean_frame[roi]))
    variances.append(np.mean(variance_frame[roi]))
    exposure_times.append(exp_time)

# Convert to numpy arrays
means = np.array(means)
variances = np.array(variances)
exposure_times = np.array(exposure_times)

# Sort by mean signal (just in case)
sort_idx = np.argsort(means)
means = means[sort_idx]
variances = variances[sort_idx]
exposure_times = exposure_times[sort_idx]

# Fit linear region to get gain
# Typically use the middle 50% of data to avoid nonlinearities at extremes
n_points = len(means)
start_idx = n_points // 4
end_idx = 3 * n_points // 4

if n_points < 2:
    raise ValueError("Not enough data points for PTC analysis. Need at least 2 different exposure levels.")

slope, intercept, r_value, p_value, std_err = linregress(
    means[start_idx:end_idx], variances[start_idx:end_idx]
)

gain = 1.0 / slope  # in electrons/DN
read_noise = intercept * gain  # in electrons

# Create the PTC plot
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(means, variances, c='b', label='Data')

# Plot linear fit
fit_line = slope * means + intercept
plt.plot(means, fit_line, 'r-', 
         label=f'Fit: slope={slope:.4f}\nGain={gain:.2f} e-/DN\nRead noise={read_noise:.2f} e-')

# Plot ideal photon noise line (slope=1)
ideal_photon_noise = means
plt.plot(means, ideal_photon_noise, 'g--', label='Ideal photon noise (slope=1)')

# Formatting
plt.xlabel('Mean Signal (DN)')
plt.ylabel('Variance (DN$^2$)')
plt.title('Photon Transfer Curve (PTC)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.xscale('log')
plt.yscale('log')

# Save the plot
output_path = os.path.join(ptc_folder, "ptc_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"PTC Analysis Results:")
print(f"---------------------")
print(f"Gain: {gain:.2f} electrons/DN")
print(f"Read noise: {read_noise:.2f} electrons")
print(f"Linear fit correlation coefficient (R): {r_value:.4f}")
print(f"Plot saved to: {output_path}")

# Optional: Save results to a text file
results_file = os.path.join(ptc_folder, "ptc_results.txt")
with open(results_file, 'w') as f:
    f.write("PTC Analysis Results\n")
    f.write("-------------------\n")
    f.write(f"Gain: {gain:.2f} electrons/DN\n")
    f.write(f"Read noise: {read_noise:.2f} electrons\n")
    f.write(f"Linear fit correlation coefficient (R): {r_value:.4f}\n")
    f.write("\nRaw Data:\n")
    f.write("Exposure_time, Mean_signal(DN), Variance(DN^2)\n")
    for exp, m, v in zip(exposure_times, means, variances):
        f.write(f"{exp}, {m:.2f}, {v:.2f}\n")

print(f"Results saved to: {results_file}")