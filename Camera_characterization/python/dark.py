import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
from scipy.stats import linregress
from astropy.io import fits

# ==== USER PATHS ====
BASE_PATH = r"C:\Users\palak\Desktop\camera_characterization"
DARK_PATH = os.path.join(BASE_PATH, "dark")
RESULTS_PATH = os.path.join(BASE_PATH, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

# ==== HELPER FUNCTIONS ====

def extract_exposure_us(filename: str) -> int:
    """Extract exposure time from filename like 'dark_exp_30_n1'."""
    match = re.search(r'exp_(\d+)', filename)
    return int(match.group(1)) if match else None

def load_fits_image(path: str) -> np.ndarray:
    """Load FITS image into a NumPy array."""
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)

# ==== MAIN FUNCTION ====

def analyze_dark_current():
    print("==== Analyzing Dark Current ====")

    fits_paths = sorted(glob(os.path.join(DARK_PATH, "*.fits")))
    if not fits_paths:
        print("❌ No dark frames found.")
        return

    # Group by exposure time
    exposure_dict = {}
    for path in fits_paths:
        exp = extract_exposure_us(os.path.basename(path))
        if exp is not None:
            exposure_dict.setdefault(exp, []).append(path)

    exposures = sorted(exposure_dict.keys())
    mean_signals = []

    for idx, exp in enumerate(exposures):
        print(f"📷 Exposure: {exp} µs")

        frames = [load_fits_image(f) for f in exposure_dict[exp]]
        stack = np.stack(frames, axis=0)
        avg_frame = np.mean(stack, axis=0)
        temporal_noise = np.std(stack, axis=0)

        mean_signal = np.mean(avg_frame)
        mean_signals.append(mean_signal)
        print(f"   Mean Signal: {mean_signal:.3f}")

        # Only plot frame/noise for first exposure set
        if idx == 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            im1 = axes[0].imshow(avg_frame, cmap='gray', vmin=0, vmax=0.1)
            axes[0].set_title(f"Mean Dark Frame ({exp}ms)")
            plt.colorbar(im1, ax=axes[0], label='ADU')

            im2 = axes[1].imshow(temporal_noise, cmap='viridis', vmin=0, vmax=0.3)
            axes[1].set_title(f"Temporal Noise ({exp}ms)")
            plt.colorbar(im2, ax=axes[1], label='ADU')

            for ax in axes:
                ax.set_xlabel('')
                ax.set_ylabel('')

            plt.tight_layout()
            noise_path = os.path.join(RESULTS_PATH, f"dark_mean_and_noise_{exp}us.png")
            plt.savefig(noise_path, dpi=300)
            plt.close()
            print(f"📊 Saved mean + noise image: {noise_path}")

    # === Linearity plot ===
    slope, intercept, r_value, _, _ = linregress(exposures, mean_signals)
    print(f"\n📈 Signal = {slope:.6f} * Exposure + {intercept:.6f}")
    print(f"   R² = {r_value**2:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(exposures, mean_signals, "o", label="Mean signal")
    x_fit = np.linspace(min(exposures), max(exposures), 100)
    plt.plot(x_fit, slope * x_fit + intercept, "r--", label="Linear fit")
    plt.xlabel("Exposure Time (µs)")
    plt.ylabel("Mean Signal")
    plt.title("Dark Current Linearity")
    plt.grid(True)
    plt.legend()
    linearity_path = os.path.join(RESULTS_PATH, "dark_current_linearity.png")
    plt.savefig(linearity_path, dpi=300)
    plt.close()
    print(f"✅ Linearity plot saved: {linearity_path}")

# ==== RUN ====
if __name__ == "__main__":
    analyze_dark_current()
