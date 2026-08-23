import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from glob import glob

# =============================================
# CONFIGURATION
# =============================================
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "camera_characterization")
flat_pattern = "flat_field_exp_*.*"  # Adjust if your files are named differently
results_dir = os.path.join(base_path, "results", "flat_field_analysis")

# Create results directory
os.makedirs(results_dir, exist_ok=True)

# =============================================
# LOAD FLAT FRAMES
# =============================================
def load_flat_frames():
    """Load all flat frames with validation"""
    flat_files = sorted(glob(os.path.join(base_path, "flat_field", flat_pattern)))
    
    if not flat_files:
        available_files = os.listdir(os.path.join(base_path, "flat_field"))
        raise FileNotFoundError(
            f"No flat frames found matching '{flat_pattern}'.\n"
            f"Files actually present: {available_files}"
        )
    
    print(f"Found {len(flat_files)} flat frames:")
    for f in flat_files[:3]:  # Show first 3 as sample
        print(f"  {os.path.basename(f)}")
    if len(flat_files) > 3:
        print(f"  (...and {len(flat_files)-3} more)")
    
    return np.array([io.imread(f) for f in flat_files])

# =============================================
# ANALYSIS FUNCTIONS
# =============================================
def analyze_flat_frames(frames):
    """Calculate flat field statistics and corrections"""
    print("\nCalculating flat field properties...")
    
    # 1. Mean flat field
    mean_flat = np.mean(frames, axis=0)
    
    # 2. Normalized flat field (for correction)
    flat_normalized = mean_flat / np.mean(mean_flat)
    
    # 3. Vignetting and PRNU analysis
    prnu_map = np.std(frames, axis=0) / mean_flat  # PRNU = noise/signal
    vignetting_map = (mean_flat - np.min(mean_flat)) / (np.max(mean_flat) - np.min(mean_flat))
    
    # Statistics
    stats = {
        "avg_signal_level": np.mean(mean_flat),
        "max_vignetting": 100 * (1 - np.min(mean_flat)/np.max(mean_flat)),
        "avg_prnu": 100 * np.mean(prnu_map),
    }
    
    # Generate plots
    plt.figure(figsize=(15, 5))
    
    # Mean flat field
    plt.subplot(1, 3, 1)
    plt.imshow(mean_flat, cmap='gray')
    plt.colorbar(label='Signal (DN)')
    plt.title("Mean Flat Field")
    
    # Vignetting map
    plt.subplot(1, 3, 2)
    plt.imshow(vignetting_map, cmap='viridis', vmin=0.5, vmax=1.0)
    plt.colorbar(label='Normalized Intensity')
    plt.title("Vignetting Map")
    
    # PRNU map
    plt.subplot(1, 3, 3)
    prnu_display = plt.imshow(prnu_map, cmap='hot', vmin=0, vmax=0.1)
    plt.colorbar(prnu_display, label='PRNU (Noise/Signal)')
    plt.title("PRNU Map")
    
    plt.tight_layout()
    
    return stats, mean_flat, flat_normalized, plt

# =============================================
# SAVE RESULTS
# =============================================
def save_flat_results(stats, mean_flat, flat_norm, plots):
    """Save all outputs to results folder"""
    # Save statistics
    with open(os.path.join(results_dir, "flat_stats.txt"), 'w') as f:
        f.write("Flat Field Analysis Results\n")
        f.write("=========================\n")
        f.write(f"Average signal level: {stats['avg_signal_level']:.2f} DN\n")
        f.write(f"Max vignetting: {stats['max_vignetting']:.2f}%\n")
        f.write(f"Average PRNU: {stats['avg_prnu']:.2f}%\n")
    
    # Save images as TIFF
    io.imsave(os.path.join(results_dir, "mean_flat.tif"), mean_flat.astype(np.float32))
    io.imsave(os.path.join(results_dir, "flat_normalized.tif"), flat_norm.astype(np.float32))
    
    # Save plots
    plots.savefig(os.path.join(results_dir, "flat_analysis_plots.png"))
    
    print(f"\nResults saved to: {results_dir}")

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    try:
        print("=== Flat Field Analysis ===")
        
        # Step 1: Load frames
        flat_frames = load_flat_frames()
        
        # Step 2: Analyze
        stats, mean_flat, flat_norm, plt = analyze_flat_frames(flat_frames)
        
        # Step 3: Save results
        save_flat_results(stats, mean_flat, flat_norm, plt)
        
        print("\nAnalysis complete!")
        print("Key Results:")
        print(f"  Average signal level: {stats['avg_signal_level']:.2f} DN")
        print(f"  Max vignetting: {stats['max_vignetting']:.2f}%")
        print(f"  Average PRNU: {stats['avg_prnu']:.2f}%")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Troubleshooting:")
        print("1. Check files exist in flat_field/")
        print("2. Verify file extensions match (e.g., .tif, .png)")
        print(f"3. Current search pattern: '{flat_pattern}'")