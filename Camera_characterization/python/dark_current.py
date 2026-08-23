import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from glob import glob

# =============================================
# CONFIGURATION (Update these paths)
# =============================================
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "camera_characterization")
dark_pattern = "dark_exp_*.*"  # Will find .tif, .png, etc.
results_dir = os.path.join(base_path, "results", "dark_analysis")

# Create results directory
os.makedirs(results_dir, exist_ok=True)

# =============================================
# LOAD DARK FRAMES
# =============================================
def load_dark_frames():
    """Load all dark frames with validation"""
    dark_files = sorted(glob(os.path.join(base_path, "dark", dark_pattern)))
    
    if not dark_files:
        available_files = os.listdir(os.path.join(base_path, "dark"))
        raise FileNotFoundError(
            f"No dark frames found matching '{dark_pattern}'.\n"
            f"Files actually present: {available_files}"
        )
    
    print(f"Found {len(dark_files)} dark frames:")
    for f in dark_files[:3]:  # Show first 3 as sample
        print(f"  {os.path.basename(f)}")
    if len(dark_files) > 3:
        print(f"  (...and {len(dark_files)-3} more)")
    
    return np.array([io.imread(f) for f in dark_files])

# =============================================
# ANALYSIS FUNCTIONS
# =============================================
def analyze_dark_frames(frames):
    """Calculate statistics and generate plots"""
    print("\nCalculating statistics...")
    
    # Basic statistics
    mean_dark = np.mean(frames, axis=0)
    temporal_noise = np.std(frames, axis=0)
    
    # Save statistics
    stats = {
        "avg_read_noise": np.mean(temporal_noise),
        "max_read_noise": np.max(temporal_noise),
        "avg_dark_current": np.mean(mean_dark),
    }
    
    # Generate plots
    plt.figure(figsize=(12, 6))
    
    # Histogram of temporal noise
    plt.subplot(1, 2, 1)
    plt.hist(temporal_noise.flatten(), bins=50, alpha=0.7, color='blue')
    plt.xlabel("Noise (DN)")
    plt.ylabel("Frequency")
    plt.title("Temporal Noise Distribution")
    
    # Spatial noise map
    plt.subplot(1, 2, 2)
    noise_map = plt.imshow(temporal_noise, cmap='viridis')
    plt.colorbar(noise_map, label='Noise (DN)')
    plt.title("Spatial Noise Map")
    
    plt.tight_layout()
    
    return stats

# =============================================
# SAVE RESULTS
# =============================================
def save_results(stats, plots):
    """Save all outputs to results folder"""
    # Save statistics as text
    with open(os.path.join(results_dir, "dark_stats.txt"), 'w') as f:
        f.write("Dark Frame Analysis Results\n")
        f.write("==========================\n")
        for key, value in stats.items():
            f.write(f"{key:20}: {value:.4f}\n")
    
    # Save plots
    plots.savefig(os.path.join(results_dir, "dark_analysis_plots.png"))
    
    print(f"\nResults saved to: {results_dir}")

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    try:
        print("=== Dark Frame Analysis ===")
        
        # Step 1: Load frames
        dark_frames = load_dark_frames()
        
        # Step 2: Analyze
        stats = analyze_dark_frames(dark_frames)
        
        # Step 3: Save results
        save_results(stats, plt)
        
        print("\nAnalysis complete!")
        print("Key Results:")
        for k, v in stats.items():
            print(f"  {k:20}: {v:.4f}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check:")
        print("1. File paths are correct")
        print("2. Image files exist in the dark/ folder")
        print("3. File extensions match (e.g., .tif vs .png)")
        
        
