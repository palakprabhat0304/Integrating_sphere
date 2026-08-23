import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import json
from tqdm import tqdm

# Set paths
desktop_path = os.path.expanduser("~/Desktop")
data_folder = os.path.join(desktop_path, "camera_characterization")
fpn_folder = os.path.join(data_folder, "FPN")
results_folder = os.path.join(data_folder, "results", "FPN")

# Create results directory if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

def find_fpn_files():
    """Find all FPN files in the directory and determine exposure times"""
    files = os.listdir(fpn_folder)
    fpn_files = [f for f in files if f.startswith('FPN_exp_')]
    
    if not fpn_files:
        raise FileNotFoundError(f"No FPN files found in {fpn_folder}")
    
    # Extract unique exposure times from filenames
    exposure_times = set()
    for f in fpn_files:
        parts = f.split('_')
        try:
            exp_time = int(parts[2])  # FPN_exp_XX_nX
            exposure_times.add(exp_time)
        except (IndexError, ValueError):
            continue
    
    if not exposure_times:
        raise ValueError("Could not determine exposure times from filenames")
    
    return sorted(exposure_times)

def load_fpn_frames():
    """Load all FPN frames into memory"""
    exposure_times = find_fpn_files()
    frames = {exp: [] for exp in exposure_times}
    
    print(f"Found exposure times: {exposure_times}")
    
    for exp in exposure_times:
        # Find all files for this exposure time
        pattern = f"FPN_exp_{exp}_n"
        matching_files = [f for f in os.listdir(fpn_folder) 
                         if f.startswith(pattern)]
        
        if not matching_files:
            print(f"Warning: No files found for exposure {exp}ms")
            continue
            
        for filename in matching_files:
            filepath = os.path.join(fpn_folder, filename)
            try:
                img = io.imread(filepath)
                frames[exp].append(img)
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    # Remove empty exposure sets
    frames = {exp: imgs for exp, imgs in frames.items() if imgs}
    
    if not frames:
        raise ValueError("No valid frames loaded - check your files")
    
    return frames

def analyze_fpn(frames):
    """Perform FPN analysis on loaded frames"""
    results = {}
    
    for exp, img_list in frames.items():
        if not img_list:
            continue
            
        # Stack frames for statistics
        img_stack = np.stack(img_list)
        
        # Calculate statistics
        mean_frame = np.mean(img_stack, axis=0)
        std_frame = np.std(img_stack, axis=0)
        
        # Calculate temporal noise (average of pixel-wise std)
        temporal_noise = np.nanmean(std_frame)
        
        # Calculate FPN (std of mean frame)
        fpn = np.nanstd(mean_frame)
        
        # Calculate PRNU (FPN normalized by mean signal)
        prnu = fpn / np.nanmean(mean_frame) if np.nanmean(mean_frame) > 0 else np.nan
        
        results[exp] = {
            'mean': mean_frame,
            'std': std_frame,
            'temporal_noise': temporal_noise,
            'fpn': fpn,
            'prnu': prnu,
            'num_frames': len(img_list)
        }
    
    return results

def save_fpn_plots(results, save_path):
    """Create and save plots for FPN analysis"""
    if not results:
        print("Warning: No results to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean frame with FPN
    plt.subplot(2, 2, 1)
    exp = list(results.keys())[0]
    plt.imshow(results[exp]['mean'], cmap='gray')
    plt.title(f'Mean Frame (Exp {exp}ms)\nFPN: {results[exp]["fpn"]:.2f} DN')
    plt.colorbar(label='Pixel Value (DN)')
    
    # Plot 2: Standard deviation frame (temporal noise)
    plt.subplot(2, 2, 2)
    plt.imshow(results[exp]['std'], cmap='hot')
    plt.title(f'Std Dev Frame (Exp {exp}ms)\nTemporal Noise: {results[exp]["temporal_noise"]:.2f} DN')
    plt.colorbar(label='Noise (DN)')
    
    # Plot 3: Histogram of pixel values showing FPN
    plt.subplot(2, 2, 3)
    plt.hist(results[exp]['mean'].flatten(), bins=256, 
             range=(0, 2**16 if results[exp]['mean'].dtype == np.uint16 else 2**8))
    plt.xlabel('Pixel Value (DN)')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Distribution (FPN)')
    plt.grid(True)
    
    # Plot 4: PRNU visualization
    plt.subplot(2, 2, 4)
    prnu_image = results[exp]['std'] / results[exp]['mean']
    prnu_image[~np.isfinite(prnu_image)] = 0  # Handle divide by zero
    plt.imshow(prnu_image, cmap='jet', vmin=0, vmax=0.1)
    plt.title(f'PRNU Map (Exp {exp}ms)\nPRNU: {results[exp]["prnu"]*100:.2f}%')
    plt.colorbar(label='PRNU')
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'fpn_analysis.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved FPN plots to: {plot_path}")

def save_fpn_metrics(results, save_path):
    """Save the calculated FPN metrics to JSON"""
    metrics = {}
    for exp, res in results.items():
        metrics[exp] = {
            'temporal_noise': res['temporal_noise'],
            'fpn': res['fpn'],
            'prnu': res['prnu'],
            'num_frames': res['num_frames']
        }
    
    metrics_file = os.path.join(save_path, 'fpn_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Also save a human-readable text version
    txt_file = os.path.join(save_path, 'fpn_metrics.txt')
    with open(txt_file, 'w') as f:
        f.write("Fixed Pattern Noise Metrics:\n")
        f.write("=" * 40 + "\n")
        for exp, res in metrics.items():
            f.write(f"Exposure: {exp}ms\n")
            f.write(f"- Temporal Noise: {res['temporal_noise']:.2f} DN\n")
            f.write(f"- Fixed Pattern Noise: {res['fpn']:.2f} DN\n")
            f.write(f"- PRNU: {res['prnu']*100:.2f}%\n")
            f.write(f"- Number of frames: {res['num_frames']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"Saved FPN metrics to: {metrics_file} and {txt_file}")

def save_fpn_frames(results, save_path):
    """Save the analysis frames as TIFF files"""
    for exp, res in results.items():
        # Save mean frame
        mean_file = os.path.join(save_path, f'fpn_mean_exp_{exp}ms.tif')
        io.imsave(mean_file, res['mean'].astype(np.float32))
        
        # Save std frame
        std_file = os.path.join(save_path, f'fpn_std_exp_{exp}ms.tif')
        io.imsave(std_file, res['std'].astype(np.float32))
        
        # Save PRNU frame
        prnu_file = os.path.join(save_path, f'fpn_prnu_exp_{exp}ms.tif')
        prnu_image = res['std'] / res['mean']
        prnu_image[~np.isfinite(prnu_image)] = 0
        io.imsave(prnu_file, prnu_image.astype(np.float32))
        
    print(f"Saved FPN frames to: {save_path}")

def main():
    try:
        print("Loading FPN frames...")
        frames = load_fpn_frames()
        
        print("\nAnalyzing FPN...")
        results = analyze_fpn(frames)
        
        print("\nSaving results...")
        save_fpn_plots(results, results_folder)
        save_fpn_metrics(results, results_folder)
        save_fpn_frames(results, results_folder)
        
        print(f"\nFPN analysis complete! Results saved to: {results_folder}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check:")
        print(f"1. The folder exists: {fpn_folder}")
        print("2. Files are named like: FPN_exp_XX_nX.tif (or .png)")
        print("3. You have permission to access these files")

if __name__ == "__main__":
    main()