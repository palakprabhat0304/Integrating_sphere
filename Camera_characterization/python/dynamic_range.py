import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage import io
import json
from tqdm import tqdm

# Set paths
desktop_path = os.path.expanduser("~/Desktop")
data_folder = os.path.join(desktop_path, "camera_characterization")
dynamic_range_folder = os.path.join(data_folder, "dynamic_range")
results_folder = os.path.join(data_folder, "results", "dynamic_range")

# Create results directory if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

def find_dynamic_range_files():
    """Find all dynamic range files in the directory and determine exposure times"""
    files = os.listdir(dynamic_range_folder)
    dynamic_range_files = [f for f in files if f.startswith('dynamic_range_exp_')]
    
    if not dynamic_range_files:
        raise FileNotFoundError(f"No dynamic range files found in {dynamic_range_folder}")
    
    # Extract unique exposure times from filenames
    exposure_times = set()
    for f in dynamic_range_files:
        parts = f.split('_')
        try:
            exp_time = int(parts[3])  # dynamic_range_exp_XX_nX
            exposure_times.add(exp_time)
        except (IndexError, ValueError):
            continue
    
    if not exposure_times:
        raise ValueError("Could not determine exposure times from filenames")
    
    return sorted(exposure_times)

def load_dynamic_range_frames():
    """Load all dynamic range frames into memory"""
    exposure_times = find_dynamic_range_files()
    frames = {exp: [] for exp in exposure_times}
    
    print(f"Found exposure times: {exposure_times}")
    
    for exp in exposure_times:
        # Find all files for this exposure time
        pattern = f"dynamic_range_exp_{exp}_n"
        matching_files = [f for f in os.listdir(dynamic_range_folder) 
                         if f.startswith(pattern)]
        
        if not matching_files:
            print(f"Warning: No files found for exposure {exp}ms")
            continue
            
        for filename in matching_files:
            filepath = os.path.join(dynamic_range_folder, filename)
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

def analyze_dynamic_range(frames):
    """Perform dynamic range analysis on loaded frames"""
    results = {}
    
    for exp, img_list in frames.items():
        if not img_list:
            continue
            
        # Stack frames for statistics
        img_stack = np.stack(img_list)
        
        # Calculate statistics
        mean_frame = np.mean(img_stack, axis=0)
        std_frame = np.std(img_stack, axis=0)
        
        # Avoid division by zero for PRNU calculation
        std_frame[mean_frame == 0] = np.nan
        mean_frame[mean_frame == 0] = np.nan
        
        # Photon Transfer Curve (mean vs variance)
        valid_pixels = ~np.isnan(mean_frame)
        if valid_pixels.any():
            slope, intercept, r_value, _, _ = stats.linregress(
                mean_frame[valid_pixels].flatten(),
                (std_frame[valid_pixels]**2).flatten()
            )
        else:
            slope, intercept, r_value = np.nan, np.nan, np.nan
        
        results[exp] = {
            'mean': mean_frame,
            'std': std_frame,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'num_frames': len(img_list)
        }
    
    return results

def save_plots(results, save_path):
    """Create and save plots for dynamic range analysis"""
    if not results:
        print("Warning: No results to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean vs Variance (Photon Transfer Curve)
    plt.subplot(2, 2, 1)
    colors = ['b', 'r', 'g', 'm', 'c']
    for i, (exp, res) in enumerate(results.items()):
        valid_pixels = ~np.isnan(res['mean'])
        if valid_pixels.any():
            plt.scatter(
                res['mean'][valid_pixels].flatten(),
                (res['std'][valid_pixels]**2).flatten(),
                color=colors[i % len(colors)], alpha=0.1, s=1,
                label=f'Exp {exp}ms (K={res["slope"]:.2f}, R²={res["r_value"]**2:.2f})'
            )
    plt.xlabel('Mean Signal (DN)')
    plt.ylabel('Variance (DN²)')
    plt.title('Photon Transfer Curve')
    plt.grid(True)
    if results:
        plt.legend()
    
    # Plot 2: Signal-to-Noise Ratio
    plt.subplot(2, 2, 2)
    for i, (exp, res) in enumerate(results.items()):
        valid_pixels = ~np.isnan(res['mean'])
        if valid_pixels.any():
            snr = res['mean'][valid_pixels] / res['std'][valid_pixels]
            plt.scatter(
                res['mean'][valid_pixels].flatten(),
                snr.flatten(),
                color=colors[i % len(colors)], alpha=0.1, s=1,
                label=f'Exp {exp}ms'
            )
    plt.xlabel('Mean Signal (DN)')
    plt.ylabel('Signal-to-Noise Ratio')
    plt.title('Signal-to-Noise Ratio vs Signal')
    plt.grid(True)
    if results:
        plt.legend()
    
    # Plot 3: Histogram of pixel values
    plt.subplot(2, 2, 3)
    for i, (exp, res) in enumerate(results.items()):
        if ~np.isnan(res['mean']).any():
            plt.hist(
                res['mean'][~np.isnan(res['mean'])].flatten(),
                bins=256, range=(0, 2**16 if res['mean'].dtype == np.uint16 else 2**8),
                alpha=0.5, color=colors[i % len(colors)],
                label=f'Exp {exp}ms'
            )
    plt.xlabel('Pixel Value (DN)')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Distribution')
    plt.grid(True)
    if results:
        plt.legend()
    
    # Plot 4: Example frame (only if we have results)
    if results:
        plt.subplot(2, 2, 4)
        exp = list(results.keys())[0]
        plt.imshow(results[exp]['mean'], cmap='gray')
        plt.title(f'Mean Frame (Exp {exp}ms)')
        plt.colorbar(label='Pixel Value (DN)')
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'dynamic_range_analysis.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plots to: {plot_path}")

def calculate_dynamic_range(results):
    """Calculate dynamic range metrics"""
    dr_results = {}
    
    for exp, res in results.items():
        try:
            # Full well capacity (maximum signal before saturation)
            max_signal = np.nanmax(res['mean'])
            
            # Read noise (from intercept of PTC)
            read_noise = np.sqrt(abs(res['intercept']))  # Take abs to avoid sqrt of negative
            
            # Dynamic range calculation
            dynamic_range = max_signal / read_noise if read_noise > 0 else np.nan
            
            dr_results[exp] = {
                'max_signal': max_signal,
                'read_noise': read_noise,
                'dynamic_range': dynamic_range,
                'dynamic_range_dB': 20 * np.log10(dynamic_range) if dynamic_range > 0 else np.nan,
                'conversion_gain': res['slope'],
                'r_squared': res['r_value']**2
            }
        except Exception as e:
            print(f"Error calculating metrics for exposure {exp}ms: {str(e)}")
            dr_results[exp] = {
                'max_signal': np.nan,
                'read_noise': np.nan,
                'dynamic_range': np.nan,
                'dynamic_range_dB': np.nan,
                'conversion_gain': np.nan,
                'r_squared': np.nan
            }
    
    return dr_results

def save_metrics(dr_results, save_path):
    """Save the calculated dynamic range metrics to JSON"""
    metrics_file = os.path.join(save_path, 'dynamic_range_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(dr_results, f, indent=4)
    
    # Also save a human-readable text version
    txt_file = os.path.join(save_path, 'dynamic_range_metrics.txt')
    with open(txt_file, 'w') as f:
        f.write("Dynamic Range Metrics:\n")
        f.write("=" * 40 + "\n")
        for exp, res in dr_results.items():
            f.write(f"Exposure: {exp}ms\n")
            f.write(f"- Max Signal: {res['max_signal']:.2f} DN\n")
            f.write(f"- Read Noise: {res['read_noise']:.2f} DN\n")
            f.write(f"- Dynamic Range: {res['dynamic_range']:.2f}:1\n")
            f.write(f"- Dynamic Range: {res['dynamic_range_dB']:.2f} dB\n")
            f.write(f"- Conversion Gain: {res['conversion_gain']:.4f} DN/e-\n")
            f.write(f"- R-squared: {res['r_squared']:.4f}\n")
            f.write("-" * 40 + "\n")
    
    print(f"Saved metrics to: {metrics_file} and {txt_file}")

def save_mean_frames(results, save_path):
    """Save the mean frames as TIFF files"""
    for exp, res in results.items():
        output_file = os.path.join(save_path, f'mean_frame_exp_{exp}ms.tif')
        try:
            io.imsave(output_file, res['mean'].astype(np.float32))
            print(f"Saved mean frame to: {output_file}")
        except Exception as e:
            print(f"Error saving mean frame for exposure {exp}ms: {str(e)}")

def main():
    try:
        print("Loading dynamic range frames...")
        frames = load_dynamic_range_frames()
        
        print("\nAnalyzing dynamic range...")
        results = analyze_dynamic_range(frames)
        
        print("\nCalculating dynamic range metrics...")
        dr_results = calculate_dynamic_range(results)
        
        print("\nSaving results...")
        save_plots(results, results_folder)
        save_metrics(dr_results, results_folder)
        save_mean_frames(results, results_folder)
        
        print(f"\nAnalysis complete! Results saved to: {results_folder}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check:")
        print(f"1. The folder exists: {dynamic_range_folder}")
        print("2. Files are named like: dynamic_range_exp_XX_nX.tif (or .png)")
        print("3. You have permission to access these files")

if __name__ == "__main__":
    main()