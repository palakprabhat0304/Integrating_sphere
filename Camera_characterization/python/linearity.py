import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
import re
from scipy.stats import linregress

def parse_exposure_from_filename(filename):
    """Extract exposure time from filename"""
    match = re.search(r'exp_(\d+)_n\d+', filename)
    if match:
        return int(match.group(1))
    print(f"⚠️ Warning: Could not parse exposure time from filename: {filename}")
    return None

def load_and_average_frames(base_path):
    """Load and average FITS frames grouped by exposure"""
    print("\n" + "="*50)
    print("LOADING AND AVERAGING FRAMES")
    print("="*50)
    
    files = glob(os.path.join(base_path, '*.fits'))
    if not files:
        raise FileNotFoundError(f"❌ ERROR: No FITS files found in {base_path}")
    print(f"Found {len(files)} FITS files")
    
    exposure_groups = {}
    for file in files:
        exp_time = parse_exposure_from_filename(os.path.basename(file))
        if exp_time is not None:
            exposure_groups.setdefault(exp_time, []).append(file)
    
    if not exposure_groups:
        raise ValueError("❌ ERROR: No valid exposure groups found. Check filename patterns.")
    
    print(f"\nFound {len(exposure_groups)} unique exposure times")
    
    averaged_frames = {}
    for exp_time, file_list in exposure_groups.items():
        print(f"\nProcessing exposure: {exp_time} µs ({len(file_list)} files)")
        frames = []
        for i, file in enumerate(file_list[:5]):
            try:
                with fits.open(file) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    frames.append(data)
            except Exception as e:
                print(f"❌ Error loading {file}: {str(e)}")
                continue
        
        if frames:
            avg_frame = np.mean(frames, axis=0)
            averaged_frames[exp_time] = avg_frame
            print(f"  Averaged mean signal: {np.mean(avg_frame):.1f}")
        else:
            print(f"❌ Skipping exposure {exp_time}: No valid frames")
    
    return averaged_frames

def analyze_linearity(averaged_frames, output_dir, saturation_threshold=15000):
    """Analyze camera response linearity and plot results"""
    if not averaged_frames:
        raise ValueError("❌ ERROR: No averaged frames for analysis")

    exposure_times = sorted(averaged_frames.keys())
    mean_signals = [np.mean(averaged_frames[exp]) for exp in exposure_times]

    print("\n" + "="*50)
    print("EXPOSURE vs MEAN SIGNAL")
    print("="*50)
    for exp, sig in zip(exposure_times, mean_signals):
        print(f"{exp:6d} µs --> {sig:8.2f} ADU")

    exposure_arr = np.array(exposure_times)
    signal_arr = np.array(mean_signals)

    # ==== Linear Fit on Linear Region Only ====
    mask_linear = exposure_arr <= saturation_threshold
    fit_exposure = exposure_arr[mask_linear]
    fit_signal = signal_arr[mask_linear]

    slope, intercept, r_value, _, _ = linregress(fit_exposure, fit_signal)
    linear_fit_full = slope * exposure_arr + intercept
    linear_fit_sub = slope * fit_exposure + intercept

    # ==== Deviation from Linearity ====
    deviation = (signal_arr - (slope * exposure_arr + intercept)) / (slope * exposure_arr + intercept) * 100

    # ==== Plotting ====
    plt.figure(figsize=(12, 5))

    # Plot 1: Linearity
    plt.subplot(1, 2, 1)
    plt.plot(exposure_arr, signal_arr, 'bo', label='Measured Data')
    plt.plot(exposure_arr, linear_fit_full, 'r-', label=f'Linear Fit (≤ {saturation_threshold}): y = {slope:.6f}x + {intercept:.2f}')
    plt.xlabel('Exposure Time (µs)')
    plt.ylabel('Mean Signal (ADU)')
    plt.title('Camera Response Linearity')
    plt.grid(True)
    plt.legend()

    # Plot 2: Deviation
    plt.subplot(1, 2, 2)
    plt.plot(exposure_arr, deviation, 'go-')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Exposure Time (µs)')
    plt.ylabel('Deviation from Linearity (%)')
    plt.title('Linearity Deviation')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'linearity_analysis.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"\n✅ Plot saved to: {plot_path}")
    print("\n" + "="*50)
    print("LINEARITY FIT SUMMARY")
    print("="*50)
    print(f"Linear Fit Range: Exposure ≤ {saturation_threshold} µs")
    print(f"Linear Fit: Signal = {slope:.6f} * Exposure + {intercept:.2f}")
    print(f"R-squared: {r_value**2:.6f}")
    print(f"Max deviation: {np.max(np.abs(deviation)):.2f}%")

    return exposure_arr, signal_arr, deviation

def main():
    base_path = os.path.expanduser('~/Desktop/camera_characterization/linearity')
    output_dir = os.path.expanduser('~/Desktop/camera_characterization/linearity_results')
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("CAMERA LINEARITY ANALYSIS - IMPROVED FIT LINE")
    print("="*70)

    try:
        averaged_frames = load_and_average_frames(base_path)
        if not averaged_frames:
            print("❌ No valid frames for analysis")
            return

        exposure_times, mean_signals, deviation = analyze_linearity(averaged_frames, output_dir)

        # Save results
        result_path = os.path.join(output_dir, 'linearity_results.csv')
        with open(result_path, 'w') as f:
            f.write("Exposure,Mean_Signal,Deviation_Pct\n")
            for exp, sig, dev in zip(exposure_times, mean_signals, deviation):
                f.write(f"{exp},{sig},{dev:.4f}\n")
        print(f"\n✅ Results saved to: {result_path}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == '__main__':
    main()
