#!/usr/bin/env python3
"""
Fast PSF Injection and Visualization Script

This script:
1. Loads a PSF from psf_array.npy
2. Calculates FWHM from the PSF
3. Injects PSFs at random non-peak locations with different SNR levels
4. Saves before/after PNG images with injection markers
5. Saves transient-added FITS files
6. Saves injection locations as CSV with RA/DEC coordinates

Requirements:
- astropy
- numpy
- photutils
- scipy
- pandas
- matplotlib
- pillow

Install with: pip install astropy numpy photutils scipy pandas matplotlib pillow
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.ndimage import maximum_filter, generate_binary_structure
from PIL import Image
import warnings
import random

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

def calculate_fwhm_from_psf(psf_array):
    """Calculate FWHM from PSF array using radial profile."""
    center_y, center_x = np.unravel_index(np.argmax(psf_array), psf_array.shape)
    
    y, x = np.ogrid[:psf_array.shape[0], :psf_array.shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create 1D radial profile
    r_int = r.astype(int)
    max_r = min(int(np.max(r)), 30)  # Limit to reasonable radius
    
    radial_profile = np.zeros(max_r + 1)
    
    for i in range(max_r + 1):
        mask = r_int == i
        if np.any(mask):
            radial_profile[i] = np.mean(psf_array[mask])
    
    # Find FWHM
    max_val = np.max(radial_profile)
    half_max = max_val / 2.0
    
    try:
        crossing_idx = np.where(radial_profile <= half_max)[0]
        if len(crossing_idx) > 0:
            fwhm = 2 * crossing_idx[0]
        else:
            fwhm = 4.0
    except:
        fwhm = 4.0
    
    return max(fwhm, 2.0)

def normalize_array(data):
    """Normalize array between 0 and 1."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min)

def normalize_with_zscale(data):
    """Normalize data using ZScale."""
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data[valid_mask])
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    normalized[~valid_mask] = 0
    return normalized

def find_peaks_simple(data, threshold_percentile=95, min_distance=64):
    """Simple peak finding for exclusion zones."""
    threshold = np.percentile(data, threshold_percentile)
    
    neighborhood = generate_binary_structure(2, 2)
    local_maxima = maximum_filter(data, footprint=neighborhood) == data
    peaks_mask = local_maxima & (data > threshold)
    peak_coords = np.column_stack(np.where(peaks_mask))
    
    # Quick distance filtering
    filtered_peaks = []
    for peak in peak_coords:
        if len(filtered_peaks) == 0:
            filtered_peaks.append(tuple(peak))
        else:
            min_dist = min([np.sqrt((peak[0] - p[0])**2 + (peak[1] - p[1])**2) 
                           for p in filtered_peaks])
            if min_dist >= min_distance:
                filtered_peaks.append(tuple(peak))
        
        if len(filtered_peaks) >= 50:  # Limit for speed
            break
    
    return filtered_peaks

def check_valid_region(data, center_y, center_x, region_size=64):
    """
    Check if a 64x64 region around the center has 95% valid (non-zero, non-NaN) pixels.
    """
    half_size = region_size // 2
    
    # Define region bounds
    y_start = max(0, center_y - half_size)
    y_end = min(data.shape[0], center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(data.shape[1], center_x + half_size)
    
    # Extract region
    region = data[y_start:y_end, x_start:x_end]
    
    # Check for valid pixels (not zero, not NaN, not infinite)
    valid_mask = np.isfinite(region) & (region != 0)
    valid_fraction = np.sum(valid_mask) / region.size
    
    return valid_fraction >= 0.95

def generate_injection_positions(data, psf_shape, existing_peaks, n_positions=20):
    """Generate injection positions avoiding peaks and invalid regions."""
    positions = []
    margin = max(psf_shape) // 2 + 32  # Extra margin for 64x64 check
    y_min, y_max = margin, data.shape[0] - margin
    x_min, x_max = margin, data.shape[1] - margin
    
    attempts = 0
    max_attempts = 20000  # Increased attempts
    
    print(f"    Searching for {n_positions} valid injection positions...")
    
    while len(positions) < n_positions and attempts < max_attempts:
        attempts += 1
        y = random.randint(y_min, y_max)
        x = random.randint(x_min, x_max)
        
        # Check if 64x64 region around this position is valid
        if not check_valid_region(data, y, x, region_size=64):
            continue
        
        # Check distance from existing peaks (minimum 80 pixels)
        too_close = any(np.sqrt((y - py)**2 + (x - px)**2) < 80 
                       for py, px in existing_peaks)
        
        # Check distance from other injections (minimum 50 pixels)
        if not too_close:
            too_close = any(np.sqrt((y - iy)**2 + (x - ix)**2) < 50 
                           for iy, ix in positions)
        
        if not too_close:
            positions.append((y, x))
            print(f"    Found position {len(positions)}: ({y}, {x}) after {attempts} attempts")
    
    if len(positions) < n_positions:
        print(f"    Warning: Only found {len(positions)} valid positions out of {n_positions} requested")
    
    return positions

def estimate_local_noise(image, position, aperture_radius=10):
    """Estimate local noise around a position using a circular annulus."""
    center_y, center_x = position
    
    try:
        # Create annulus for background estimation
        annulus = CircularAnnulus((center_x, center_y), 
                                r_in=aperture_radius * 2, 
                                r_out=aperture_radius * 4)
        
        annulus_mask = annulus.to_mask()
        annulus_data = annulus_mask.multiply(image)
        
        if annulus_data is not None:
            annulus_data_1d = annulus_data[annulus_mask.data > 0]
            # Remove outliers for better noise estimation
            valid_data = annulus_data_1d[np.isfinite(annulus_data_1d)]
            if len(valid_data) > 10:
                # Use sigma clipping to remove outliers
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                mask = np.abs(valid_data - mean_val) < 3 * std_val
                if np.sum(mask) > 5:
                    return np.std(valid_data[mask])
                else:
                    return std_val
            else:
                return 1.0  # Fallback
        else:
            return 1.0  # Fallback
    except:
        return 1.0  # Fallback

def calculate_required_scale_factor(target_snr, psf_array, noise_std, fwhm):
    """Calculate the scale factor needed to achieve target SNR."""
    # Calculate aperture area for photometry
    aperture_radius = 0.6731 * fwhm
    aperture_area = np.pi * aperture_radius**2
    
    # Estimate PSF flux in aperture (approximate)
    psf_center_y, psf_center_x = np.array(psf_array.shape) // 2
    
    # Create circular mask for PSF flux estimation
    y, x = np.ogrid[:psf_array.shape[0], :psf_array.shape[1]]
    mask = (x - psf_center_x)**2 + (y - psf_center_y)**2 <= aperture_radius**2
    
    # Sum PSF flux in aperture
    psf_flux_fraction = np.sum(psf_array[mask])
    
    if psf_flux_fraction <= 0:
        return max(1.0, target_snr * noise_std * 10)  # Fallback with minimum 1.0
    
    # Approximate solution (ignoring signal in noise term for simplicity)
    # SNR ≈ Signal / (Background_noise * sqrt(Area))
    background_noise = noise_std * np.sqrt(aperture_area)
    
    # Handle case where noise is very small or zero
    if background_noise <= 0:
        background_noise = 1e-10  # Small positive value to avoid division by zero
    
    required_signal = target_snr * background_noise
    scale_factor = required_signal / psf_flux_fraction
    
    # Ensure minimum scale factor of 1.0
    return max(1.0, scale_factor)

def iteratively_scale_for_target_snr(image, psf, position, target_snr, fwhm, max_iterations=10):
    """Iteratively adjust PSF scaling to achieve target SNR."""
    center_y, center_x = position
    psf_half_y, psf_half_x = psf.shape[0] // 2, psf.shape[1] // 2
    
    # Initial estimate
    noise_std = estimate_local_noise(image, position)
    scale_factor = calculate_required_scale_factor(target_snr, psf, noise_std, fwhm)
    
    # Ensure minimum scale factor
    scale_factor = max(1.0, scale_factor)
    
    for iteration in range(max_iterations):
        # Create temporary image with current scaling
        temp_image = image.copy()
        
        # Calculate bounds for injection
        img_y_start = max(0, center_y - psf_half_y)
        img_y_end = min(image.shape[0], center_y + psf_half_y)
        img_x_start = max(0, center_x - psf_half_x)
        img_x_end = min(image.shape[1], center_x + psf_half_x)
        
        psf_y_start = max(0, psf_half_y - center_y)
        psf_y_end = psf_y_start + (img_y_end - img_y_start)
        psf_x_start = max(0, psf_half_x - center_x)
        psf_x_end = psf_x_start + (img_x_end - img_x_start)
        
        # Inject scaled PSF
        temp_image[img_y_start:img_y_end, img_x_start:img_x_end] += \
            psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end] * scale_factor
        
        # Measure actual SNR
        actual_snr = calculate_snr_photutils(temp_image, (center_x, center_y), fwhm)
        
        # Check if we're close enough (within 20% tolerance)
        if abs(actual_snr - target_snr) / target_snr < 0.2:
            return scale_factor, actual_snr
        
        # Adjust scale factor based on ratio
        if actual_snr > 0:
            adjustment_ratio = target_snr / actual_snr
            # Limit adjustment to prevent overshooting
            adjustment_ratio = np.clip(adjustment_ratio, 0.5, 2.0)
            scale_factor *= adjustment_ratio
            scale_factor = max(1.0, scale_factor)  # Maintain minimum
        else:
            # If SNR is 0 or negative, increase scale factor significantly
            scale_factor *= 2.0
        
        # Prevent infinite scaling
        if scale_factor > 1000:
            break
    
    return scale_factor, actual_snr

def inject_multiple_psfs_iterative(image, psf, positions, target_snrs, fwhm):
    """Inject multiple PSFs with iterative scaling to achieve target SNRs."""
    modified_image = image.copy()
    final_scale_factors = []
    final_snrs = []
    successful_positions = []  # Track positions where injection actually succeeded
    
    print(f"    Iteratively scaling PSFs for accurate SNRs...")
    
    for i, (center_y, center_x) in enumerate(positions):
        if i >= len(target_snrs):
            break
        
        # Find optimal scale factor for this position
        scale_factor, achieved_snr = iteratively_scale_for_target_snr(
            modified_image, psf, (center_y, center_x), target_snrs[i], fwhm
        )
        
        # Only proceed if we achieved a reasonable SNR (> 0.5)
        if achieved_snr > 0.5:
            # Apply the final injection to the cumulative image
            psf_half_y, psf_half_x = psf.shape[0] // 2, psf.shape[1] // 2
            
            img_y_start = max(0, center_y - psf_half_y)
            img_y_end = min(image.shape[0], center_y + psf_half_y)
            img_x_start = max(0, center_x - psf_half_x)
            img_x_end = min(image.shape[1], center_x + psf_half_x)
            
            psf_y_start = max(0, psf_half_y - center_y)
            psf_y_end = psf_y_start + (img_y_end - img_y_start)
            psf_x_start = max(0, psf_half_x - center_x)
            psf_x_end = psf_x_start + (img_x_end - img_x_start)
            
            modified_image[img_y_start:img_y_end, img_x_start:img_x_end] += \
                psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end] * scale_factor
            
            final_scale_factors.append(scale_factor)
            final_snrs.append(achieved_snr)
            successful_positions.append((center_y, center_x))
            
            print(f"      Position {len(successful_positions)}: Target={target_snrs[i]:.2f}, Achieved={achieved_snr:.2f}, Scale={scale_factor:.2f}")
        else:
            print(f"      Position {i+1}: SKIPPED - Failed to achieve reasonable SNR (got {achieved_snr:.2f})")
    
    print(f"    Successfully injected {len(successful_positions)} out of {len(positions)} attempted positions")
    return modified_image, final_scale_factors, final_snrs, successful_positions

def calculate_snr_photutils(image, center_position, fwhm=2.17):
    """Calculate SNR using photutils."""
    aperture_radius = 0.6731 * fwhm
    aperture = CircularAperture(center_position, r=aperture_radius)
    annulus = CircularAnnulus(center_position,
                             r_in=aperture_radius * 1.5,
                             r_out=aperture_radius * 2.5)
    
    try:
        phot_table = aperture_photometry(image, aperture)
        bkg_table = aperture_photometry(image, annulus)
        
        annulus_area = annulus.area
        aperture_area = aperture.area
        bkg_mean = bkg_table['aperture_sum'][0] / annulus_area
        total_bkg = bkg_mean * aperture_area
        source_flux = phot_table['aperture_sum'][0] - total_bkg
        
        annulus_mask = annulus.to_mask()
        annulus_data = annulus_mask.multiply(image)
        if annulus_data is not None:
            annulus_data_1d = annulus_data[annulus_mask.data > 0]
            bkg_std = np.std(annulus_data_1d)
        else:
            return 0
        
        noise = np.sqrt(abs(source_flux) + (aperture_area * bkg_std**2))
        snr = source_flux / noise if noise > 0 else 0
        return max(snr, 0)
    except:
        return 0

def pixel_to_radec(x, y, header):
    """Convert pixel to RA/DEC coordinates."""
    try:
        wcs = WCS(header)
        sky_coord = wcs.pixel_to_world(x, y)
        ra_str = sky_coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)
        dec_str = sky_coord.dec.to_string(unit=u.deg, sep=':', precision=2)
        return ra_str, dec_str
    except:
        return f"{x:.2f}", f"{y:.2f}"

def create_visualization_png(data, positions, filepath, title, circle_color='red', 
                           circle_radius=15):
    """Create PNG visualization with circles marking positions."""
    # Normalize data using ZScale
    normalized_data = normalize_with_zscale(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(normalized_data, cmap='gray', origin='lower')
    
    # Add circles at injection positions
    for y, x in positions:
        circle = patches.Circle((x, y), circle_radius, linewidth=2, 
                              edgecolor=circle_color, facecolor='none')
        ax.add_patch(circle)
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def process_single_fits_file(fits_path, psf_array, psf_fwhm, output_dir):
    """Process a single FITS file with PSF injection."""
    fits_path = Path(fits_path)
    tile_name = fits_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing {tile_name}...")
    
    # Read FITS file - FIXED to handle extension HDUs
    original_data = None
    header = None
    
    try:
        with fits.open(fits_path) as hdul:
            # Print HDU info for debugging
            print(f"  Number of HDUs: {len(hdul)}")
            for i, hdu in enumerate(hdul):
                print(f"    HDU {i}: {type(hdu).__name__}, shape: {hdu.data.shape if hdu.data is not None else 'None'}")
            
            # Find the image data in extension HDUs
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    original_data = hdu.data.copy()  # Copy data before file closes
                    header = hdu.header.copy()  # Copy header too
                    print(f"  Found image data in HDU {i}, shape: {original_data.shape}")
                    break
            
            if original_data is None:
                print(f"  Error: No valid 2D image data found in {tile_name}")
                return []
    
    except Exception as e:
        print(f"  Error reading FITS file: {e}")
        return []
    
    # Handle data dimensions (keep this as backup)
    if original_data.ndim > 2:
        original_data = original_data[0] if original_data.ndim == 3 else original_data.squeeze()
    
    # Clean the data
    original_data = np.nan_to_num(original_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Image shape: {original_data.shape}")
    print(f"  Data range: min={np.min(original_data):.4f}, max={np.max(original_data):.4f}")
    
    # Find existing peaks to avoid
    existing_peaks = find_peaks_simple(original_data)
    print(f"  Found {len(existing_peaks)} existing peaks")
    
    # Generate injection positions - only valid ones
    injection_positions = generate_injection_positions(
        original_data, psf_array.shape, existing_peaks, n_positions=500
    )
    
    if len(injection_positions) == 0:
        print(f"  Warning: No valid injection positions found for {tile_name}")
        return []
    
    print(f"  Successfully found {len(injection_positions)} valid injection positions")
    
    # Define target SNRs only for positions we actually have
    target_snrs = np.linspace(3.0, 10.0, len(injection_positions))
    
    # Inject PSFs with iterative scaling
    print("  Injecting PSFs with iterative scaling...")
    try:
        modified_data, scale_factors, achieved_snrs, successful_positions = inject_multiple_psfs_iterative(
            original_data, psf_array, injection_positions, target_snrs, psf_fwhm
        )
    except ValueError as e:
        print(f"  Error during injection: {e}")
        return []
    
    # Now create both images using only successful positions
    before_png = output_dir / f"before_{tile_name}.png"
    create_visualization_png(original_data, successful_positions, before_png, 
                           f"Before injection - {tile_name} ({len(successful_positions)} valid positions)", 'blue')
    
    # Create after image with red circles - show the same successful positions
    after_png = output_dir / f"after_{tile_name}.png"
    create_visualization_png(modified_data, successful_positions, after_png,
                           f"After injection - {tile_name} ({len(successful_positions)} injections)", 'red')
    # Save 64x64 cutouts (z-scaled and normalized)
    save_cutouts(modified_data, successful_positions, output_dir, tile_name)
    
    # Save transient-added FITS file
    transient_fits = output_dir / f"transient_added_{tile_name}.fits"
    hdu = fits.PrimaryHDU(data=modified_data, header=header)
    hdu.writeto(transient_fits, overwrite=True)
    
    print(f"  Validating final SNRs...")
    # Calculate final SNRs for validation - ONLY for successful injections
    results = []
    for i, (pos_y, pos_x) in enumerate(successful_positions):
        if i >= len(achieved_snrs):
            break
        
        # Use the achieved SNR from iterative process
        final_snr = achieved_snrs[i]
        
        # Also verify with independent photometry measurement
        verification_snr = calculate_snr_photutils(
            modified_data, (pos_x, pos_y), psf_fwhm
        )
        
        ra_str, dec_str = pixel_to_radec(pos_x, pos_y, header)
        
        # Calculate corresponding target SNR (need to map back to original target)
        original_index = injection_positions.index((pos_y, pos_x))
        corresponding_target_snr = target_snrs[original_index]
        
        results.append({
            'tile_name': tile_name,
            'injection_id': i,
            'pixel_x': pos_x,
            'pixel_y': pos_y,
            'ra_sexagesimal': ra_str,
            'dec_sexagesimal': dec_str,
            'target_snr': corresponding_target_snr,
            'achieved_snr': final_snr,
            'verification_snr': verification_snr,
            'scale_factor': scale_factors[i],
            'psf_fwhm': psf_fwhm
        })
        
        print(f"    Final validation {i+1}: Target={corresponding_target_snr:.2f}, Achieved={final_snr:.2f}, Verified={verification_snr:.2f}")
    
    print(f"  Saved: {before_png.name}, {after_png.name}, {transient_fits.name}")
    print(f"  Successfully injected {len(successful_positions)} transients")
    return results

def save_cutouts(image, positions, output_base_dir, tile_name):
    """
    Extract 64x64 cutouts around positions, z-scale, normalize to [0,1],
    and save as PNG and FITS in output_psf_added/positives/
    """
    cutout_size = 64
    half_size = cutout_size // 2
    
    # Create output directories
    png_dir = Path(output_base_dir) / "output_psf_added" / "positives" / "png"
    fits_dir = Path(output_base_dir) / "output_psf_added" / "positives" / "fits"
    png_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving {len(positions)} cutouts...")
    
    for i, (pos_y, pos_x) in enumerate(positions):
        # Extract 64x64 cutout
        y_start = pos_y - half_size
        y_end = pos_y + half_size
        x_start = pos_x - half_size
        x_end = pos_x + half_size
        
        # Skip if cutout would go out of bounds
        if (y_start < 0 or y_end > image.shape[0] or 
            x_start < 0 or x_end > image.shape[1]):
            continue
        
        cutout = image[y_start:y_end, x_start:x_end].copy()
        
        # Z-scale the cutout
        zscale = ZScaleInterval()
        valid_mask = np.isfinite(cutout)
        if np.any(valid_mask):
            vmin, vmax = zscale.get_limits(cutout[valid_mask])
        else:
            vmin, vmax = 0, 1
        
        zscaled_cutout = np.clip((cutout - vmin) / (vmax - vmin + 1e-10), 0, 1)
        zscaled_cutout[~valid_mask] = 0
        
        # Save as PNG
        png_filename = png_dir / f"{tile_name}_cutout_{i:04d}.png"
        # Convert to 8-bit for PNG
        png_data = (zscaled_cutout * 255).astype(np.uint8)
        Image.fromarray(png_data).save(png_filename)
        
        # Save as FITS (normalized float data)
        fits_filename = fits_dir / f"{tile_name}_cutout_{i:04d}.fits"
        fits.PrimaryHDU(data=zscaled_cutout.astype(np.float32)).writeto(
            fits_filename, overwrite=True
        )
    
    print(f"    Saved cutouts to {png_dir} and {fits_dir}")
def main():
    """Main processing function."""
    # Setup paths
    input_dir = Path("test_diff_files")
    output_dir = Path("output_psf_added")
    psf_file = Path("PSF/psf_array.npy")
    
    # Validation
    if not input_dir.exists():
        print(f"Error: '{input_dir}' does not exist")
        return
    if not psf_file.exists():
        print(f"Error: '{psf_file}' does not exist")
        return
    
    # Load PSF
    print("Loading PSF...")
    psf_array = np.load(psf_file)
    normalized_psf = normalize_array(psf_array)
    psf_fwhm = calculate_fwhm_from_psf(normalized_psf)
    
    print(f"PSF shape: {psf_array.shape}")
    print(f"PSF FWHM: {psf_fwhm:.2f} pixels")
    
    # Find FITS files
    fits_files = list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit"))
    
    if not fits_files:
        print(f"No FITS files found in '{input_dir}'")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Process files
    all_results = []
    for fits_file in fits_files:
        try:
            results = process_single_fits_file(fits_file, normalized_psf, 
                                             psf_fwhm, output_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {fits_file.name}: {e}")
    
    # Save CSV results
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = output_dir / "injection_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {csv_path}")
        print(f"Total injections: {len(all_results)}")
        print(f"SNR range: {results_df['achieved_snr'].min():.2f} - {results_df['achieved_snr'].max():.2f}")
    else:
        print("No results generated")

if __name__ == "__main__":
    main()