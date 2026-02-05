#!/usr/bin/env python3
"""
FITS Non-Peak Finder and Cutout Generator - Improved Version

This script processes FITS files to find uniform non-peak regions (avoiding edges, artifacts, and invalid pixels)
and create 64x64 cutouts. Uses zscale normalization and saves outputs as both PNG and FITS files.

Requirements:
- astropy
- numpy
- matplotlib
- scipy
- photutils (for peak detection)
- pillow (for exact PNG dimensions)

Install with: pip install astropy numpy matplotlib scipy photutils pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.ndimage import label, generate_binary_structure, uniform_filter
from PIL import Image
import warnings

# Suppress astropy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def find_uniform_non_peaks(data, min_distance=10, num_non_peaks=500, cutout_size=64):
    """
    Find non-peak regions by avoiding edges and areas with invalid pixels.
    Fast approach using vectorized operations.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of image data
    min_distance : int
        Minimum distance between non-peaks in pixels
    num_non_peaks : int
        Maximum number of non-peaks to return
    cutout_size : int
        Size of cutouts (used to avoid edges)
    
    Returns:
    --------
    non_peaks : list of tuples
        List of (y, x) coordinates of detected non-peaks
    """
    # Create a buffer zone to avoid edges
    buffer = cutout_size // 2 + 10
    
    # Create mask for valid pixels (not zero, not NaN)
    valid_pixel_mask = np.isfinite(data) & (data != 0)
    
    # Use scipy's uniform_filter to quickly check for invalid pixels in neighborhoods
    from scipy.ndimage import uniform_filter
    
    # Create a binary mask (1 for valid, 0 for invalid)
    valid_binary = valid_pixel_mask.astype(float)
    
    # Apply uniform filter - this gives the fraction of valid pixels in each neighborhood
    kernel_size = cutout_size
    valid_fraction = uniform_filter(valid_binary, size=kernel_size, mode='constant', cval=0.0)
    
    # Only keep regions where ALL pixels in the cutout area are valid (fraction = 1.0)
    fully_valid_mask = valid_fraction == 1.0
    
    # Apply edge buffer
    fully_valid_mask[:buffer, :] = False
    fully_valid_mask[-buffer:, :] = False
    fully_valid_mask[:, :buffer] = False
    fully_valid_mask[:, -buffer:] = False
    
    # Find low-intensity regions within valid areas
    if np.any(fully_valid_mask):
        threshold = np.percentile(data[fully_valid_mask], 40)
        candidate_mask = fully_valid_mask & (data < threshold)
    else:
        print("  Warning: No fully valid regions found")
        return []
    
    # Get candidate coordinates
    candidate_coords = np.column_stack(np.where(candidate_mask))
    
    if len(candidate_coords) == 0:
        print("  Warning: No candidate regions found")
        return []
    
    # Randomly shuffle for better distribution
    np.random.seed(42)
    np.random.shuffle(candidate_coords)
    
    # Filter by minimum distance
    filtered_non_peaks = []
    for candidate in candidate_coords:
        if len(filtered_non_peaks) >= num_non_peaks:
            break
            
        if len(filtered_non_peaks) == 0:
            filtered_non_peaks.append(tuple(candidate))
        else:
            distances = [np.sqrt((candidate[0] - p[0])**2 + (candidate[1] - p[1])**2) 
                        for p in filtered_non_peaks]
            if min(distances) >= min_distance:
                filtered_non_peaks.append(tuple(candidate))
    
    return filtered_non_peaks

def create_cutout(data, center_y, center_x, cutout_size=64):
    """
    Create a square cutout from 2D data centered at given coordinates.
    Simplified validation - just check for invalid pixels.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of image data
    center_y, center_x : int
        Center coordinates for the cutout
    cutout_size : int
        Size of the square cutout
    
    Returns:
    --------
    cutout : numpy.ndarray or None
        Cutout array. Returns None if invalid.
    """
    half_size = cutout_size // 2
    
    # Calculate bounds
    y_min = center_y - half_size
    y_max = center_y + half_size
    x_min = center_x - half_size
    x_max = center_x + half_size
    
    # Extract the cutout
    cutout = data[y_min:y_max, x_min:x_max]
    
    # Simple check: any invalid pixels means reject
    if np.any(~np.isfinite(cutout)) or np.any(cutout == 0):
        return None
    
    return cutout

def normalize_with_zscale(data):
    """
    Normalize data using astropy's ZScale algorithm.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    
    Returns:
    --------
    normalized_data : numpy.ndarray
        Normalized data between 0 and 1
    """
    # Handle invalid values
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    
    # Apply ZScale normalization
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data[valid_mask])
    
    # Normalize to 0-1 range
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Handle any remaining invalid values
    normalized[~valid_mask] = 0
    
    return normalized

def save_cutout_png(cutout, filepath):
    """
    Save cutout as PNG file with exact pixel dimensions.
    
    Parameters:
    -----------
    cutout : numpy.ndarray
        Normalized cutout data (0-1 range)
    filepath : str or Path
        Output file path
    """
    # Convert to 8-bit integers for PNG
    cutout_8bit = (cutout * 255).astype(np.uint8)
    
    # Create PIL Image and save directly
    # PIL expects (height, width) which matches our numpy array shape
    img = Image.fromarray(cutout_8bit, mode='L')  # 'L' mode for grayscale
    img.save(filepath)

def save_cutout_fits(cutout, filepath, header=None):
    """
    Save cutout as FITS file.
    
    Parameters:
    -----------
    cutout : numpy.ndarray
        Cutout data
    filepath : str or Path
        Output file path
    header : astropy.io.fits.Header or None
        FITS header to include
    """
    hdu = fits.PrimaryHDU(data=cutout, header=header)
    hdu.writeto(filepath, overwrite=True)

def process_fits_file(fits_path, output_dir, cutout_size=64, num_non_peaks=500):
    """
    Process a single FITS file to find uniform non-peak regions and create cutouts.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    
    # Get tile name (filename without extension)
    tile_name = fits_path.stem
    
    # Create output directories
    png_dir = output_dir / 'png' / tile_name
    fits_dir = output_dir / 'fits' / tile_name
    png_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read FITS file - HANDLE EXTENSION HDUs LIKE MAIN.PY
        print(f"Processing {fits_path.name}...")
        
        data = None
        header = None
        
        with fits.open(fits_path) as hdul:
            print(f"  Number of HDUs: {len(hdul)}")
            
            # Look for SCI extension first (same as your main.py)
            if 'SCI' in hdul:
                data = hdul['SCI'].data.copy()
                header = hdul['SCI'].header.copy()
                print(f"  Found SCI extension, shape: {data.shape}")
            else:
                # Otherwise search for first valid 2D image
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and len(hdu.data.shape) == 2:
                        data = hdu.data.copy()
                        header = hdu.header.copy()
                        print(f"  Found 2D image data in HDU {i}, shape: {data.shape}")
                        break
        
        if data is None:
            print(f"  Error: No valid image data found in {fits_path.name}")
            return
        
        # Handle different data shapes
        if data.ndim > 2:
            data = data[0] if data.ndim == 3 else data.squeeze()
        
        # CRITICAL FIX: Clean and normalize data BEFORE finding non-peaks
        print(f"  Raw data range: min={np.nanmin(data):.6f}, max={np.nanmax(data):.6f}")
        
        # Replace NaN/inf with 0 (same as main.py)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply ZScale normalization for better peak detection
        zscale = ZScaleInterval()
        valid_mask = np.isfinite(data) & (data != 0)
        
        if not np.any(valid_mask):
            print(f"  Error: No valid data in {fits_path.name}")
            return
        
        zmin, zmax = zscale.get_limits(data[valid_mask])
        print(f"  ZScale limits: zmin={zmin:.6f}, zmax={zmax:.6f}")
        
        # Normalize to [0, 1] range for processing
        data_normalized = np.clip((data - zmin) / (zmax - zmin), 0, 1)
        print(f"  Normalized data range: min={np.min(data_normalized):.6f}, max={np.max(data_normalized):.6f}")
        
        # Find non-peaks using NORMALIZED data
        non_peaks = find_uniform_non_peaks(data_normalized, min_distance=cutout_size//2, 
                                          num_non_peaks=num_non_peaks, cutout_size=cutout_size)
        
        if not non_peaks:
            print(f"  No valid non-peaks found in {fits_path.name}")
            return
        
        print(f"  Found {len(non_peaks)} valid non-peak candidates")
        
        # Create cutouts for each non-peak
        valid_cutouts = 0
        for i, (non_peak_y, non_peak_x) in enumerate(non_peaks):
            # Create cutout from NORMALIZED data
            cutout = create_cutout(data_normalized, non_peak_y, non_peak_x, cutout_size)
            
            # Skip invalid cutouts
            if cutout is None:
                continue
            
            # Cutout is already normalized, but apply zscale again for consistency
            normalized_cutout = normalize_with_zscale(cutout)
            
            # Generate output filenames
            cutout_name = f"{tile_name}_nonpeak_{valid_cutouts:03d}_y{non_peak_y}_x{non_peak_x}"
            png_path = png_dir / f"{cutout_name}.png"
            fits_path_out = fits_dir / f"{cutout_name}.fits"
            
            # Save PNG
            save_cutout_png(normalized_cutout, png_path)
            
            # Save FITS (save the cutout as-is, already normalized)
            save_cutout_fits(cutout, fits_path_out, header)
            
            valid_cutouts += 1
        
        print(f"  Saved {valid_cutouts} valid non-peak cutouts for {fits_path.name}")
        
    except Exception as e:
        print(f"  Error processing {fits_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to process all FITS files in the input directory.
    """
    # Define paths
    input_dir = Path("test_diff_files/")
    output_dir = Path("non_peaks")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Find all FITS files
    fits_files = list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit"))
    
    if not fits_files:
        print(f"No FITS files found in '{input_dir}'")
        return
    
    print(f"Found {len(fits_files)} FITS files to process")
    
    # Process each file
    for fits_file in fits_files:
        process_fits_file(fits_file, output_dir, num_non_peaks=10000)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()