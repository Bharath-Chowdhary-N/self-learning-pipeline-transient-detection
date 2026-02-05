#!/usr/bin/env python3
"""
Complete Testing Pipeline for Transient Detection

This pipeline orchestrates the following steps:
1. Find peaks in test diff files and create cutouts
2. Run ensemble models (DenseNet + DeiT) on cutouts
3. Save predictions and results

Usage:
    python test_pipeline.py

Requirements:
- find_peaks_above_k_sigma_test.py must be in the same directory
- testing_script.py must be in the same directory
- Trained model files must be present
"""

import subprocess
import sys
from pathlib import Path
import time

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"{message}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_step(step_num, total_steps, message):
    """Print a formatted step message"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}/{total_steps}] {message}{Colors.ENDC}")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def run_script(script_name, description):
    """
    Run a Python script and handle errors.
    
    Parameters:
    -----------
    script_name : str
        Name of the script to run
    description : str
        Description of what the script does
    
    Returns:
    --------
    success : bool
        True if script ran successfully, False otherwise
    """
    script_path = Path(script_name)
    
    if not script_path.exists():
        print_error(f"Script not found: {script_name}")
        return False
    
    print(f"\n{Colors.OKBLUE}Running: {script_name}{Colors.ENDC}")
    print(f"Description: {description}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print_success(f"Completed {script_name} in {elapsed_time:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(f"Failed to run {script_name} (after {elapsed_time:.1f} seconds)")
        print_error(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(f"Unexpected error running {script_name} (after {elapsed_time:.1f} seconds)")
        print_error(f"Error: {str(e)}")
        return False

def count_files(directory, pattern="*.fits"):
    """Count files matching pattern in directory"""
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))

def main():
    """Main pipeline execution"""
    
    pipeline_start_time = time.time()
    
    print_header("TRANSIENT DETECTION TESTING PIPELINE")
    print(f"{Colors.BOLD}This pipeline will:{Colors.ENDC}")
    print("  1. Find peaks in test diff files and create 64x64 cutouts")
    print("  2. Run DenseNet ensemble on cutouts")
    print("  3. Run DeiT ensemble on cutouts")
    print("  4. Save predictions to ML_results/")
    
    # Configuration
    TOTAL_STEPS = 3
    
    # Script names
    FIND_PEAKS_TEST_SCRIPT = "find_peaks_above_k_sigma_test.py"
    TESTING_SCRIPT = "testing_script.py"
    
    # Directory paths
    TEST_DIR = Path("test_directory")
    CUTOUTS_DIR = Path("output_peaks_from_test_directory")
    RESULTS_DIR = Path("ML_results")
    
    # Track success of each step
    steps_status = []
    
    # =========================================================================
    # STEP 0: Check Prerequisites
    # =========================================================================
    print_step(0, TOTAL_STEPS, "Checking Prerequisites")
    
    # Check if test directory exists
    if not TEST_DIR.exists():
        print_error(f"Test directory not found: {TEST_DIR}")
        print_error("Please create 'test_directory/' and add FITS diff files to it")
        return
    else:
        test_fits_count = count_files(TEST_DIR, "*.fits") + count_files(TEST_DIR, "*.fit")
        if test_fits_count == 0:
            print_error(f"No FITS files found in {TEST_DIR}")
            return
        print_success(f"Found {test_fits_count} FITS files in {TEST_DIR}")
    
    # Check if model files exist
    densenet_models = list(Path(".").glob("DenseNet169_Ensemble_Model*_best.pth"))
    deit_models = list(Path(".").glob("DeiT_Ensemble_Model*_best.pth"))
    
    if not densenet_models and not deit_models:
        print_error("No trained model files found!")
        print_error("Please train models first using training_script.py")
        return
    else:
        print_success(f"Found {len(densenet_models)} DenseNet model(s)")
        print_success(f"Found {len(deit_models)} DeiT model(s)")
    
    # =========================================================================
    # STEP 1: Find Peaks and Create Cutouts
    # =========================================================================
    print_step(1, TOTAL_STEPS, "Finding Peaks and Creating Cutouts")
    
    success = run_script(
        FIND_PEAKS_TEST_SCRIPT,
        "Finds peaks in test diff files and creates 64x64 cutouts"
    )
    steps_status.append(("Find Peaks & Create Cutouts", success))
    
    if not success:
        print_error("Peak finding failed. Cannot continue pipeline.")
        return
    
    # Verify output
    cutouts_count = count_files(CUTOUTS_DIR)
    if cutouts_count == 0:
        print_error("No cutouts were created!")
        return
    print_success(f"Created {cutouts_count} test cutouts")
    
    # =========================================================================
    # STEP 2: Run Ensemble Testing
    # =========================================================================
    print_step(2, TOTAL_STEPS, "Running Ensemble Models on Cutouts")
    
    success = run_script(
        TESTING_SCRIPT,
        "Runs DenseNet and DeiT ensembles on test cutouts"
    )
    steps_status.append(("Ensemble Testing", success))
    
    if not success:
        print_error("Ensemble testing failed.")
        return
    
    # Verify results were saved
    densenet_csv = RESULTS_DIR / "densenet_ensemble_predictions.csv"
    deit_csv = RESULTS_DIR / "deit_ensemble_predictions.csv"
    
    if densenet_csv.exists():
        print_success(f"DenseNet results saved: {densenet_csv}")
    if deit_csv.exists():
        print_success(f"DeiT results saved: {deit_csv}")
    
    # =========================================================================
    # STEP 3: Summary and Analysis Suggestions
    # =========================================================================
    print_step(3, TOTAL_STEPS, "Analysis Complete - Generating Summary")
    
    import pandas as pd
    
    summary_lines = []
    
    if densenet_csv.exists():
        dn_df = pd.read_csv(densenet_csv)
        dn_transients = dn_df[dn_df['prediction'] == 'transient']
        dn_high_conf = dn_df[dn_df['confidence'] > 0.8]
        
        summary_lines.append(f"\n{Colors.BOLD}DenseNet Ensemble Results:{Colors.ENDC}")
        summary_lines.append(f"  Total samples analyzed: {len(dn_df)}")
        summary_lines.append(f"  Predicted transients: {len(dn_transients)} ({100*len(dn_transients)/len(dn_df):.1f}%)")
        summary_lines.append(f"  High confidence (>0.8): {len(dn_high_conf)} ({100*len(dn_high_conf)/len(dn_df):.1f}%)")
        summary_lines.append(f"  Mean confidence: {dn_df['confidence'].mean():.3f}")
    
    if deit_csv.exists():
        dt_df = pd.read_csv(deit_csv)
        dt_transients = dt_df[dt_df['prediction'] == 'transient']
        dt_high_conf = dt_df[dt_df['confidence'] > 0.8]
        
        summary_lines.append(f"\n{Colors.BOLD}DeiT Ensemble Results:{Colors.ENDC}")
        summary_lines.append(f"  Total samples analyzed: {len(dt_df)}")
        summary_lines.append(f"  Predicted transients: {len(dt_transients)} ({100*len(dt_transients)/len(dt_df):.1f}%)")
        summary_lines.append(f"  High confidence (>0.8): {len(dt_high_conf)} ({100*len(dt_high_conf)/len(dt_df):.1f}%)")
        summary_lines.append(f"  Mean confidence: {dt_df['confidence'].mean():.3f}")
    
    # Agreement analysis if both exist
    if densenet_csv.exists() and deit_csv.exists():
        # Merge on filename
        merged = dn_df.merge(dt_df, on='filename', suffixes=('_dn', '_dt'))
        agreement = merged[merged['prediction_dn'] == merged['prediction_dt']]
        both_transient = merged[(merged['prediction_dn'] == 'transient') & 
                               (merged['prediction_dt'] == 'transient')]
        
        summary_lines.append(f"\n{Colors.BOLD}Model Agreement:{Colors.ENDC}")
        summary_lines.append(f"  Both models agree: {len(agreement)} ({100*len(agreement)/len(merged):.1f}%)")
        summary_lines.append(f"  Both predict transient: {len(both_transient)}")
        summary_lines.append(f"  Disagreement: {len(merged) - len(agreement)}")
    
    steps_status.append(("Analysis Summary", True))
    
    # =========================================================================
    # Pipeline Summary
    # =========================================================================
    pipeline_elapsed_time = time.time() - pipeline_start_time
    
    print_header("TESTING PIPELINE EXECUTION SUMMARY")
    
    print(f"\n{Colors.BOLD}Step Results:{Colors.ENDC}")
    for step_name, step_success in steps_status:
        status_symbol = "✓" if step_success else "✗"
        status_color = Colors.OKGREEN if step_success else Colors.FAIL
        print(f"  {status_color}{status_symbol} {step_name}{Colors.ENDC}")
    
    # Print detailed summary
    for line in summary_lines:
        print(line)
    
    print(f"\n{Colors.BOLD}Overall Status:{Colors.ENDC}")
    all_steps_passed = all([status for _, status in steps_status])
    
    if all_steps_passed:
        print_success(f"Pipeline completed successfully in {pipeline_elapsed_time/60:.1f} minutes")
        
        print(f"\n{Colors.BOLD}Output Locations:{Colors.ENDC}")
        print(f"  Test cutouts: {CUTOUTS_DIR}")
        print(f"  Predictions: {RESULTS_DIR}")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"  1. Review predictions in {RESULTS_DIR}")
        print(f"  2. Filter for high-confidence detections (confidence > 0.8)")
        print(f"  3. Investigate cases where both models agree on transients")
        print(f"  4. Visually inspect flagged transients in {CUTOUTS_DIR}")
        print(f"  5. Cross-reference with original diff files in {TEST_DIR}")
        
        if densenet_csv.exists() and deit_csv.exists():
            print(f"\n{Colors.BOLD}Analysis Commands:{Colors.ENDC}")
            print(f"  # Find high-confidence detections where both models agree:")
            print(f"  python -c \"import pandas as pd; dn=pd.read_csv('{densenet_csv}'); dt=pd.read_csv('{deit_csv}'); m=dn.merge(dt,on='filename',suffixes=('_dn','_dt')); print(m[(m['prediction_dn']=='transient')&(m['prediction_dt']=='transient')&(m['confidence_dn']>0.8)&(m['confidence_dt']>0.8)][['filename','mean_probability_dn','mean_probability_dt']])\"")
    else:
        print_error(f"Pipeline completed with errors after {pipeline_elapsed_time/60:.1f} minutes")
        print("\nPlease review the errors above and re-run the pipeline")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Unexpected error in pipeline:{Colors.ENDC}")
        print(f"{Colors.FAIL}{str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
