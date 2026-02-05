# Transient Detection Pipeline

A complete machine learning pipeline for detecting astronomical transients in difference images using ensemble deep learning models.

## Getting Started

This pipeline has two main stages: training and testing. You'll run the training pipeline first to build your models, then run the testing pipeline to find transients in new data.

## Prerequisites

Before running anything, you need to prepare three directories with your data:

**1. Training Sample Files**

Create a folder called test_diff_files and add one or more FITS difference files. These will be used to generate training data.
```
test_diff_files/
    diff_file_1.fits
    diff_file_2.fits
    diff_file_3.fits
```

**2. Point Spread Function**

Create a PSF folder and add your PSF array as psf_array.npy. This PSF will be used to inject synthetic transients.
```
PSF/
    psf_array.npy
```


**3. Test Data for Discovery**

Create a test_directory folder and add the FITS difference files where you want to discover real transients.
```
test_directory/
    new_diff_file_1.fits
    new_diff_file_2.fits
```

## Installation

Install the required Python packages:
```bash
pip install torch torchvision timm astropy numpy pandas matplotlib scipy photutils pillow splitfolders scikit-image tqdm
```

## Part 1: Training the Models

Run the training pipeline to create your ensemble models:
```bash
python train_pipeline.py
```

### What the Training Pipeline Does

The training pipeline runs through 8 automated steps:

**Step 1: PSF Injection**

Takes your sample difference files and injects synthetic transients at random locations. Creates positive training samples with known transients at various signal-to-noise ratios ranging from 3 to 10.

Output: Positive samples saved to pos_and_neg/positives/

**Step 2: Find Non-Peak Regions**

Scans the same difference files to find uniform, quiet regions with no detections. These become your negative training samples.

Output: Negative samples saved to pos_and_neg/negatives/

**Step 3: Find Real Peaks**

Identifies real peaks in the difference images to add more realistic negative samples. This helps the model learn what real astronomical sources look like versus true transients.

Output: Additional negative samples added to pos_and_neg/negatives/

**Step 4: Organize Positive Samples**

Copies all PSF-injected cutouts into the final training directory structure.

**Step 5: Organize Negative Samples**

Gathers all negative samples from various sources into a single organized directory.

**Step 6: Verify Dataset**

Checks that you have a balanced dataset and displays statistics about class distribution.

**Step 7: Optional Cleanup**

Intermediate files are kept by default for verification. You can manually delete them later if needed.

**Step 8: Train Models**

Trains an ensemble of 6 models:
- 4 DenseNet169 models
- 2 DeiT transformer models

Each model trains for 10 epochs with early stopping. The best model from each training run is automatically saved.

### Training Outputs

After training completes, you'll have:
```
DenseNet169_Ensemble_Model1_best.pth
DenseNet169_Ensemble_Model2_best.pth
DenseNet169_Ensemble_Model3_best.pth
DenseNet169_Ensemble_Model4_best.pth
DeiT_Ensemble_Model1_best.pth
DeiT_Ensemble_Model2_best.pth

DenseNet169_Ensemble_Model1_progress.png
DenseNet169_Ensemble_Model2_progress.png
(and so on...)
```

The training also creates:
```
pos_and_neg/
    positives/
        (thousands of FITS files with injected transients)
    negatives/
        (thousands of FITS files with non-transients)

split_folders/
    train/
    val/
    test/
```

## Part 2: Testing on New Data

After training is complete, run the testing pipeline to find transients in your test data:
```bash
python test_pipeline.py
```

### What the Testing Pipeline Does

The testing pipeline runs through 3 automated steps:

**Step 1: Find Peaks in Test Data**

Scans all FITS files in test_directory and finds peaks above a detection threshold. Creates 64x64 pixel cutouts around each detected peak.

Output: Cutouts saved to output_peaks_from_test_directory/

**Step 2: Run Ensemble Models**

Loads all trained models and runs them on each cutout. Each model votes on whether the cutout contains a real transient or not. The ensemble averages the predictions to give you a final probability and confidence score.

**Step 3: Save Results and Visualizations**

Creates detailed CSV files with predictions and saves PNG images sorted by prediction.

### Testing Outputs

After testing completes, you'll have:
```
ML_results/
    densenet_ensemble_predictions.csv
    deit_ensemble_predictions.csv
    positives/
        DenseNet_positive_0000_peak_001.png
        DenseNet_positive_0001_peak_005.png
        DeiT_positive_0000_peak_001.png
        (PNG images of predicted transients)
    negatives/
        DenseNet_negative_0000_peak_002.png
        DenseNet_negative_0001_peak_003.png
        DeiT_negative_0000_peak_002.png
        (PNG images of predicted non-transients)
```

### Understanding the Results

**CSV Files**

Each CSV contains predictions with these columns:

- filename: Name of the FITS cutout file
- filepath: Full path to the file
- mean_probability: Average probability from the ensemble (0 to 1)
- std_probability: How much the models disagreed
- prediction: Either "transient" or "non-transient"
- confidence: How confident the prediction is (0 to 1)
- num_models: Number of models in the ensemble
- DenseNet169_model_1_prob: Individual model probabilities
- DenseNet169_model_2_prob: (and so on for each model)

**What to Look For**

High confidence transients are those where:
- mean_probability is close to 1.0
- confidence is above 0.8
- std_probability is low (models agree)
- Both DenseNet and DeiT predict "transient"

**PNG Files**

Visual inspection is crucial. Look through the images in ML_results/positives/ to verify the detections make sense. Real transients should appear as point sources near the center of the cutout.

## Typical Workflow

Here's how you'd normally use this pipeline:

1. Collect a few representative difference images and put them in test_diff_files/
2. Prepare your PSF and save it to PSF/psf_array.npy
3. Run python train_pipeline.py and wait for training to complete (this may take several hours)
4. Check the training plots to ensure models learned properly
5. Put your science target difference images in test_directory/
6. Run python test_pipeline.py to find transients
7. Review the predictions in ML_results/
8. Visually inspect high-confidence detections
9. Cross-reference with your original difference images

## Customization

You can adjust parameters by editing the scripts:

**Training Parameters** (in training_script.py):
- num_epochs: How long to train each model
- batch_size: Adjust based on your GPU memory
- learning_rate: Control training speed

**Detection Parameters** (in find_peaks_above_k_sigma_test.py):
- sigma_threshold: Lower values find fainter sources
- edge_buffer: Distance from image edges
- max_peaks: Limit number of detections per image

**Confidence Threshold** (when analyzing results):
- Filter CSV files for confidence greater than 0.8 for high-quality detections
- Require both DenseNet and DeiT to agree for extra confidence

## Troubleshooting

**Training pipeline fails at Step 1**

Make sure you have FITS files in test_diff_files/ and PSF/psf_array.npy exists.

**No cutouts generated during testing**

Check that test_directory/ contains valid FITS files and try lowering sigma_threshold.

**Models predict everything as non-transient**

Your training data might be imbalanced. Check that pos_and_neg/positives/ and pos_and_neg/negatives/ have similar numbers of files.

**Out of memory errors**

Reduce batch_size in training_script.py or use a GPU with more memory.

## File Organization

After running both pipelines, your directory structure will look like:
```
project/
    train_pipeline.py
    test_pipeline.py
    training_script.py
    testing_script.py
    psf_injection_script.py
    find_non_peaks_64.py
    find_peaks_above_k_sigma_training.py
    find_peaks_above_k_sigma_test.py
    
    test_diff_files/
        (your training FITS files)
    
    PSF/
        psf_array.npy
    
    test_directory/
        (your science target FITS files)
    
    pos_and_neg/
        positives/
        negatives/
    
    output_peaks_from_test_directory/
        (cutouts from test data)
    
    ML_results/
        positives/
        negatives/
        densenet_ensemble_predictions.csv
        deit_ensemble_predictions.csv
    
    DenseNet169_Ensemble_Model1_best.pth
    DenseNet169_Ensemble_Model2_best.pth
    (and other model files...)
```

## Technical Details

**Data Normalization**

All images are normalized using ZScale, which is the astronomical standard. This uses robust statistics to determine optimal display ranges rather than simple min-max scaling.

**Model Architecture**

- DenseNet169: A convolutional neural network with dense connections
- DeiT: A vision transformer model adapted for 64x64 images

Both architectures are proven for image classification tasks.

**Ensemble Strategy**

Using multiple models reduces overfitting and provides uncertainty estimates. When models disagree (high std_probability), you should be more cautious about the prediction.

**Training Strategy**

- 80% of data used for training
- 10% for validation (used to save best model)
- 10% held out for testing
- Early stopping prevents overtraining
- Data augmentation is intentionally minimal to preserve astronomical features


