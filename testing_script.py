#!/usr/bin/env python3
"""
Ensemble Testing Script for Transient Detection

This script tests trained ensemble models (DenseNet169 + DeiT) on FITS cutouts
and saves predictions with confidence scores. Also saves PNG visualizations
of predicted positives and negatives.

Requirements:
- torch
- torchvision
- timm
- astropy
- numpy
- pandas
- tqdm
- pillow

Install with: pip install torch torchvision timm astropy numpy pandas tqdm pillow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from tqdm import tqdm
from PIL import Image
import shutil
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# ZScale Normalization (MUST MATCH TRAINING)
# ============================================================================

def normalize_with_zscale(data):
    """
    Normalize data using astropy's ZScale algorithm.
    CRITICAL: Must match training normalization exactly.
    """
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    
    zscale = ZScaleInterval()
    try:
        vmin, vmax = zscale.get_limits(data[valid_mask])
        
        if vmax > vmin:
            normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(data)
        
        normalized[~valid_mask] = 0
        
    except Exception as e:
        if data.max() > data.min():
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            normalized = np.zeros_like(data)
    
    return normalized

# ============================================================================
# Model Architectures (MUST MATCH TRAINING)
# ============================================================================

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def bn_function(self, inputs):
        if isinstance(inputs, torch.Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs
            
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 
                                                   kernel_size=7, stride=2, 
                                                   padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return torch.sigmoid(out).squeeze()

def densenet169(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)

class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(DeiTClassifier, self).__init__()
        
        self.deit = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            img_size=64,
            global_pool='avg'
        )
        
        feature_dim = self.deit.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.deit(x)
        out = self.classifier(features)
        return torch.sigmoid(out).squeeze()

def create_deit(**kwargs):
    return DeiTClassifier(**kwargs)

# ============================================================================
# Model Loading and Ensemble Prediction
# ============================================================================

def load_model(model_path, model_type, device):
    """
    Load a trained model from checkpoint.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to model checkpoint
    model_type : str
        'densenet' or 'deit'
    device : torch.device
        Device to load model on
    
    Returns:
    --------
    model : nn.Module
        Loaded model in eval mode
    """
    if model_type.lower() == 'densenet':
        model = densenet169(num_classes=1)
    elif model_type.lower() == 'deit':
        model = create_deit(num_classes=1, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def load_fits_image(fits_path):
    """
    Load and preprocess a FITS image for model inference.
    
    Parameters:
    -----------
    fits_path : str or Path
        Path to FITS file
    
    Returns:
    --------
    tensor : torch.Tensor
        Preprocessed image tensor (1, 3, 64, 64)
    raw_data : numpy.ndarray
        Raw normalized image data for visualization
    """
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data.astype(np.float32)
        
        # Handle different data shapes
        if image_data.ndim > 2:
            image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()
        
        # Ensure 64x64
        if image_data.shape != (64, 64):
            from skimage.transform import resize
            image_data = resize(image_data, (64, 64), mode='constant', anti_aliasing=True)
        
        # ZScale normalization (MUST match training)
        image_data = normalize_with_zscale(image_data)
        
        # Save raw normalized data for PNG creation
        raw_data = image_data.copy()
        
        # Convert to tensor and replicate to 3 channels
        image_tensor = torch.from_numpy(image_data).float()
        image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, raw_data

def save_png(image_data, output_path):
    """
    Save normalized image data as PNG.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Normalized image data (0-1 range)
    output_path : str or Path
        Output PNG file path
    """
    # Convert to 8-bit
    image_8bit = (image_data * 255).astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(image_8bit, mode='L')
    img.save(output_path)

def ensemble_predict(models, image_tensor, device, model_type_name):
    """
    Make ensemble prediction from multiple models.
    
    Parameters:
    -----------
    models : list of nn.Module
        List of models
    image_tensor : torch.Tensor
        Input image tensor
    device : torch.device
        Device for inference
    model_type_name : str
        Name of model type for column naming (e.g., 'DenseNet169', 'DeiT')
    
    Returns:
    --------
    mean_prob : float
        Mean probability across ensemble
    std_prob : float
        Standard deviation of probabilities
    individual_probs : dict
        Dictionary with individual model probabilities
    """
    image_tensor = image_tensor.to(device)
    
    probabilities = []
    individual_probs = {}
    
    with torch.no_grad():
        for i, model in enumerate(models, 1):
            output = model(image_tensor)
            if output.dim() == 0:
                prob = output.item()
            else:
                prob = output[0].item()
            probabilities.append(prob)
            individual_probs[f'{model_type_name}_model_{i}_prob'] = prob
    
    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    
    return mean_prob, std_prob, individual_probs

# ============================================================================
# Main Testing Function
# ============================================================================

def main():
    """Main testing function"""
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE TESTING - TRANSIENT DETECTION")
    print(f"{'='*70}\n")
    
    # Configuration
    test_dir = Path("output_peaks_from_test_directory")
    output_dir = Path("ML_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for PNG outputs
    positives_dir = output_dir / "positives"
    negatives_dir = output_dir / "negatives"
    positives_dir.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if test directory exists
    if not test_dir.exists():
        print(f"\n❌ Error: Test directory not found: {test_dir}")
        print("Please run find_peaks_above_k_sigma_test.py first to generate test cutouts")
        return
    
    # Find all FITS files
    fits_files = list(test_dir.glob("*.fits"))
    if not fits_files:
        print(f"\n❌ Error: No FITS files found in {test_dir}")
        return
    
    print(f"\nFound {len(fits_files)} FITS files to test")
    
    # ========================================================================
    # Load DenseNet Models
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Loading DenseNet169 Ensemble Models")
    print(f"{'='*70}")
    
    densenet_models = []
    densenet_model_files = sorted(Path(".").glob("DenseNet169_Ensemble_Model*_best.pth"))
    
    if not densenet_model_files:
        print("\n⚠️  Warning: No DenseNet model files found")
    else:
        for model_file in densenet_model_files:
            print(f"  Loading {model_file.name}...")
            try:
                model = load_model(model_file, 'densenet', device)
                densenet_models.append(model)
                print(f"    ✓ Loaded successfully")
            except Exception as e:
                print(f"    ✗ Failed to load: {e}")
        
        print(f"\n✓ Loaded {len(densenet_models)} DenseNet models")
    
    # ========================================================================
    # Load DeiT Models
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Loading DeiT Ensemble Models")
    print(f"{'='*70}")
    
    deit_models = []
    deit_model_files = sorted(Path(".").glob("DeiT_Ensemble_Model*_best.pth"))
    
    if not deit_model_files:
        print("\n⚠️  Warning: No DeiT model files found")
    else:
        for model_file in deit_model_files:
            print(f"  Loading {model_file.name}...")
            try:
                model = load_model(model_file, 'deit', device)
                deit_models.append(model)
                print(f"    ✓ Loaded successfully")
            except Exception as e:
                print(f"    ✗ Failed to load: {e}")
        
        print(f"\n✓ Loaded {len(deit_models)} DeiT models")
    
    if not densenet_models and not deit_models:
        print("\n❌ Error: No models loaded. Cannot proceed with testing.")
        return
    
    # ========================================================================
    # Run Inference on All Test Images
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Running Inference on Test Images")
    print(f"{'='*70}\n")
    
    densenet_results = []
    deit_results = []
    
    # Counters for PNG saving
    dn_positive_count = 0
    dn_negative_count = 0
    dt_positive_count = 0
    dt_negative_count = 0
    
    for fits_file in tqdm(fits_files, desc="Processing images"):
        try:
            # Load image
            image_tensor, raw_image_data = load_fits_image(fits_file)
            
            # DenseNet predictions
            if densenet_models:
                dn_mean, dn_std, dn_individual_probs = ensemble_predict(
                    densenet_models, image_tensor, device, 'DenseNet169'
                )
                
                dn_prediction = 'transient' if dn_mean >= 0.5 else 'non-transient'
                
                densenet_results.append({
                    'filename': fits_file.name,
                    'filepath': str(fits_file),
                    'mean_probability': dn_mean,
                    'std_probability': dn_std,
                    'prediction': dn_prediction,
                    'confidence': max(dn_mean, 1 - dn_mean),
                    'num_models': len(densenet_models),
                    **dn_individual_probs
                })
                
                # Save PNG based on prediction
                if dn_prediction == 'transient':
                    png_path = positives_dir / f"DenseNet_positive_{dn_positive_count:04d}_{fits_file.stem}.png"
                    save_png(raw_image_data, png_path)
                    dn_positive_count += 1
                else:
                    png_path = negatives_dir / f"DenseNet_negative_{dn_negative_count:04d}_{fits_file.stem}.png"
                    save_png(raw_image_data, png_path)
                    dn_negative_count += 1
            
            # DeiT predictions
            if deit_models:
                dt_mean, dt_std, dt_individual_probs = ensemble_predict(
                    deit_models, image_tensor, device, 'DeiT'
                )
                
                dt_prediction = 'transient' if dt_mean >= 0.5 else 'non-transient'
                
                deit_results.append({
                    'filename': fits_file.name,
                    'filepath': str(fits_file),
                    'mean_probability': dt_mean,
                    'std_probability': dt_std,
                    'prediction': dt_prediction,
                    'confidence': max(dt_mean, 1 - dt_mean),
                    'num_models': len(deit_models),
                    **dt_individual_probs
                })
                
                # Save PNG based on prediction
                if dt_prediction == 'transient':
                    png_path = positives_dir / f"DeiT_positive_{dt_positive_count:04d}_{fits_file.stem}.png"
                    save_png(raw_image_data, png_path)
                    dt_positive_count += 1
                else:
                    png_path = negatives_dir / f"DeiT_negative_{dt_negative_count:04d}_{fits_file.stem}.png"
                    save_png(raw_image_data, png_path)
                    dt_negative_count += 1
        
        except Exception as e:
            print(f"\n⚠️  Error processing {fits_file.name}: {e}")
            continue
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Saving Results")
    print(f"{'='*70}\n")
    
    # Save DenseNet results
    if densenet_results:
        densenet_df = pd.DataFrame(densenet_results)
        densenet_csv = output_dir / "densenet_ensemble_predictions.csv"
        densenet_df.to_csv(densenet_csv, index=False)
        print(f"✓ Saved DenseNet results: {densenet_csv}")
        
        # Statistics
        dn_transients = densenet_df[densenet_df['prediction'] == 'transient']
        print(f"  DenseNet Predictions:")
        print(f"    Total samples: {len(densenet_df)}")
        print(f"    Predicted transients: {len(dn_transients)}")
        print(f"    Predicted non-transients: {len(densenet_df) - len(dn_transients)}")
        print(f"    Mean confidence: {densenet_df['confidence'].mean():.3f}")
        print(f"    Mean probability: {densenet_df['mean_probability'].mean():.3f}")
        print(f"  DenseNet PNG outputs:")
        print(f"    Positives: {dn_positive_count} files in {positives_dir}")
        print(f"    Negatives: {dn_negative_count} files in {negatives_dir}")
    
    # Save DeiT results
    if deit_results:
        deit_df = pd.DataFrame(deit_results)
        deit_csv = output_dir / "deit_ensemble_predictions.csv"
        deit_df.to_csv(deit_csv, index=False)
        print(f"\n✓ Saved DeiT results: {deit_csv}")
        
        # Statistics
        dt_transients = deit_df[deit_df['prediction'] == 'transient']
        print(f"  DeiT Predictions:")
        print(f"    Total samples: {len(deit_df)}")
        print(f"    Predicted transients: {len(dt_transients)}")
        print(f"    Predicted non-transients: {len(deit_df) - len(dt_transients)}")
        print(f"    Mean confidence: {deit_df['confidence'].mean():.3f}")
        print(f"    Mean probability: {deit_df['mean_probability'].mean():.3f}")
        print(f"  DeiT PNG outputs:")
        print(f"    Positives: {dt_positive_count} files in {positives_dir}")
        print(f"    Negatives: {dt_negative_count} files in {negatives_dir}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"TESTING COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"Results saved to: {output_dir}")
    print(f"  CSV files:")
    print(f"    - densenet_ensemble_predictions.csv")
    print(f"    - deit_ensemble_predictions.csv")
    print(f"  PNG visualizations:")
    print(f"    - {positives_dir}/ (predicted transients)")
    print(f"    - {negatives_dir}/ (predicted non-transients)")
    
    print(f"\nNext steps:")
    print(f"  1. Review predictions in CSV files")
    print(f"  2. Visually inspect PNGs in {positives_dir} and {negatives_dir}")
    print(f"  3. Filter by confidence threshold (e.g., > 0.8)")
    print(f"  4. Analyze agreement between DenseNet and DeiT")
    print(f"  5. Investigate high-confidence detections")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()