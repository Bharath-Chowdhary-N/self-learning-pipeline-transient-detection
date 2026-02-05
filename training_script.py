import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import splitfolders
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import torch.nn.functional as F
import timm  # ADD THIS IMPORT for DeiT

# ============================================================================
# ZScale Normalization Function (CRITICAL FOR ASTRONOMICAL DATA)
# ============================================================================

def normalize_with_zscale(data):
    """
    Normalize data using astropy's ZScale algorithm, then normalize to 0-1 range.
    This is the CORRECT normalization for astronomical data.
    
    ZScale uses robust statistics to determine optimal display range,
    which is much better than simple min-max for astronomical images.
    
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
    try:
        vmin, vmax = zscale.get_limits(data[valid_mask])
        
        # Normalize to 0-1 range
        if vmax > vmin:
            normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            # If vmax == vmin, return zeros
            normalized = np.zeros_like(data)
        
        # Handle any remaining invalid values
        normalized[~valid_mask] = 0
        
    except Exception as e:
        # Fallback to simple min-max if ZScale fails
        print(f"Warning: ZScale failed, using min-max fallback: {e}")
        if data.max() > data.min():
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            normalized = np.zeros_like(data)
    
    return normalized

# ============================================================================
# DenseNet Architecture
# ============================================================================

class _DenseLayer(nn.Module):
    """Dense layer implementation following original DenseNet paper"""
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
        """Bottleneck function"""
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
    """Dense block implementation following original DenseNet paper"""
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
    """Transition layer implementation following original DenseNet paper"""
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """DenseNet-BC model implementation optimized for 64x64 inputs"""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):
        super(DenseNet, self).__init__()

        # First convolution - adapted for 64x64 input
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 
                                                   kernel_size=7, stride=2, 
                                                   padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Each denseblock
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

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
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
    """DenseNet-169 model"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)

# ============================================================================
# DeiT (Data-efficient Image Transformer) Architecture
# ============================================================================

class DeiTClassifier(nn.Module):
    """
    DeiT (Data-efficient Image Transformer) wrapper for binary classification.
    Uses pre-trained DeiT-Small model from timm library.
    """
    def __init__(self, num_classes=1, pretrained=False):
        super(DeiTClassifier, self).__init__()
        
        # Load DeiT-Small model (can also use 'deit_tiny_patch16_224' or 'deit_base_patch16_224')
        # Note: DeiT expects 224x224 images, so we'll need to resize or use a smaller variant
        # For 64x64 images, we use deit_tiny with img_size=64
        self.deit = timm.create_model(
            'deit_tiny_patch16_224',  # You can change to 'deit_small_patch16_224' for larger model
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            img_size=64,  # Set to 64x64 input size
            global_pool='avg'
        )
        
        # Get the feature dimension
        feature_dim = self.deit.num_features
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.deit(x)
        
        # Classify
        out = self.classifier(features)
        
        return torch.sigmoid(out).squeeze()

def create_deit(**kwargs):
    """Create DeiT model"""
    return DeiTClassifier(**kwargs)

# ============================================================================
# Training Progress Tracking
# ============================================================================

class TrainingProgress:
    def __init__(self, model_name):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.model_name = model_name
        
    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def plot_progress(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'{self.model_name} - Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        plt.title(f'{self.model_name} - Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_progress.png', dpi=300, bbox_inches='tight')
        plt.close()

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), 100 * correct / total

def train_single_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    progress = TrainingProgress(model_name)
    best_val_acc = 0.0
    patience = 5  # Reduced patience for shorter training
    patience_counter = 0
    
    model_file = f'{model_name}_best.pth'
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Removed existing {model_file}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'{model_name}, Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler with validation loss
        scheduler.step(val_loss)
        
        progress.update(train_loss, train_acc, val_loss, val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_file)
            print(f"✓ Saved best {model_name} with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        progress.plot_progress()
        
        print(f'\n{model_name}, Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered for {model_name}")
            break
    
    return best_val_acc

class FITSDataset(Dataset):
    """
    FITS Dataset with ZScale normalization for astronomical data.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.fits'):
                        self.files.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_idx)
        
        print(f"Loaded {len(self.files)} files from {root_dir}")
        print(f"Using ZScale normalization for all images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with fits.open(self.files[idx]) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                
                # Handle different data shapes
                if image_data.ndim > 2:
                    image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()
                
                # Resize if needed
                if image_data.shape != (64, 64):
                    from skimage.transform import resize
                    image_data = resize(image_data, (64, 64), mode='constant', anti_aliasing=True)
                
                # CRITICAL: Use ZScale normalization (SAME as inference script)
                image_data = normalize_with_zscale(image_data)
                
                # Convert to tensor and replicate to 3 channels
                image_tensor = torch.from_numpy(image_data).float()
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
                
                if self.transform:
                    image_tensor = self.transform(image_tensor)

                return image_tensor, self.labels[idx]

        except Exception as e:
            print(f"Error loading file {self.files[idx]}: {str(e)}")
            return torch.zeros((3, 64, 64)), self.labels[idx]

def plot_sample_images(loader):
    positive_sample = None
    negative_sample = None
    
    for data, labels in loader:
        for i in range(len(labels)):
            if labels[i] == 1 and positive_sample is None:
                positive_sample = data[i]
            elif labels[i] == 0 and negative_sample is None:
                negative_sample = data[i]
            
            if positive_sample is not None and negative_sample is not None:
                break
        if positive_sample is not None and negative_sample is not None:
            break

    if positive_sample is None or negative_sample is None:
        print("Could not find samples from both classes.")
        return

    positive_image = positive_sample[0].cpu().numpy()
    negative_image = negative_sample[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(positive_image, cmap='viridis')
    ax1.set_title('Positive Sample (with PSF)\n[ZScale Normalized]', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(negative_image, cmap='viridis')
    ax2.set_title('Negative Sample (non-peak)\n[ZScale Normalized]', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('sample_images_64x64_zscale.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Sample images saved as 'sample_images_64x64_zscale.png'")

def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.0001
    num_epochs = 1  # 10 epochs per model
    num_densenet_models = 4   # Number of DenseNet ensemble members
    num_deit_models = 2       # Number of DeiT ensemble members
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING ENSEMBLE: 4 DENSENET169 + 2 DeiT MODELS")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Normalization: ZScale (astronomical standard)")
    print(f"DenseNet models: {num_densenet_models}")
    print(f"DeiT models: {num_deit_models}")
    print(f"Total ensemble size: {num_densenet_models + num_deit_models} models")
    print(f"Epochs per model: {num_epochs}")
    print(f"{'='*70}\n")

    # Paths
    input_folder = 'pos_and_neg/'
    output_folder = 'split_folders/'
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        return
    
    # Split the dataset
    print("📂 Splitting dataset into train/val/test...")
    if not os.path.exists(output_folder):
        splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1))
    
    # No data transforms for astronomical data - preserve original pixel values
    train_transform = None
    
    # Create datasets
    train_dataset = FITSDataset(root_dir=f"{output_folder}/train", transform=train_transform)
    val_dataset = FITSDataset(root_dir=f"{output_folder}/val", transform=None)
    test_dataset = FITSDataset(root_dir=f"{output_folder}/test", transform=None)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # Plot sample images
    print("\n🖼️  Plotting sample images...")
    plot_sample_images(train_loader)
    
    # Train ensemble
    best_val_accs = []
    model_types = []
    
    # ========================================================================
    # PART 1: Train DenseNet169 Models
    # ========================================================================
    
    for model_idx in range(num_densenet_models):
        print(f"\n{'='*70}")
        print(f"🔥 Training DenseNet169 Model {model_idx + 1}/{num_densenet_models}")
        print(f"{'='*70}\n")
        
        # Create a fresh DenseNet169 model
        model = densenet169(num_classes=1)
        model_name = f"DenseNet169_Ensemble_Model{model_idx + 1}"
        
        model = model.to(device)
        
        # Create optimizer and scheduler for this model
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Train this ensemble member
        best_val_acc = train_single_model(
            model, model_name, train_loader, val_loader, 
            criterion, optimizer, scheduler, device, num_epochs
        )
        best_val_accs.append(best_val_acc)
        model_types.append('DenseNet169')
        
        # Clear GPU memory
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        print(f"\n✓ Completed training {model_name}")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%\n")
    
    # ========================================================================
    # PART 2: Train DeiT Models
    # ========================================================================
    
    for model_idx in range(num_deit_models):
        print(f"\n{'='*70}")
        print(f"🔥 Training DeiT Model {model_idx + 1}/{num_deit_models}")
        print(f"{'='*70}\n")
        
        # Create a fresh DeiT model
        model = create_deit(num_classes=1, pretrained=False)
        model_name = f"DeiT_Ensemble_Model{model_idx + 1}"
        
        model = model.to(device)
        
        # Create optimizer and scheduler for this model
        # Note: Transformers often benefit from lower learning rates
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Train this ensemble member
        best_val_acc = train_single_model(
            model, model_name, train_loader, val_loader, 
            criterion, optimizer, scheduler, device, num_epochs
        )
        best_val_accs.append(best_val_acc)
        model_types.append('DeiT')
        
        # Clear GPU memory
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        print(f"\n✓ Completed training {model_name}")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%\n")
    
    # ========================================================================
    # Print Final Results
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"🎉 ENSEMBLE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📈 Individual Ensemble Member Best Validation Accuracies:")
    
    for i, (model_type, acc) in enumerate(zip(model_types, best_val_accs)):
        print(f"  Model {i+1} ({model_type}): {acc:.2f}%")
    
    # Calculate statistics
    avg_acc = np.mean(best_val_accs)
    densenet_accs = [acc for acc, mtype in zip(best_val_accs, model_types) if mtype == 'DenseNet169']
    deit_accs = [acc for acc, mtype in zip(best_val_accs, model_types) if mtype == 'DeiT']
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Average Validation Accuracy: {avg_acc:.2f}%")
    print(f"  Best Individual Model: {max(best_val_accs):.2f}%")
    print(f"  Worst Individual Model: {min(best_val_accs):.2f}%")
    print(f"  Std Dev: {np.std(best_val_accs):.2f}%")
    
    print(f"\n📊 By Model Type:")
    print(f"  DenseNet169 Average: {np.mean(densenet_accs):.2f}%")
    print(f"  DeiT Average: {np.mean(deit_accs):.2f}%")
    
    print(f"\n{'='*70}")
    print(f"✓ Ensemble models trained with ZScale normalization")
    print(f"✓ Model files saved as:")
    for i in range(num_densenet_models):
        print(f"  - DenseNet169_Ensemble_Model{i+1}_best.pth")
    for i in range(num_deit_models):
        print(f"  - DeiT_Ensemble_Model{i+1}_best.pth")
    print(f"\n✓ Use ensemble classification script for inference")
    print(f"✓ Make sure to install: pip install timm")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()