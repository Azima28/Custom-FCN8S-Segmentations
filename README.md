```markdown
# FCN-8s Image Segmentation

A complete PyTorch implementation of FCN-8s (Fully Convolutional Network) architecture for semantic image segmentation using VGG16 backbone with automatic dataset analysis and flexible mask format support.

## Features

- 🎯 **FCN-8s Architecture** - Skip connections from pool3, pool4, and pool5 for precise segmentation
- 🔧 **VGG16 Backbone** - Pre-trained VGG16 encoder for transfer learning
- 🔍 **Automatic Dataset Analysis** - Detects number of classes and mask format automatically
- 🎨 **Flexible Mask Support** - Handles both RGB and grayscale masks with auto value remapping
- 🚀 **GPU Acceleration** - CUDA support with automatic CPU fallback
- 📊 **Training Visualization** - Real-time progress with loss and accuracy metrics
- 📈 **Performance Metrics** - IoU (Intersection over Union) calculation
- 🖼️ **Easy Prediction** - Simple inference pipeline with overlay visualization
- ✨ **Auto-Detection** - Automatically detects number of classes from saved models

## Architecture

FCN-8s (Fully Convolutional Network with 8x upsampling) consists of:

```
Input Image (256x256)
    ↓
VGG16 Encoder:
    → Pool3 (1/8 scale) ─────────────┐
    → Pool4 (1/16 scale) ──────┐     │
    → Pool5 (1/32 scale)       │     │
         ↓                     │     │
    FC6 → FC7 (4096 channels)  │     │
         ↓                     │     │
    Score FR                   │     │
         ↓                     │     │
    Upsample 2x ───────────────┘     │
         ↓                           │
    Fuse with Pool4                  │
         ↓                           │
    Upsample 2x ─────────────────────┘
         ↓
    Fuse with Pool3
         ↓
    Upsample 8x
         ↓
    Output Segmentation
```

**Key Components:**
- **Encoder**: VGG16 pre-trained on ImageNet
- **Skip Connections**: From pool3, pool4, pool5 for multi-scale features
- **Fully Convolutional**: FC layers converted to 1x1 convolutions
- **Upsampling**: Transposed convolutions for spatial resolution recovery

## Installation

```bash
pip install torch torchvision numpy matplotlib pillow opencv-python scikit-learn tqdm
```

## Dataset Structure

```
dataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── masks/
    ├── mask1.png
    ├── mask2.png
    └── ...
```

## Usage

### 1. Training

Configure your dataset paths in the main function:

```python
MASK_DIR = "/content/drive/MyDrive/datasets/masks"
IMAGE_DIR = "/content/drive/MyDrive/datasets/images"

# Hyperparameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 4  # FCN-8s uses more memory
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
```

Run training:
```python
if __name__ == "__main__":
    main()
```

### 2. Auto-Detect Number of Classes

Automatically detect classes from saved FCN-8s model:

```python
def detect_num_classes(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    for key in state_dict.keys():
        if "score_fr.weight" in key:
            return state_dict[key].shape[0]
    raise ValueError("Not a valid FCN-8s model")

NUM_CLASSES = detect_num_classes('/content/best_fcn8s_model.pth')
print(f"Detected: {NUM_CLASSES} classes")
```

### 3. Prediction

Predict on new images:

```python
FCN_MODEL_PATH = '/content/drive/MyDrive/best_fcn8s_model.pth'
FCN_IMAGE_PATH = '/content/drive/MyDrive/test_image.jpg'

prediction = predict_image_fcn8s(FCN_MODEL_PATH, FCN_IMAGE_PATH)
```

The function automatically detects the number of classes from the model.

## Key Features Explained

### FCN-8s vs Other Architectures

| Feature | FCN-8s | U-Net | FCN-32s |
|---------|--------|-------|---------|
| Skip Connections | 3 levels (pool3,4,5) | All levels | None |
| Upsampling | 8x at end | Progressive | 32x at end |
| Backbone | VGG16 | Custom | VGG16 |
| Parameters | ~134M | ~31M | ~134M |
| Best For | Transfer learning | Medical imaging | Quick baseline |

### Automatic Mask Value Remapping

Handles non-continuous mask values automatically:
- **Before**: `[0, 85, 170, 255]`
- **After**: `[0, 1, 2, 3]`

### Pre-trained VGG16 Backbone

- Uses ImageNet pre-trained weights
- Faster convergence than training from scratch
- Better feature extraction for natural images

## Model Architecture Details

```python
FCN8s(
  Encoder (VGG16):
    pool3: Conv layers 0-16  → 256 channels (1/8 scale)
    pool4: Conv layers 17-23 → 512 channels (1/16 scale)
    pool5: Conv layers 24-30 → 512 channels (1/32 scale)
  
  Fully Convolutional:
    fc6: Conv2d(512 → 4096, kernel=7x7)
    fc7: Conv2d(4096 → 4096, kernel=1x1)
  
  Scoring:
    score_fr: Conv2d(4096 → num_classes)
    score_pool4: Conv2d(512 → num_classes)
    score_pool3: Conv2d(256 → num_classes)
  
  Upsampling:
    upscore2: Upsample 2x (1/32 → 1/16)
    upscore_pool4: Upsample 2x (1/16 → 1/8)
    upscore8: Upsample 8x (1/8 → 1/1)
)
```

## Output Files

Training generates:
- `best_fcn8s_model.pth` - Best model weights based on validation loss
- `training_history_fcn8s.png` - Loss and accuracy curves
- `predictions_fcn8s.png` - Sample predictions on validation set

Prediction generates:
- `prediction_fcn8s_result.png` - Original, mask, and overlay visualization

## Performance Metrics

- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Mean IoU**: Average intersection over union across all classes
- **Class-wise Statistics**: Pixel count and percentage per class
- **Training/Validation Loss**: Cross-entropy loss tracking

## Training Output Example

```
============================================================
Epoch [10/50] Summary:
  Train Loss: 0.2847 | Train Acc: 91.23%
  Val Loss:   0.3521 | Val Acc:   88.45%
  Time: 52.3s | Elapsed: 8.7m | ETA: 34.9m
  ✅ Best model saved! (Val Loss: 0.3521, Val Acc: 88.45%)
  GPU Memory: 3248.75 MB
============================================================
```

## Memory Requirements

FCN-8s uses more memory than lighter models due to:
- VGG16 backbone (134M parameters)
- Multiple skip connections stored in memory
- Large feature maps from FC layers

**Recommended settings:**
- GPU: 6GB+ VRAM
- Batch Size: 4-8
- Image Size: 256×256 (512×512 requires more memory)

## Tips for Best Results

1. **Use Pre-trained Weights**: Keep `pretrained=True` for better initialization
2. **Smaller Batch Size**: FCN-8s needs more memory, use batch_size=4
3. **Learning Rate**: Start with 1e-4, use scheduler for fine-tuning
4. **Fine-tuning Strategy**: 
   - Freeze VGG16 layers initially
   - Train only decoder for first few epochs
   - Unfreeze all layers for end-to-end training
5. **Data Augmentation**: Use flips, rotations, color jittering

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solutions:**
- Reduce batch size to 2 or 4
- Use smaller image size (128×128)
- Enable gradient checkpointing
- Use mixed precision training (fp16)

### Issue: Slow convergence
**Solution:** 
- Ensure pretrained=True
- Check learning rate (try 1e-3 to 1e-5)
- Verify mask alignment

### Issue: Model not loading
**Solution:** 
- Check num_classes matches training
- Use `map_location='cpu'` for CPU inference
- Verify model was saved correctly

## Comparison with U-Net

| Aspect | FCN-8s | U-Net |
|--------|--------|-------|
| **Speed** | Faster (VGG16 optimized) | Moderate |
| **Memory** | High (~3GB) | Lower (~2GB) |
| **Accuracy** | Good for natural images | Better for medical |
| **Training** | Transfer learning ready | Train from scratch |
| **Use Case** | General segmentation | Medical/precise edges |

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision (for VGG16 pre-trained weights)
- CUDA 10.2+ (optional, for GPU acceleration)
- 8GB+ RAM
- GPU with 6GB+ VRAM recommended

## Advanced Usage

### Fine-tuning Pre-trained Model

```python
model = FCN8s(num_classes=num_classes, pretrained=True)

# Freeze VGG16 encoder
for param in model.pool3.parameters():
    param.requires_grad = False
for param in model.pool4.parameters():
    param.requires_grad = False
for param in model.pool5.parameters():
    param.requires_grad = False

# Train only decoder
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3
)
```

### Custom Loss Function

```python
# Weighted Cross Entropy for imbalanced classes
class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0])  # Adjust per class
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3431--3440},
  year={2015}
}
```

## References

- Original FCN Paper: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- VGG16 Paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- PyTorch Documentation: [torchvision.models.vgg16](https://pytorch.org/vision/stable/models.html)

## License

MIT License

## Acknowledgments

- Original FCN implementation by Jonathan Long, Evan Shelhamer, and Trevor Darrell
- VGG16 architecture by Visual Geometry Group, Oxford
- PyTorch implementation inspired by the deep learning community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.


