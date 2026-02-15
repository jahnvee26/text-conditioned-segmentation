# Text-Conditioned Segmentation for Construction Defect Detection

A deep learning pipeline for segmenting cracks and drywall taping areas using text prompts. Built with PyTorch and trained on Roboflow datasets.

## Project Overview

This project trains a text-conditioned segmentation model that can:
- **Segment cracks** when prompted with "segment crack"
- **Segment taping areas** when prompted with "segment taping area"

The model uses a UNet architecture with FiLM (Feature-wise Linear Modulation) layers to condition image features on text embeddings.

##  Project Structure

```
├── coco_to_masks.ipynb                          # Dataset preparation
├── text_conditioned_segmentation_pipeline.ipynb # Model training & inference
├── regenerate_masks_correctly.py                # Mask generation script
├── final_dataset/                               # Processed dataset
│   ├── train/
│   │   ├── images/                             # 5,984 training images
│   │   └── masks/                              # 5,984 binary masks
│   └── valid/
│       ├── images/                             # 403 validation images
│       └── masks/                              # 403 binary masks
├── best_text_seg_model.pth                     # Trained model weights
├── tokenizer.pkl                               # Text tokenizer
└── model_config.pkl                            # Model configuration
```

## 1. Dataset Preparation (`coco_to_masks.ipynb`)

### Purpose
Converts COCO format annotations to binary segmentation masks with proper Roboflow augmentation handling.

### Key Steps
1. **Install dependencies**: `opencv-python`, `pillow`, `tqdm`
2. **Generate masks**: Uses polygon segmentation (not bounding boxes) for accurate shapes
3. **Preserve Roboflow IDs**: Keeps full filenames (e.g., `00002_jpg.rf.3119f...`) to match augmented images
4. **Merge datasets**: Combines cracks and drywall datasets into `final_dataset/`

### Output Format
- **Mask files**: `{image_id}__segment_crack.png` or `{image_id}__segment_taping_area.png`
- **Values**: 0 (background), 255 (segmented region)
- **Format**: Single-channel PNG

##  2. Model Training (`text_conditioned_segmentation_final.ipynb`)

### Architecture
```
Input Image (256×256) → ImageEncoder (UNet) → f1, f2, f3
Input Text → TextEncoder (Embedding) → text_vec
                    ↓
            FiLM(f3, text_vec) → modulated features
                    ↓
            Decoder (UNet) → Segmentation Mask (256×256)
```

**Components**:
- **ImageEncoder**: 3-level UNet encoder (32→64→128 channels)
- **TextEncoder**: Word embeddings + FC layer (vocab_size=6)
- **FiLM Layer**: Modulates image features with text embeddings
- **Decoder**: UNet decoder with skip connections
- **Total Parameters**: 264,497 trainable

### Training Configuration
```python
Optimizer: Adam (lr=1e-3, reduced to 5e-4 at epoch 19)
Loss: BCEWithLogitsLoss
Batch Size: 16
Epochs: 20
Early Stopping: Patience 10
Device: CUDA (if available)
```

### Results (Baseline Model)
| Metric | Epoch 1 | Epoch 10 | Epoch 20 | Change |
|--------|---------|----------|----------|--------|
| Train Loss | 0.2626 | 0.1461 | 0.1331 | -49% |
| Val Loss | 0.2605 | 0.2072 | 0.2057 | -21% |
| **Best Val Loss** | - | - | **0.2057** | - |

**Mean IoU**: 0.2808 ± 0.1797  
**Overfitting Gap**: 0.0726 (7.3%)

 Model shows mild overfitting (train loss lower than validation loss)

### Positional Encoding Experiment

I tested adding XY coordinate channels to improve taping area detection:

**Hypothesis**: Adding spatial position information (5-channel input: RGB+XY) could help the model better localize taping areas.

**Results**: 
- Validation Loss: 0.2222 (vs 0.2057 baseline) - **8% worse**
- Overfitting Gap: 0.0804 (vs 0.0726 baseline) - **11% more overfitting**
- Mixed IoU changes: 2 images improved, 2 degraded

**Conclusion**:  Positional encoding did not improve performance. **Baseline 3-channel RGB model is recommended.**

##  Quick Start

### Training from Scratch
```bash
# 1. Prepare dataset
jupyter notebook coco_to_masks.ipynb
# Run cells 2, 4, 5, 7

# 2. Train model
jupyter notebook text_conditioned_segmentation_final.ipynb
# Run cells 2-20 (20 epochs, ~1-2 hours on GPU)

# 3. Evaluate
# Run cell 26 to visualize predictions
```

### Inference on New Images
```python
import pickle
import torch
from PIL import Image

# Load model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = TextConditionedUNet()
model.text_encoder = TextEncoder(vocab_size=6, embed_dim=128)
model.load_state_dict(torch.load("best_text_seg_model.pth"))
model = model.to(device)
model.eval()

# Predict
mask = predict_mask(model, "image.jpg", "segment crack", tokenizer, device, "output.png")
```

## Datasets

**Source**: Roboflow (COCO format)
- `cracks.v1i.coco`: Crack detection dataset
- `Drywall-Join-Detect.v1i.coco`: Drywall taping area dataset

**Final Statistics**:
- Training: 5,984 samples (5,164 cracks + 820 drywall)
- Validation: 403 samples (201 cracks + 202 drywall)
- Prompts: "segment crack", "segment taping area"
- Vocabulary: 6 tokens (`<PAD>`, `<UNK>`, `segment`, `crack`, `taping`, `area`)

##  Key Features

1. **Text Conditioning**: Uses natural language prompts to control segmentation
2. **FiLM Modulation**: Conditions image features on text for better performance
3. **Polygon-Based Masks**: Accurate segmentation using COCO polygons (not bounding boxes)
4. **Augmentation Support**: Handles Roboflow augmented images correctly
5. **Portable**: Model + tokenizer saved for easy deployment


##  Dependencies

```bash
pip install torch torchvision opencv-python pillow tqdm matplotlib
```


**Status**:  Training complete (20 epochs) | Baseline model recommended | Best validation loss: 0.2057 | Mean IoU: 0.2808 ± 0.1797
