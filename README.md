# Generalized Nuclei Instance Segmentation

A comprehensive comparison of deep learning architectures for nuclei instance segmentation in histopathology images, with novel approaches for handling vague/ambiguous regions.

## Overview

This project implements and evaluates multiple state-of-the-art nuclei instance segmentation pipelines on the **NuInsSeg dataset**, focusing on:
- Reproducing and analyzing published baseline methods
- Investigating the impact of post-processing techniques on segmentation quality
- Developing vague-region-aware architectures to handle ambiguous annotations

## Motivation

Accurate nuclei segmentation is crucial for computational pathology and disease diagnosis. However, real-world histopathology images often contain:
- Overlapping nuclei with unclear boundaries
- Vague/ambiguous regions where annotation is uncertain
- High variability across different tissue types

This project addresses these challenges through systematic evaluation of architectures and loss functions designed to handle annotation ambiguity.

## Implemented Architectures

### 1. Shallow U-Net
- Baseline segmentation architecture with 5 encoding/decoding levels
- Binary segmentation with optional watershed post-processing
- Reproduces published baseline results on NuInsSeg dataset

### 2. Dual-Decoder U-Net (DDUNet)
- Novel architecture with two parallel decoders
- Simultaneous prediction of binary masks and distance maps
- Enhanced boundary localization through multi-task learning

### 3. Mask R-CNN
- **Standard Mask R-CNN**: Pretrained ResNet-50 backbone with FPN
- **Vague-Region-Aware Mask R-CNN**: Novel contribution that applies **spatially-aware loss weighting** by using RoI-aligned vague masks to reduce loss contribution in ambiguous regions (weight 0.7 for vague areas vs 1.0 for clear regions)

## Dataset

**NuInsSeg (Nuclei Instance Segmentation Dataset)**
- Multi-organ histopathology images
- Includes:
  - Tissue images (RGB)
  - Binary masks
  - Distance transform maps
  - Instance label masks
  - **Vague region annotations** (ambiguous areas)
- Organized by tissue/organ type

## Key Findings

### Post-Processing Analysis
Evaluated watershed-based post-processing on shallow U-Net predictions:
- ✅ **Marginal improvements** in Aggregate Jaccard Index (AJI) for watershed variants
- ❌ **Negative impact** on Dice coefficient and Panoptic Quality (PQ)
- **Root cause**: Over-segmentation of nuclei, creating false instance boundaries

### Vague-Region Handling
Designed **spatially-aware loss weighting** strategy for Mask R-CNN:
- Uses RoI Align to crop vague region masks corresponding to each detected nuclei proposal
- Applies per-pixel weighting: vague regions (weight 0.7) vs clear regions (weight 1.0)
- Handles coordinate scaling between original vague masks and resized image proposals
- **Slight improvements** in AJI and PQ metrics
- Better handling of uncertain annotations compared to standard baseline

## Evaluation Metrics

Three complementary metrics used for comprehensive evaluation:

1. **Dice Coefficient**: Measures pixel-wise overlap (sensitive to segmentation quality)
2. **Aggregate Jaccard Index (AJI)**: Evaluates instance-level segmentation accuracy
3. **Panoptic Quality (PQ)**: Combines detection and segmentation quality

## Project Structure

```
.
├── base-shallow-unet.ipynb                          # Shallow U-Net baseline
├── ddunet-complete.ipynb                            # Dual-decoder U-Net implementation
├── dualdecoderunet.ipynb                            # DDUNet experiments
├── maskrcnn.ipynb                                   # Standard Mask R-CNN
├── best-maskrcnn-with-vague-region-handling.ipynb   # Vague-aware Mask R-CNN
├── postprocessing.ipynb                             # Post-processing experiments
├── eda.ipynb                                        # Exploratory data analysis
├── Models/
│   └── DDUnet/                                      # Trained model weights
└── README.md
```

## Technical Implementation

### Computational Environment
- **Platform**: Kaggle Notebooks
- **GPU**: NVIDIA Tesla P100 (16GB)
- All experiments conducted in Kaggle's cloud environment for reproducibility

### Frameworks & Libraries
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Image Processing**: OpenCV, scikit-image
- **Data Augmentation**: Albumentations
- **Evaluation**: Custom implementations of AJI, PQ metrics

### Training Strategy
- **5-fold cross-validation** for robust evaluation
- Data augmentation: random crops, flips, rotations, brightness/contrast adjustment
- Loss function: Combined binary cross-entropy and Dice loss
- Learning rate scheduling with step decay

### Key Techniques
1. **Distance Transform Prediction**: Helps separate touching nuclei
2. **Watershed Post-processing**: Instance separation using predicted distance maps
3. **Spatially-Aware Vague Region Weighting**: Uses RoI Align to apply per-pixel loss weights based on annotation uncertainty (0.7 for vague, 1.0 for clear)
4. **Small Object Removal**: Filters false positive predictions (min_size=50 pixels)

## Results Summary

| Model | Post-Processing | Dice (%) | AJI (%) | PQ (%) |
|-------|----------------|----------|---------|--------|
| Shallow U-Net | None | Baseline | Baseline | Baseline |
| Shallow U-Net | Watershed | Decreased | Marginal increase | Decreased |
| Shallow U-Net | Watershed (w/o vague) | Decreased | Marginal increase | Decreased |
| Mask R-CNN | Standard | Strong performance | Strong performance | Strong performance |
| Mask R-CNN | Vague-aware | Similar | Slight improvement | Slight improvement |

*Note: Over-segmentation in watershed variants negatively impacts Dice and PQ despite minor AJI gains. The vague-region-aware Mask R-CNN shows modest improvements in instance-level metrics (AJI, PQ) while maintaining comparable pixel-level accuracy (Dice).*

## Setup & Usage

### Prerequisites

**Note**: All notebooks were developed and tested on Kaggle with NVIDIA Tesla P100 GPU.

```bash
pip install tensorflow keras torch torchvision
pip install opencv-python scikit-image albumentations
pip install numpy pandas matplotlib tqdm
```

### Running Experiments

**Recommended**: Run notebooks on Kaggle or similar GPU-enabled platform (P100 or better).

1. **Shallow U-Net Baseline**:
   ```bash
   jupyter notebook base-shallow-unet.ipynb
   ```

2. **Dual-Decoder U-Net**:
   ```bash
   jupyter notebook ddunet-complete.ipynb
   ```

3. **Vague-Region-Aware Mask R-CNN**:
   ```bash
   jupyter notebook best-maskrcnn-with-vague-region-handling.ipynb
   ```

### Configuration
Key hyperparameters can be adjusted in the notebooks:
- `epoch_num`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 16)
- `k_fold`: Number of cross-validation folds (default: 5)
- `crop_size`: Input patch size (default: 512)
- `init_LR`: Initial learning rate (default: 0.001)
- `threshold`: Binary prediction threshold (default: 0.5)

## Key Contributions

1. **Systematic reproduction** of published shallow U-Net baselines with detailed analysis
2. **Empirical evidence** that watershed post-processing can harm overall segmentation quality despite marginal AJI improvements
3. **Novel spatially-aware loss weighting** for Mask R-CNN using RoI-aligned vague masks to handle ambiguous annotations at the pixel level
4. **Comprehensive evaluation** using multiple complementary metrics (Dice, AJI, PQ)

## Future Work

### Architecture Improvements
- Extend vague-region handling to U-Net architectures
- Investigate attention mechanisms for boundary refinement
- Explore self-supervised pre-training on unlabeled histopathology data
- Integrate transformer-based architectures for global context modeling

### Evaluation & Robustness
- **Extended cross-validation**: Perform additional fold evaluations for more statistically robust performance estimates
- **Multi-dataset evaluation**: Test on external datasets (MoNuSeg, PanNuke, CoNIC) to assess generalization
- **Statistical significance testing**: Apply paired t-tests and Wilcoxon signed-rank tests across folds
- **Per-organ analysis**: Report metrics stratified by tissue type to identify organ-specific performance patterns
- **Confidence intervals**: Calculate 95% confidence intervals for all reported metrics
- **Ablation studies**: Systematic analysis of individual components (augmentation, loss functions, post-processing)

### Clinical Application
- Test generalization across different staining protocols (H&E, IHC)
- Evaluate on clinical datasets with diverse imaging conditions
- Investigate failure modes and edge cases in rare tissue types

## References

### Datasets
- **NuInsSeg**: Kumar, N., et al. "A Multi-Organ Nucleus Segmentation Challenge." IEEE Transactions on Medical Imaging (2020).
- **MoNuSeg**: Kumar, N., et al. "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology." IEEE TMI (2017).

### Architectures
- **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI (2015).
- **Mask R-CNN**: He, K., Gkioxari, G., Dollár, P., & Girshick, R. "Mask R-CNN." ICCV (2017).

### Post-Processing & Evaluation
- **Evaluation Metrics**: Kumar, N., et al. "A Multi-Organ Nucleus Segmentation Challenge." IEEE TMI (2020). [AJI, PQ metrics]
- **Panoptic Quality**: Kirillov, A., et al. "Panoptic Segmentation." CVPR (2019).

### Related Work
- Graham, S., et al. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis (2019).

## License

This project is intended for research and educational purposes.

---

**Author**: Nishal Ahmed P. K.  
**Institution**: Birla Institute of Technology and Science, Pilani – Dubai

