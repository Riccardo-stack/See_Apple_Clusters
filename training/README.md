# Training Pipeline

This directory contains the scripts for augmenting data, training the YOLO model, and exporting it to CoreML format.

## Overview
The training pipeline consists of three main steps:
1. **Data Augmentation**: Takes the original images and YOLO labels, applies augmentations, and prepares the dataset.
2. **Training**: Trains a YOLO model on the augmented dataset.
3. **Export**: Exports the trained weights to CoreML format for inference.

## Prerequisites
- Ensure all Python dependencies are installed.
- Python 3.12+ is required.

## Step-by-Step Instructions

**Note**: All scripts should be run from the project root directory.

### 1. Data Collection
Place your original photos in the `Immagini/` directory in the project root.

### 2. Labeling
Labels should be in YOLO format (`class x_center y_center w h`).
Place the label files in: `Database_apples/Apples_dataset/labels/train/`

### 3. Data Augmentation
To run the augmentation script:
```bash
python training/augment_data.py
```
This script applies 8 augmentation pipelines (e.g., lighting variations, geometric transforms, color/hue shifts, blur/focus simulation, weather simulation, noise/quality degradation, crop & zoom, and heavy/combined). It processes both images and bounding boxes simultaneously and outputs the final dataset to `dataset/`.

### 4. Training
To train the YOLO model:
```bash
python training/train.py
```
You can configure parameters such as `--data`, `--epochs`, `--batch`, `--device`, and `--project`. 
The default configuration will train `yolo26n.pt` and output the results to the `runs/` directory.

### 5. Export
To export the trained model to CoreML format:
```bash
python training/export_coreml.py --weights models/best.pt
```
This uses an isolated environment to bypass numpy 2.0 incompatibilities with `coremltools`.
