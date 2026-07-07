# Training Detection Transformer

This guide provides instructions for training DETR (Detection Transformer) models on custom dataset.

## Training Detection Transformer 

```bash
PYTHONPATH=. python pipelines/training_pipeline.py \
    --train_dir /path/to/dataset \
    --val_dir /path/to/dataset \
    --test_dir /path/to/dataset \
    --train_annotations dataset/annotations/training_set_f.json \
    --val_annotations dataset/annotations/validation_set_f.json \
    --test_annotations dataset/annotations/testing_set_f.json \
    --log_dir results \
    --epochs 30 \
    --batch_size 2 \
    --val_batch_size 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4
```

## Parameters

| Parameter             | Description                  | Value                                                             |
|-----------------------|------------------------------|-------------------------------------------------------------------|
| `--train_dir`         | Training dataset directory   | `/path/to/dataset` |
| `--val_dir`           | Validation dataset directory | `/path/to/dataset` |
| `--test_dir`          | Test dataset directory       | `/path/to/dataset` |
| `--train_annotations` | Training annotations file    | `training_set_f.json`                                             |
| `--val_annotations`   | Validation annotations file  | `validation_set_f.json`                                           |
| `--test_annotations`  | Test annotations file        | `testing_set_f.json`                                              |
| `--log_dir`           | Results output directory     | `BikeSafeAI/results`                                              |
| `--epochs`            | Number of training epochs    | `30`                                                              |
| `--batch_size`        | Training batch size          | `2`                                                               |
| `--val_batch_size`    | Validation batch size        | `4`                                                               |
| `--lr`                | Learning rate                | `1e-4`                                                            |
| `--lr_backbone`       | Backbone learning rate       | `1e-5`                                                            |
| `--weight_decay`      | Weight decay coefficient     | `1e-4`                                                            |



# BikeSafeAI Inference

## Overview

This repository contains inference scripts and evaluation metrics for the BikeSafeAI project, focusing on bicycle detection, tracking, and speed estimation using DETR (DEtection TRansformer) combined with DeepSort tracking.

## Demo: Cyclists Counting and Speed Estimation

### Objective
Compare the performance of different feature extractors integrated with DeepSort for object tracking in bicycle safety scenarios.

### Feature Extractors Tested
1. **CLIP (ViT-B-32)** - Vision Transformer based feature extraction
2. **MobileNet** - Lightweight CNN for mobile deployment
3. **OSNet** - Omni-Scale Network for person re-identification

### Model Variants
- **Base Models**: Trained with baseline configuration (epoch=30)

## Inference Scripts

### Base Models

#### 1. DETR + CLIP
```bash
PYTHONPATH=. python scripts/run_inference.py \
    --frames-dir "/path/to/dataset/test" \
    --annotations-list "dataset/artifacts/test_images_with_annotations_f.txt" \
    --checkpoint "dataset/models/baseline/checkpoints/epoch=30-step=18879.ckpt" \
    --embedder "clip_ViT-B/32" \
    --embedder-weights "dataset/artifacts/ViT-B-32.pt" \
    --output-video "results/tracking_base_detr_mobilenet.mp4" \
    --output-file "results/predictions_DETR_deepsort_mobilenet_working.txt" \
    --pred-csv "results/predicted_base_mobilenet_speeds.csv" \
    --coords "649,748,1083,775,1127,239,1020,242" \
    --fps 10 \
    --progress-interval 50
```

#### 2. DETR + MobileNet
```bash
PYTHONPATH=. python scripts/run_inference.py \
    --frames-dir "/path/to/dataset/test" \
    --annotations-list "dataset/artifacts/test_images_with_annotations_f.txt" \
    --checkpoint "dataset/models/baseline/checkpoints/epoch=30-step=18879.ckpt" \
    --embedder "mobilenet" \
    --embedder-weights "" \
    --output-video "results/tracking_base_detr_mobilenet.mp4" \
    --output-file "results/predictions_DETR_deepsort_mobilenet_working.txt" \
    --pred-csv "results/predicted_base_mobilenet_speeds.csv" \
    --coords "649,748,1083,775,1127,239,1020,242" \
    --fps 10 \
    --progress-interval 50
```

#### 3. DETR + OSNet
```bash
PYTHONPATH=. python scripts/run_inference.py \
    --frames-dir "/path/to/dataset/test" \
    --annotations-list "dataset/artifacts/test_images_with_annotations_f.txt" \
    --checkpoint "dataset/models/baseline/checkpoints/epoch=30-step=18879.ckpt" \
    --embedder "torchreid" \
    --embedder-weights "" \
    --output-video "results/tracking_base_detr_osnet.mp4" \
    --output-file "results/predictions_DETR_deepsort_osnet_working.txt" \
    --pred-csv "results/predicted_base_osnet_speeds.csv" \
    --coords "649,748,1083,775,1127,239,1020,242" \
    --fps 10 \
    --progress-interval 50
```


## Experiment: Estimating Speeds of Cyclists

### Objective
To estimate the speeds of each individual cyclist tracked within the region of interest, simply run the same inference scripts used in the previous experiment.

> 💡 If the previous experiment has already been executed, you do not need to re-run any scripts for this experiment.

---

### Ground Truth Speeds Generation
Ground truth speeds are calculated only within a region whose real-world dimensions are known. The speed is computed based on pixel displacement over time, then translated into meters per second.

---

### Ground Truth Speed Script

```bash
PYTHONPATH=. python pipelines/generate_gt_speed_pipeline.py \
    --frames-dir "/path/to/dataset/test" \
    --annotations-list "dataset/artifacts/test_images_with_annotations_f.txt" \
    --checkpoint "dataset/models/baseline/checkpoints/epoch=30-step=18879.ckpt" \
    --gt-output-csv "dataset/artifacts/gt_speeds_generated_from_pipeline.csv" \
    --output-video "results/gt_speed_visualization_video.mp4" \
    --polygon-coords "937,390,1100,388,1003,1080,469,1079" \
    --embedder "mobilenet" \
    --fps 5 \
    --progress-interval 50

```
### Predicted Speed Script

```bash
PYTHONPATH=. python pipelines/generate_homography_speed_pipeline.py \
    --frames-dir "/path/to/dataset/test" \
    --annotations-list "dataset/artifacts/test_images_with_annotations_f.txt" \
    --checkpoint "dataset/models/baseline/checkpoints/epoch=30-step=18879.ckpt" \
    --pred-output-csv "results/predicted_homography_speeds.csv" \
    --output-video "results/homography_speed_visualization_video.mp4" \
    --homography-src-ref "937.30,390.20,1100.30,388.70,1084.60,506.20,862.40,504.00" \
    --homography-dst-ref "0,0,4,0,4,10.5,0,10.5" \
    --polygon-coords "937,390,1100,388,1003,1080,469,1079" \
    --embedder "mobilenet" \
    --fps 5 \
    --progress-interval 50
```


## Experiment : Counting Cyclists

### Objective
Count the number of cyclists passing through a defined region of interest using DETR detection capabilities.

### Implementation
The cyclist counting experiment uses a single configuration optimized for detection and counting rather than tracking individual objects over time.

If the previous experiment has already been executed, you do not need to re-run any scripts for this experiment.

The counts can be visualized in the generated outputs !

## Command Line Arguments

### Required Arguments
- `--frames-dir`: Directory containing test frame images
- `--annotations-test`: Path to test annotations JSON file
- `--annotations-list`: List of test images with annotations
- `--checkpoint`: Model checkpoint file path
- `--gt-csv`: Ground truth CSV file with speed data
- `--output-video`: Output video file path
- `--output-file`: Output text file for tracking results
- `--pred-csv`: Output CSV file for predicted speeds
- `--predictions`: Output predictions file
- `--coords`: Coordinate string for region of interest (format: "x1,y1,x2,y2,x3,y3,x4,y4")

### Optional Arguments
- `--embdr-wt`: Pre-trained embedding weights (required for CLIP and OSNet)
- `--fps`: Frames per second for output video (default: 10)
- `--progress-interval`: Progress reporting interval (default: 50)

## Output Files

### Generated Outputs
1. **Tracking Videos**: Visual representation of detection and tracking results
   - `tracking_base_detr_[extractor].mp4`

2. **Tracking Results**: Detailed tracking information in text format
   - `tracking_results_base_[extractor].txt`

3. **Speed Predictions**: CSV files containing predicted speeds
   - `predicted_base_[extractor]_speeds.csv`

4. **Prediction Files**: Complete prediction results
   - `predictions_DETR_deepsort_base_[extractor]_working.txt`

### Comparison Matrix

| Model | Feature Extractor | Checkpoint | Expected Performance                   |
|-------|-------------------|------------|----------------------------------------|
| Base  | CLIP              | epoch=30   | High accuracy, slower inference        |
| Base  | MobileNet         | epoch=30   | Balanced performance, faster inference |
| Base  | OSNet             | epoch=30   | Good re-identification, moderate speed |


## Usage Instructions

1. **Prerequisites**: Ensure all dependencies are installed and dataset is properly structured
2. **Environment**: Set up the BikeSafeAI environment with required packages
3. **Execution**: Run scripts sequentially or in parallel based on computational resources
4. **Analysis**: Compare output files to evaluate relative performance

## Notes
- Coordinate parameter represents the region of interest for tracking
- Progress interval controls how frequently status updates are displayed
- All models use the same test dataset for fair comparison
- FPS setting affects output video quality and file size
