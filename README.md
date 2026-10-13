# Pneumonia Diagnosis, Detection and Localization

UE24CS352A Machine Learning mini-project (Team 34).
Weakly supervised pneumonia localization on chest X-rays: a CNN classifies whether an X-ray shows pneumonia, and a Class Activation Map (CAM) localizes the opacity without bounding-box labels. Based on *Weakly Supervised Pneumonia Localization* (Huang, Monam, Cortes).

**Team:** SIRI KRISHNA CR (classification), SANKET SHIVAJI RATHOD (localization and demo)

## Repository structure

```
├── README.md
├── requirements.txt
├── classification/     # data preprocessing, baselines, CNN training and results
└── localization/       # CAM, heatmap, DFS clustering, bounding boxes, IoU, demo
```

## Setup

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

A GPU is recommended for training (we used a Kaggle GPU notebook).

## Dataset

Kaggle **RSNA Pneumonia Detection Challenge** (chest X-ray DICOM images):
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

The data is not included in this repo. Join the competition on Kaggle and download it. The scripts expect:

```
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_train_images/<patientId>.dcm
```

We keep only the "Lung Opacity" (pneumonia) and "Normal" (healthy) classes, resize images to 128x128, normalize per pixel, and split 70/20/10 into train/validation/test (stratified).

## Part 1: Classification (`classification/`)

Run in this order (from inside `classification/`):

```bash
python preprocess.py --data_dir /path/to/rsna --out data/rsna_128.npz   # add --prefix stage_1 if needed
python baselines.py  --data data/rsna_128.npz --size 64                 # LR, SVM, Random Forest
python train.py      --data data/rsna_128.npz --epochs 20 --lr 1e-4     # trains the CNN
```

`part1_classification.ipynb` runs the whole pipeline in a single Kaggle notebook.

**Model:** 10 convolutional layers (3x3, ReLU, zero padding, BatchNorm), then Global Average Pooling and one fully connected layer. This structure is required for CAM. Trained with Adam (lr 0.0001) for 20 epochs. The best-validation checkpoint is saved as `cnn_best.pt`.

**Outputs:** `cnn_best.pt`, `training_curves.png`, `test_report.json`, `baseline_results.json`.

### Results (test set)

| Model | Test accuracy |
|---|---|
| Logistic Regression | 81.6% |
| SVM | 87.8% |
| Random Forest | 87.5% |
| **CNN (ours)** | **91.7%** |

CNN confusion matrix: TP 548, FN 54, FP 69, TN 816. Sensitivity 91.0%, specificity 92.2%, precision 88.8%.

## Part 2: Localization (`localization/`)
### Methodology

1. **CAM Generation:** Generates a heatmap highlighting image regions contributing to the CNN's pneumonia prediction.
2. **Heatmap Processing:** Normalizes the heatmap and applies a relative activation threshold.
3. **Region Clustering:** Groups connected activated pixels into candidate regions.
4. **Bounding Box Generation:** Converts activated regions into predicted bounding boxes.
5. **Peak-Based Localization:** Identifies prominent local maxima in the heatmap to generate candidate boxes.
6. **Evaluation:** Compares predicted boxes with ground-truth annotations using Intersection over Union (IoU), precision, recall, and F1-score.


## Demo
The demo displays the original chest X-ray, CAM heatmap, and localization overlay. Red boxes indicate predicted regions, while green dashed boxes indicate ground-truth annotations.

**Peak-based localization:**

```bash
python localization/demo.py --method peak --fraction 0.70 --peak-fraction 0.90 --save localization/demo_result.png --no-show
```


### Validation Results

Results from a separate exploratory validation sample of 100 annotated images, using an IoU threshold of 0.50.

| Metric | Original CAM | Peak-Based CAM |
|---|---:|---:|
| True Positives (TP) | 10 | 16 |
| False Positives (FP) | 136 | 128 |
| False Negatives (FN) | 149 | 143 |
| Precision | 6.85% | 11.11% |
| Recall | 6.29% | 10.06% |
| F1-score | 6.56% | 10.56% |
| Mean matched IoU | 0.5608 | 0.5855 |

The peak-based method improved the measured localization metrics on this validation sample. However, precision and recall remain low. Mean matched IoU is calculated over matched boxes and is not classification accuracy.

### Limitations

CAM highlights regions associated with the model's prediction but does not guarantee exact lesion boundaries. Predicted boxes may miss annotated regions or include irrelevant areas. These results are exploratory, and the system is not a clinically validated diagnostic tool.

## Reference

Huang, Monam, Cortes. *Weakly Supervised Pneumonia Localization.* Zhou et al., *Learning Deep Features for Discriminative Localization* (CAM), CVPR 2016.
