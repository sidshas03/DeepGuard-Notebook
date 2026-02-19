# DeepGuard: Deepfake Face Image Detector

A Jupyter notebook that trains an **EfficientNet-B0** based binary classifier to distinguish real human face photographs from AI-generated/deepfake faces, with **explainable forensic analysis** for each prediction.

## Features

- **Transfer learning** from ImageNet-pretrained EfficientNet-B0
- **Two-phase training**: frozen backbone warm-up, then full fine-tuning with early stopping
- **Explainable analysis**: Grad-CAM heatmaps, frequency spectrum (FFT), and forensic evidence
- **Fused detection**: combines CNN predictions with handcrafted forensic cues (noise uniformity, texture complexity, edge gradients, symmetry)
- Supports **CUDA**, **MPS** (Apple Silicon), and **CPU**

## Example Output

### Full Forensic Analysis

![Forensic analysis output](notebook_outputs/prediction_output_main_case.png)

### Grad-CAM (Model Attention)

![Grad-CAM heatmap](notebook_outputs/case_gradcam.png)

### Frequency Spectrum (FFT)

![Frequency spectrum](notebook_outputs/case_fft_spectrum.png)

### Detection Scores

![Detection scores](notebook_outputs/case_detection_scores.png)

### Input Image & Forensic Report

![Predicted face](notebook_outputs/case_predicted_face.png)
![Forensic report text](notebook_outputs/case_forensic_report_text.png)

In this example, the model classifies an AI-generated image as **FAKE** (66.5% fused verdict). The forensic analysis explains why:
- Noise is very uniform across the image — characteristic of AI-generated content
- Edge gradients are unusually uniform — real faces show more variation across eyes, nose, jawline

## Training Outputs

![Training loss and accuracy](notebook_outputs/training_loss_accuracy.png)
![Validation AUC, F1, and learning rate](notebook_outputs/val_auc_f1_lr_schedule.png)
![ROC curve and confusion matrix](notebook_outputs/roc_curve_confusion_matrix.png)
![Precision-recall curve](notebook_outputs/precision_recall_curve.png)
![Sample dataset images](notebook_outputs/sample_images.png)
![Sample predictions](notebook_outputs/sample_predictions.png)

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch, torchvision, timm, scikit-learn, matplotlib, seaborn, Pillow, ipywidgets, opencv-python

### Setup

pip install torch torchvision timm scikit-learn seaborn matplotlib pillow ipywidgets opencv-python
