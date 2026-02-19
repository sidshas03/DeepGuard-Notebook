# DeepGuard: Deepfake Face Image Detector

A Jupyter notebook that trains an **EfficientNet-B0**–based binary classifier to distinguish real human face photographs from AI-generated/deepfake faces, with **explainable forensic analysis** for each prediction.

## Features

- **Transfer learning** from ImageNet-pretrained EfficientNet-B0
- **Two-phase training**: frozen backbone warm-up, then full fine-tuning with early stopping
- **Explainable analysis**: Grad-CAM heatmaps, frequency spectrum (FFT), and forensic evidence
- **Fused detection**: combines CNN predictions with handcrafted forensic cues (noise uniformity, texture complexity, edge gradients, symmetry)
- Supports **CUDA**, **MPS** (Apple Silicon), and **CPU**

## Output

For each uploaded image, the notebook provides:

| Component | Description |
|-----------|-------------|
| **Prediction** | Real vs AI-generated with confidence |
| **Grad-CAM** | Model attention heatmap on the face |
| **Frequency spectrum** | FFT magnitude visualization |
| **Detection scores** | CNN model and fused verdict |
| **Forensic report** | Evidence such as noise uniformity, edge gradient uniformity, texture detail |

### Example

Uploading an AI-generated image yields a verdict like:

- **VERDICT:** AI-GENERATED / DEEPFAKE  
- **Combined confidence:** 67% | CNN: 8% fake | Forensic: 0.55  
- **Evidence:**  
  1. Noise is very uniform across the image — real cameras produce spatially varying sensor noise  
  2. Unusually uniform edge gradients (CV: 0.95, real avg ~1.55) — real faces have varied edge strengths across eyes, nose, jawline  

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch, torchvision
- timm, scikit-learn, matplotlib, seaborn, Pillow, ipywidgets, opencv-python

### Setup

```bash
pip install torch torchvision timm scikit-learn seaborn matplotlib pillow ipywidgets opencv-python
```

### Dataset

The notebook downloads the DF40 deepfake face dataset (~588 MB) from HuggingFace on first run. It contains 3,212 images (1,606 real + 1,606 fake) from 40 deepfake generation methods.

### Run

1. Open `DeepGuard_Deepfake_Detector.ipynb` in Jupyter.
2. Run all cells to train the model (~13 min on Apple M2 Pro with MPS).
3. Use the upload widget at the end to analyze your own face images.

## Files

| File | Description |
|------|-------------|
| `DeepGuard_Deepfake_Detector.ipynb` | Main notebook (training + inference) |
| `notebook_outputs/` | Sample plots from a training run |

## License

MIT
