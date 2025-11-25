# Bottle Cap Color Detection  
**Real-time Edge-Deployable YOLOv11n Model for Bottle Cap Color Sorting**  
**Author:** John | **Date:** November 25, 2025  
**GitHub:** https://github.com/SpidieBot/bottlecap-color-detection  
**WandB Public Dashboard:** https://wandb.ai/spidiebot-personal/bottle_caps  

**CI/CD Status**  
[![CI/CD](https://github.com/SpidieBot/bottlecap-color-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/SpidieBot/bottlecap-color-detection/actions)

---

### Project Overview
This repository implements a **real-time bottle cap color detection system** using **Ultralytics YOLOv11n**, optimized for deployment on **Raspberry Pi 5** with **<10ms inference** using NCNN.

**Task:** Detect and classify bottle caps into 3 color classes:
- `light_blue`
- `dark_blue`
- `others` (green, red, yellow, etc.)

**Constraints Solved:**
- Tiny dataset (only 12 images) → Solved with heavy augmentation + transfer learning
- Edge deployment (RPi5, CPU-only) → NCNN export achieves **12 ms inference**
- Production readiness → Full ML pipeline with CLI, Docker, CI/CD, config-driven

**Best Model:** `runs/detect/train2/weights/best.pt` → **mAP@0.5 = 0.72**, **NCNN CPU: ~12 ms**

---

### Key Results

| Metric                  | Value              | Notes                                      |
|-------------------------|--------------------|--------------------------------------------|
| Model                   | YOLOv11n           | 2.6M params, 5.5 MB                        |
| Input Size              | 320×320            | Optimal speed/accuracy trade-off           |
| mAP@0.5 (test set)      | **0.72**           | Strong on "others", good on blues          |
| mAP@0.5:0.95            | 0.48               | Tight bounding boxes                       |
| Inference (NCNN CPU)    | **12 ms** (PC)    | Tested on personal Laptop    |
| Model Size (NCNN)       | ~4.8 MB            | Fits easily on embedded devices            |

---

### Installation

#### Option 1: Local (Recommended)
```bash
git clone https://github.com/SpidieBot/bottlecap-color-detection.git
cd bottlecap-color-detection
pip install poetry
poetry install
pip install -e .
```

#### Option 2: Docker
```bash
docker build -t bsort .
docker run -it --rm -v $(pwd)/dataset:/app/dataset bsort
```

---

### CLI Usage (`bsort`)

```bash
# Train model
bsort train --config configs/settings.yaml

# Inference on single image
bsort infer --config configs/settings.yaml \
  --image dataset/raw/test/images/raw-250110_dc_s001_b5_2.jpg

# Inference on directory + save JSON
bsort infer --config configs/settings.yaml \
  --dir dataset/raw/test/images --json

# Export best model to NCNN (for Raspberry Pi)
bsort export --weights runs/detect/train2/weights/best.pt --format ncnn
```

---

### Model Training & Tracking

All experiments are tracked publicly on **Weights & Biases**:

**Public Dashboard:** https://wandb.ai/spidiebot-personal/bottle_caps  
**Best Run:** `train2_final`  
**Key Observations:**
- Heavy mosaic + HSV augmentation critical for small dataset
- `imgsz=320` gives best speed/accuracy balance
- `lr0=0.001`

---

### Edge Deployment (Raspberry Pi 5)

```bash
# On Raspberry Pi 5 (aarch64)
pip install ncnn
python tools/test_cpu_inference_time.py \
  --model-dir runs/detect/train2/weights/best_ncnn_model \
  --image sample.jpg --runs 200
```

**Expected Output:**
```
[NCNN] Average inference: 12ms on AMD Ryzen 5 4600H with Radeon Graphics with background using the CPU for others
```

---

### CI/CD Pipeline (GitHub Actions)

Every push/PR runs:
- `black` + `isort` formatting check
- `pylint` code quality
- `pytest` unit tests
- Docker image build

See: [.github/workflows/ci.yml](.github/workflows/ci.yml)

---

### Future Improvements

| Idea                     | Benefit                            |
|--------------------------|------------------------------------|
| Add synthetic data       | Boost blue cap representation      |
| Use Repvit             | On Iphone 10 it is able to run 10ms
| Add confidence filtering | Reduce false positives in production |
| ONNX + TensorRT (Jetson) | For NVIDIA edge devices            |

---

### Acknowledgments

- Ultralytics YOLOv11 – https://github.com/ultralytics/ultralytics
- NCNN – https://github.com/Tencent/ncnn
- Weights & Biases – Model tracking

---