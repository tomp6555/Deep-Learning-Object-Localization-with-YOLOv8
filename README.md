# Deep-Learning-Object-Localization-with-YOLOv8

## 📌 Overview
This repository provides a complete pipeline for **deep learning-based object detection (Bounding Box Based)**  using **YOLOv8**.

The project focuses on **bounding box detection** because:
- ✅ Faster training & inference compared to segmentation.  
- ✅ Sufficient for tasks that only require locating beads.  
- ⚠️ If bead **width, continuity, or coverage** must be measured, consider pixel-level segmentation.  

I use **Ultralytics YOLOv8**, known for **real-time performance** and **accuracy**, to train, evaluate, and deploy a custom model for bead detection.

---

## 🚀 Workflow
1. Analyze requirements for bead detection.  
2. Data preparation: Convert annotations into YOLO format.  
3. Model selection: YOLOv8 chosen for speed and accuracy.  
4. Training & inference with YOLOv8.  
5. Integration into `bead_analysis.py` pipeline.  
6. Documentation & examples included.  

---

## 🛠 Annotation Methods
Several tools can be used for dataset annotation:
- **LabelImg**: Simple GUI tool for drawing bounding boxes.  
- **CVAT (Computer Vision Annotation Tool)**: Web-based, supports bounding boxes, segmentation, and interpolation for videos.  
- **Roboflow**: Cloud-based annotation, preprocessing, and dataset management.  

---

## 📂 Repository Structure
```
.
├── train_yolov8.py        # Script for training YOLOv8 on bead dataset
├── analysis.py       # Example analysis pipeline with detection integration
├── dataset.yaml           # Dataset configuration file for YOLO
├── yolo_dataset/          # Prepared dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── README.md              # Project documentation
```

---

## ⚙️ Prerequisites
- Python **3.8+**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
  ```bash
  pip install ultralytics
  ```

---

## 🗂 Dataset Preparation
1. Annotate images with bounding boxes (LabelImg, CVAT, or Roboflow).  
2. Convert annotations to **YOLO format** using `prepare_yolo_dataset.py`.  
   - Format: `class_id x_center y_center width height`  
   - Coordinates are normalized (0–1).  

### Example dataset structure
```
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### `dataset.yaml`
```yaml
# dataset.yaml
path: ../yolo_dataset  # Adjust path as needed
train: images/train
val: images/val
nc: 1
names:
  0: bead
```

---

## 🏋️ Training
Run training with:
```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

- `yolov8n.pt` → Nano model (fastest, lightweight)  
- Replace with `yolov8s.pt`, `yolov8m.pt`, etc., for larger models  

Training outputs (including `best.pt`) are saved in:
```
runs/detect/train/
```

---

## 🔎 Inference
Once trained, run detection:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images
```

- Supports **images**, **videos**, or **webcam**.  
- Detected beads are saved with bounding boxes.  

---

## 🔗 Integration with Analysis Pipeline
`analysis.py` integrates YOLO detection:
- Loads trained YOLOv8 model  
- Runs bead detection on input images  
- Exports bounding box coordinates for further analysis  
  - Bead count  
  - Size distribution  
  - Gap/break detection  

---

## 🛠 Model Choices
For deep learning-based bounding box detection, three popular models are commonly used:

1. **YOLO (You Only Look Once)**  
   - Known for its **speed** and **good accuracy**  
   - Suitable for **real-time applications** such as bead detection  
   - We use **YOLOv8** in this project  

2. **SSD (Single Shot MultiBox Detector)**  
   - Also fast and accurate  
   - Provides a **good balance between speed and performance**  
   - Often used in embedded or mobile applications  

3. **Faster R-CNN**  
   - Generally more **accurate but slower**  
   - Often used for **high-precision tasks** where detection speed is less critical  

In this project, we focus on **YOLOv8** for its strong balance between performance and efficiency.

---

## 📊 Example Results
- **Input**: Raw bead image  
- **Output**: Bounding boxes around beads + analysis results  

(You can add example images here after training.)

---

## ✅ Next Steps
- Add training logs & sample detections  
- Extend `bead_analysis.py` with quantitative bead metrics  
- (Optional) Implement segmentation for bead continuity analysis  

---

