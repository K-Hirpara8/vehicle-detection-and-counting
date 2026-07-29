# Real-Time Vehicle Detection and Counting Using YOLOv8

## 📌 Overview

This project detects, tracks, and counts vehicles in traffic videos using a fine-tuned **YOLOv8s** object-detection model.

The model detects vehicles in each video frame. A centroid-based tracking method assigns temporary IDs and follows their movement. Vehicles are counted when they cross predefined entry or exit lines. The processed video is displayed with bounding boxes, vehicle IDs, counts, and a **Vehicle Counting System** dashboard.

---

## ❗ Problem Statement

Manual vehicle counting from traffic videos is time-consuming and may produce errors, especially when many vehicles are moving in different directions.

The main challenge is to:

- Detect vehicles correctly in every frame
- Track the same vehicle across consecutive frames
- Count each vehicle only once
- Separate entering and exiting movement
- Save the final annotated video

---

## 🎯 Project Goal

The goal of this project is to build a complete computer-vision workflow that can:

- Fine-tune a pretrained YOLOv8s model on a custom vehicle dataset
- Detect vehicles in traffic videos
- Assign temporary IDs using centroid-distance matching
- Count vehicles crossing entry and exit lines
- Reduce duplicate counting
- Display the results on the video
- Save the processed video automatically

---

## 🗂 Dataset

The model was fine-tuned using the **Top-View Vehicle Detection Image Dataset** from Kaggle.

| Dataset Information | Value |
|---|---:|
| Training images | 536 |
| Validation images | 90 |
| Total images | 626 |
| Classes | 1 |
| Class name | `vehicle` |
| Image size | 640 × 640 |
| Annotation format | YOLO |

The dataset is not included in this repository.

The dataset location and class information are configured in `data.yaml`:

```yaml
path: archive (1)/Vehicle_Detection_Image_Dataset

train: train/images
val: valid/images

names:
  0: vehicle
```

---

## 🧠 Model Approach

The project starts with the pretrained **YOLOv8 Small** model:

```python
model = YOLO("yolov8s.pt")
```

The pretrained model is then fine-tuned on the custom vehicle dataset.

YOLOv8s was selected because it provides a practical balance between:

- Detection accuracy
- Inference speed
- Model size
- GPU-memory usage

After training, the best model is saved as `best.pt` and used for video detection and counting.

---

## ⚙️ Training Configuration

```python
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    workers=0,
    device=0,
    pretrained=True,
    project="runs",
    name="train1"
)
```

| Parameter | Value |
|---|---:|
| Model | YOLOv8s |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 8 |
| Device | NVIDIA GPU |
| Workers | 0 |
| Pretrained weights | Yes |

The trained weights are stored in:

```text
runs/detect/runs/train1/weights/
├── best.pt
└── last.pt
```

- `best.pt` contains the checkpoint with the best validation performance.
- `last.pt` contains the checkpoint from the final training epoch.

---

## 📊 Model Results

The final validation results for `best.pt` were:

| Metric | Result |
|---|---:|
| Precision | 0.915 |
| Recall | 0.944 |
| mAP@50 | 0.975 |
| mAP@50–95 | 0.730 |

The model was validated on:

- 90 validation images
- 937 vehicle instances

Ultralytics also generated training graphs, precision-recall curves, F1 curves, confusion matrices, and `results.csv`.

---

## 🚗 Detection and Counting Workflow

The counting process follows these steps:

1. Open the traffic video using OpenCV.
2. Read one video frame at a time.
3. Detect vehicles using the fine-tuned `best.pt` model.
4. Draw a bounding box around each detected vehicle.
5. Calculate the centre point of each bounding box.
6. Compare the centre with vehicle positions from the previous frame.
7. Reuse an existing ID or create a new vehicle ID.
8. Check whether the vehicle crosses the entry or exit line.
9. Store counted IDs to reduce duplicate counting.
10. Display and save the annotated frame.

The centroid is calculated using:

```python
cx = (x1 + x2) // 2
cy = (y1 + y2) // 2
```

The distance between two centroids is calculated using:

```python
distance = math.hypot(cx - px, cy - py)
```

A distance threshold of `40` pixels is used for ID matching.

---

## ↕️ Counting Logic

Two virtual horizontal lines are used:

```python
line_B = 500   # Exit line
line_A = 700   # Entry line
```

- A vehicle moving downward across the entry line increases the **Entered** count.
- A vehicle moving upward across the exit line increases the **Exited** count.

The following sets prevent the same ID from being counted repeatedly:

```python
counted_entered = set()
counted_left = set()
```

The line positions can be changed for videos with a different resolution or camera angle.

---

## 🖥 Output

The final video displays:

- Vehicle bounding boxes
- Temporary vehicle IDs
- Entry and exit lines
- Entered vehicle count
- Exited vehicle count
- Vehicle Counting System dashboard

The annotated video is saved automatically as:

```text
vehicle_counting_output.mp4
```

---

## 🛠 Technologies and Versions

### Verified development environment

| Technology | Version / Details |
|---|---|
| Python | 3.11.9 |
| Ultralytics | 8.4.108 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Model | YOLOv8s |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| GPU memory | 8 GB |
| Operating system | Windows |
| IDE | Visual Studio Code |
| Notebook | Jupyter Notebook |

### Main libraries

- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Torchvision
- ipykernel

---

## 📂 Project Structure

```text
Real-Time-Vehicle-Detection-and-Counting-YOLOv8/
│
├── VehicleDetection.ipynb
├── counting.py
├── data.yaml
├── requirements.txt
├── README.md
├── yolov8s.pt
├── vehicle_counting_output.mp4
│
├── archive (1)/
│   └── Vehicle_Detection_Image_Dataset/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── valid/
│           ├── images/
│           └── labels/
│
└── runs/
    └── detect/
        └── runs/
            └── train1/
                ├── weights/
                │   ├── best.pt
                │   └── last.pt
                └── training results
```

---

## 📥 Installation

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install CUDA-enabled PyTorch

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install the remaining libraries

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Model training and evaluation

Open:

```text
VehicleDetection.ipynb
```

Select the project virtual environment as the Jupyter kernel and run the notebook cells in order.

The notebook performs:

- Python and CUDA verification
- Dataset verification
- YOLOv8s model loading
- Model fine-tuning
- Model validation
- Best-weight verification
- Video prediction

The training cell does not need to be run again when `best.pt` already exists.

### Vehicle counting

Check the model and video paths inside `counting.py`, then run:

```powershell
python counting.py
```

Press `q` while the video window is active to stop the program.

The processed video is saved as:

```text
vehicle_counting_output.mp4
```

---

## ⚠️ Limitations

The project uses lightweight centroid-based tracking rather than an advanced multi-object tracker.

Counting accuracy may decrease when:

- Vehicles move very close to each other
- Vehicles overlap or become temporarily hidden
- The model misses a vehicle in one or more frames
- A vehicle moves more than 40 pixels between frames
- The camera position or video resolution changes
- The entry and exit lines are not positioned correctly
- Traffic becomes very dense

The tracking threshold and line coordinates may need adjustment for a different video.

---

## 🎯 What I Learned

Through this project, I gained practical experience in:

- Fine-tuning a pretrained YOLOv8 model
- Preparing and validating a YOLO-format dataset
- Training a model with GPU acceleration
- Evaluating object-detection performance
- Processing videos frame by frame with OpenCV
- Extracting bounding boxes from YOLO predictions
- Calculating centroids and matching object positions
- Assigning temporary vehicle IDs
- Implementing entry and exit counting logic
- Reducing duplicate vehicle counts
- Creating and saving an annotated output video
