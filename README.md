
# 🤖 Computer Vision Project

This repository contains Python scripts and assets for various computer vision tasks, including **real-time face detection**, **facial filtering/overlay**, and **object detection**.

## 📁 Project Structure Overview

The key files and directories in this project are:

| File/Directory | Description |
| :--- | :--- |
| `requirements.txt` | Lists all necessary Python dependencies (libraries). |
| `camera.py` | base script for accessing the webcam feed. |
| `filterface.py` | Core script for applying filters or overlays to detected faces. |
| `webcam_face.py` | A utility script for basic face detection using the webcam. |
| `multifunctionalhand.py` | a module for hand or gesture detection/tracking. |
| `haarcascade_frontalface_default.xml` | OpenCV's pre-trained classifier for frontal face detection. |
| `opencv_face_detector_uint8.pb` / `.pbtxt` | Files for a Deep Learning-based OpenCV Face Detector model. |
| `yolov8l.pt` / `yolov8n.pt` / `yolov8x.pt` | Pre-trained models (large, nano, extra-large) for YOLOv8 (You Only Look Once) object detection. |

-----

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1\. Prerequisites

  * Python (3.x recommended)
  * A functional webcam

### 2\. Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [your-project-folder]
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3\. Running the Scripts

Activate the virtual environment first (if not already done).

  * **Run the basic webcam face detector:**
    ```bash
    python webcam_face.py
    ```
  * **Run the face filtering application:**
    ```bash
    python filterface.py
    ```
  * **Run the hand/gesture detection script:**
    ```bash
    python multifunctionalhand.py
    ```
    *(Note: Functionality may vary based on the specific code inside each file.)*

-----

## 🛠️ Key Technologies & Models

This project utilizes several key technologies:

  * **OpenCV (`cv2`):** The primary library for computer vision tasks, handling video streams, image processing, and loading models.
  * **Haar Cascades:** Used for rapid, classic frontal face detection.
  * **Deep Learning Face Detection:** Uses the `opencv_face_detector` files for a more robust face detection method.
  * **YOLOv8:** State-of-the-art models for real-time object detection, enabling the recognition of various objects beyond faces.

-----
This `README.md` focuses on the **models** present in your file structure, what their **purpose** is, and the general **requirements** needed to run a project using them.

---

# ⚙️ Model and Technology Guide

This project utilizes three main types of Computer Vision models for different detection tasks: **YOLOv8**, an **OpenCV Deep Learning Face Detector**, and **Haar Cascades**.

## 1. YOLOv8 Models (`yolov8n.pt`, `yolov8l.pt`, `yolov8x.pt`)

YOLO (You Only Look Once) models are state-of-the-art for **real-time object detection** and segmentation. They are designed to be fast and accurate.

| Model File | Purpose/Size | Inference Speed & Accuracy |
| :--- | :--- | :--- |
| **`yolov8n.pt`** (Nano) | Smallest, fastest, but least accurate. Ideal for **edge devices** or **low-power CPUs**. | Highest speed, lowest accuracy. |
| **`yolov8l.pt`** (Large) | A robust, balanced model. Suitable for general-purpose applications. | Good balance of speed and accuracy. |
| **`yolov8x.pt`** (Extra-Large) | Largest, most accurate, but slowest. Used for high-precision tasks. | Highest accuracy, lowest speed. |

### System Requirements for Training/Inference (General)
These models are demanding for **training**, but can run **inference** (detection) on consumer hardware.

* **Minimum for Inference:** Standard modern **CPU** (multi-core), **4GB+ RAM**.
* **Recommended for Fast Inference/Training:** **GPU** (NVIDIA with CUDA support) and the appropriate driver/library installation (`torch`, `ultralytics`, `opencv-python`).

---

## 2. OpenCV Deep Learning Face Detector

Files: `opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt`

This pair of files represents a **pre-trained Deep Neural Network (DNN)** model for face detection. It is often more robust and accurate than the classic Haar Cascade method, especially under varying lighting and pose conditions.

* **`.pb` (Protocol Buffer):** The actual model weights (the brain of the detector).
* **`.pbtxt` (Text File):** The model configuration/graph definition.

### Requirements

* **OpenCV (`cv2`):** The primary Python library is required to load and run this DNN module.
* **Minimal Hardware:** Runs reasonably well on most modern **CPUs** for real-time webcam applications, as it is a relatively lightweight DNN.

---

## 3. Haar Cascade Classifier

File: `haarcascade_frontalface_default.xml`

This is a classic, machine learning-based object detection algorithm (specifically for **frontal face detection**) that is part of the OpenCV library. It is extremely **fast** and **low-computation**, making it useful for very resource-constrained devices or as a quick initial detection step.

### Requirements

* **OpenCV (`cv2`):** Requires the core Python library.
* **Minimal Hardware:** Can run on very basic **CPUs** or older hardware since it doesn't use modern deep learning techniques. It relies on simple feature calculations for speed. 


## 🤝 Contributing

Contributions are welcome\! Please feel free to open issues or submit pull requests.

-----
