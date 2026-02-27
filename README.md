# SCAS AI

SCAS AI is a Python-based object detection application using YOLOv3. It provides a professional UI for managing camera sessions and detecting objects in real-time.

## Features

- Real-time object detection using YOLOv3.
- Professional Tkinter-based UI.
- Camera session management (Activate/Deactivate).
- Detection logs and visualization.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Tkinter (usually comes with Python)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/keerthivasanrk/SMART-Camera-Alert-System-SCAS-AI-.git
    cd SMART-Camera-Alert-System-SCAS-AI-
    ```

2.  **Install dependencies**:
    ```bash
    pip install opencv-python numpy
    ```

3.  **Download YOLOv3 Weights**:
    The weights file (`yolov3.weights`) is too large for GitHub. Please download it from the official source:
    - [YOLOv3 Weights (236MB)](https://pjreddie.com/media/files/yolov3.weights)
    
    Place the `yolov3.weights` file in the root directory of the project.

4.  **Run the application**:
    ```bash
    python scas.py
    ```
