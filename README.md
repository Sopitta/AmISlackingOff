# AmISlackingOff - Computer Vision Based Slacking Detection

This project uses computer vision to detect whether a person is slacking off or working. It uses YOLO11 for real-time detection and provides visual and audio alerts when slacking behavior is detected.

## Project Workflow

### 1. Data Collection
- Use the webcam to collect images of both working and slacking behaviors
- The data collection script is located in `utils/collect_data.py`
- Run the script with:
  ```bash
  python utils/collect_data.py --behavior [working/slacking] --num-images [number] --delay [seconds]
  ```
- Images will be saved in the `data/images` directory

### 2. Data Annotation
- Install Label Studio:
  ```bash
  pip install label-studio
  ```
- Start Label Studio:
  ```bash
  label-studio
  ```
- Create a new project and set up the labeling interface
- Import your collected images
- Annotate the images with the annotations (working/slacking)
- Make sure to export in YOLO format.

### 3. Model Training
- The project uses YOLO11n for training
- Training configuration is specified in `data.yaml`
- Make sure to split the data into training and validation before.
- Run the training with (example):
  ```bash
  yolo train model=yolo11n.pt data=datasets/data.yaml epochs=100 imgsz=640
  ```
- The trained model will be saved in the `runs/detect/train/weights` directory

### 4. Testing the Model
- Use `test_trained_model.py` to test the trained model in real-time
- Run the script:
  ```bash
  python test_trained_model.py
  ```
- The script will:
  - Load the trained model
  - Access your webcam
  - Detect working/slacking behavior in real-time
  - Trigger an alarm when slacking is detected

## Requirements
- Python 3.8+
- Required packages (see requirements.txt):
  - numpy==1.26.4
  - opencv_python==4.11.0.86
  - ultralytics==8.3.98
  - label-studio

## Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
.
├── data/                  # Training data directory
├── datasets/             # Dataset storage
├── recordings/           # Recorded video data
├── runs/                 # Training results and model weights
├── utils/                # Utility scripts
│   └── collect_data.py   # Data collection script
├── test_trained_model.py # Model testing script
├── data.yaml            # Training configuration
└── requirements.txt     # Project dependencies
```
