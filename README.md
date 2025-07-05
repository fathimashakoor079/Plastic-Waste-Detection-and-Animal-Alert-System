# Plastic Waste Detection and Animal Alert System

This project leverages deep learning and computer vision to detect plastic waste in real-time and provide alerts, potentially to protect animals from ingesting or coming into contact with harmful plastic materials. The system uses a pre-trained MobileNetSSD model for object detection and can be run on a standard webcam setup.

## Features

- Real-time plastic waste detection using a webcam.
- Customizable set of detectable objects including various plastic items.
- Audible alert (beep) when plastic waste is detected in the camera feed.
- Easily extendable for other applications such as animal presence alerts.

## Files in the Repository

- `real_time_plastic_detection.py`: Main script that runs the detection loop, processes video frames, detects plastic waste, and triggers an audible alert.
- `MobileNetSSD_deploy.caffemodel`: Pre-trained weights for the MobileNetSSD object detection model.
- `MobileNetSSD_deploy.prototxt.txt`: Model architecture definition for the Caffe framework.
- `beep.mpeg`: Audio file played as an alert when plastic waste is detected.

## How It Works

1. The script initializes the video stream from a webcam.
2. Each frame is processed through the MobileNetSSD model to detect various objects, with an emphasis on plastic-related classes.
3. If a plastic item is detected with confidence > 0.5, a beep sound is played to alert the user.
4. The detection results are displayed in a window with bounding boxes and labels for detected items.

## Setup and Usage

### Prerequisites

- Python 3.x
- Required Python packages:
  - opencv-python
  - imutils
  - numpy
  - playsound

Install dependencies using pip:
```bash
pip install opencv-python imutils numpy playsound
```

### Running the Project

1. Make sure all the model files (`MobileNetSSD_deploy.caffemodel` and `MobileNetSSD_deploy.prototxt.txt`) and the beep audio (`beep.mpeg`) are in the same directory as the Python script.
2. Run the detection script:
```bash
python real_time_plastic_detection.py
```
3. A window will open showing the webcam feed. Plastic items will be highlighted, and an alert will sound when detected.
4. Press `q` to quit the application.

## Customization

- To add or remove detectable classes, modify the `CLASSES` list in `real_time_plastic_detection.py`.
- You can replace `beep.mpeg` with any other audio file as needed for alerts.

## Model Details

- The detection model is based on MobileNetSSD, a lightweight and efficient deep neural network for object detection.
- Model architecture is defined in `MobileNetSSD_deploy.prototxt.txt`.
- Pre-trained weights are stored in `MobileNetSSD_deploy.caffemodel`.

## Applications

- Wildlife protection: Alerting when animals are near plastic waste.
- Environmental monitoring: Automated detection of plastic waste in various environments.
- Educational demonstrations on the impact of plastic waste.

## License

This project is for educational and research purposes. Please ensure you comply with the licenses of the model and any datasets you use.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MobileNetSSD](https://github.com/chuanqi305/MobileNet-SSD)
- Sound effect from [SoundJay.com](https://www.soundjay.com/)
