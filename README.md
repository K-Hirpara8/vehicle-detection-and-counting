# Real-Time Vehicle Detection and Counting using YOLOv8

## Overview
This project implements a real-time vehicle detection and counting system using a pre-trained YOLOv8 object detection model. The system detects vehicles in video streams and counts their movement across predefined virtual lines to estimate the number of vehicles entering and leaving a scene.

## Model
- Pre-trained YOLOv8 object detection model (Ultralytics)
- Model weights loaded from a trained checkpoint file
- Inference performed frame-by-frame on video input

## Data
- Video input containing road traffic scenes
- Frames extracted from the video stream for detection and analysis

## Implementation Details
- Vehicle detection using YOLOv8 and OpenCV
- Centroid-based object tracking to maintain vehicle identities across frames
- Line-crossing logic to count vehicles entering and leaving the scene
- Frame skipping applied to balance detection accuracy and runtime efficiency

## Tools & Libraries
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

## Learning Outcomes
- Practical experience with real-time object detection using YOLOv8
- Image preprocessing and video-based inference workflows
- Understanding the impact of model configuration and training settings on detection performance
- Hands-on experience with performance evaluation in image-based deep learning systems
