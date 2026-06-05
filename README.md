# Real-Time Hand Gesture Recognition

## Overview

This project implements a real-time hand gesture recognition system using:

- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- MediaPipe Hands
- OpenCV
- Python

The model was trained on the LeapGestRecog dataset and deployed for live webcam-based gesture recognition.

---

## Features

- Real-time hand detection using MediaPipe
- Hand landmark extraction
- Bounding box generation
- CNN-based gesture classification
- Live webcam inference
- Confidence score display

---

## Dataset

Dataset used:

LeapGestRecog Dataset (Kaggle)

Classes:

- Palm
- L
- Fist
- Fist Moved
- Thumb
- Index
- OK
- Palm Moved
- C
- Down

---

## Model Architecture

CNN Architecture:

- Conv2D (32 filters)
- MaxPooling2D
- Conv2D (64 filters)
- MaxPooling2D
- Conv2D (128 filters)
- MaxPooling2D
- Dense (256 neurons)
- Dropout (0.5)
- Softmax Output Layer

Input Shape:

64 × 64 × 3

Output Classes:

10

---

## Results

Validation Accuracy:

~99%

---

## Project Workflow

Dataset
↓
Preprocessing
↓
CNN Training
↓
Model Evaluation
↓
Model Export (.keras)
↓
MediaPipe Hand Detection
↓
Real-Time Gesture Prediction

---

## Files

- vision_self_project.ipynb → Model development and training
- gesture_model.keras → Trained model
- real_time_gesture.py → Real-time deployment
- mediapipe_test.py → Hand detection testing
- gesture_bbox.py → Bounding box generation

---

## Author

Raj Shahi

Department of Agricultural and Food Engineering

Indian Institute of Technology Kharagpur
