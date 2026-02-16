# Sign Language Detection using Computer Vision and Deep Learning

This project is a real-time Sign Language Detection system developed using Python, OpenCV, MediaPipe, and TensorFlow. It detects hand gestures from a webcam feed and classifies them into predefined sign language categories using a trained deep learning model.

The purpose of this project was to gain practical experience in computer vision, dataset creation, and deploying deep learning models for real-time applications.

---

## Key Highlights

* Developed a real-time hand gesture recognition system using MediaPipe and TensorFlow
* Built a complete pipeline including data collection, preprocessing, training, and real-time prediction
* Implemented image normalization and aspect ratio handling for improved model performance
* Achieved reliable real-time detection using a custom-trained classification model
* Designed modular and reusable code structure

---

## How It Works

1. Webcam captures live video input
2. MediaPipe detects and extracts the hand region
3. The hand image is resized and normalized
4. The processed image is passed to a trained Keras model
5. The model predicts the corresponding sign label
6. The result is displayed on the screen in real time

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* CVZone
* NumPy

---

## Project Structure

```
Sign-language-detection/

Data/
    Dataset images used for training

converted_keras/
    keras_model.h5      Trained classification model
    labels.txt          Class labels

dataCollection.py
    Script for collecting training data

test.py
    Script for real-time prediction

README.md
    Project documentation
```

---

## Installation and Setup

Clone the repository:

```
git clone https://github.com/maserblack/Sign-language-detection.git
cd Sign-language-detection
```

Create environment:

```
conda create -n signlang python=3.10
conda activate signlang
```

Install dependencies:

```
pip install opencv-python==4.7.0.72
pip install opencv-contrib-python==4.7.0.72
pip install mediapipe==0.10.5
pip install cvzone==1.5.6
pip install numpy==1.23.5
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal
```

---

## Running the Application

Run the real-time detection script:

```
python test.py
```

The webcam will open and display the detected sign.

---

## Learning Outcomes

Through this project, I gained hands-on experience with:

* Computer vision and real-time image processing
* Hand landmark detection using MediaPipe
* Deep learning model integration using TensorFlow
* Dataset collection and preprocessing
* Building complete end-to-end machine learning applications

---

## Future Improvements

* Add support for more sign classes
* Improve model accuracy with larger datasets
* Deploy as a web application
* Optimize performance for faster inference

---

## Demo Video

Link:- https://drive.google.com/file/d/1DtC8DXpVILXVgxr-KVu_2jW_dJjIeLUd/view?usp=sharing

## Author

Anshuman Singh Raghuvanshi
GitHub: https://github.com/maserblack

---

This project was developed as part of my learning in computer vision and machine learning, with a focus on building real-world applications.

