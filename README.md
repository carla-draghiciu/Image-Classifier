# Happy vs Sad - Image Classifier
## Project Overview
This is binary image classification system that predicts whether the people in the image seem happy or sad.
The model was built using TensorFlow and Keras for the deep learning model and uses a simple and interactive Streamlit web interface.

This project was built as a part of my learning journey in AI and deep learning, with the goal of understanding:
* Convolutional Neural Networks
* image processing
* model training and evaluation

## Screenshots
<img width="959" height="412" alt="image" src="https://github.com/user-attachments/assets/ed0cf47a-03f3-4734-9f2a-bc8eb99709c0"/>

<img width="955" height="412" alt="image" src="https://github.com/user-attachments/assets/3e71d81c-6827-4add-b0b8-d54b17161467" />

## How it works
* user uploads an image (jpg, jpeg, png);
* image is preprocessed: resized to 256x256 pixels, converted to a NumPy array, normalized (pixel values between 0-1);
* prediction is made and displayed;
