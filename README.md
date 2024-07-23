# About 
This repository details the implementation of the BrainEye Take-Home Assignment. It involves developing a deep learning model to predict age from facial images. The goal is to accurately estimate the age of an individual based on their facial appearance and to outline the thought process behind the computer vision pipeline.

# Step 1: Problem definition 
The problem asks for the implementation of a deep learning model that can estimate a person's age based on various observable features from facial images. Age estimation is often performed using computer vision techniques to analyze facial characteristics such as wrinkles, skin texture, and hair color, among others. Computer vision algorithms can estimate an individual's age by comparing these facial attributes with a dataset of known age examples. However, the model's performance (accuracy) can be affected by factors such as image lighting, facial expressions, race, and makeup.

# Step 2: Methodology 
ResNet and regression are combined to create a powerful model to solve age estimation from facial images. [Residual Network(ResNet)](https://huggingface.co/docs/transformers/en/model_doc/resnet) is a deep learning model designed to help deep neural networks learn effectively and improve accuracy across various computer vision tasks. By utilizing ResNet, we can effectively capture intricate patterns and features in images, which is crucial for precise age estimation. The regression part of the model predicts the numerical age based on the features extracted by ResNet. This approach leverages the strengths of ResNet with regression to create a robust age estimation model capable of handling complex variations.

## Diagram of the method chosen 
![alt-text](https://github.com/KhaingNaing/Age_estimation_CV/raw/main/pics/ResNet.png)

# Step 3: Implementation

## Dataset 
The dataset can be found [here](https://drive.google.com/file/d/1uNA2JzKTtTaGIWtrHsrBUAg2k3eoDZHA/view?usp=drive_link). It contains facial images of people aged between 20 and 50 years. Each folder is named according to the age group it represents.

### 1. Exploratory Data Analysis (EDA)


