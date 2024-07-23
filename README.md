# About 
This repository details the implementation of the BrainEye Take-Home Assignment. It involves developing a deep learning model to predict age from facial images. The goal is to accurately estimate the age of an individual based on their facial appearance and to outline the thought process behind the computer vision pipeline.

# Step 1: Problem definition 
The problem asks for the implementation of a deep learning model that can estimate a person's age based on various observable features from facial images. Age estimation is often performed using computer vision techniques to analyze facial characteristics such as wrinkles, skin texture, and hair color, among others. Computer vision algorithms can estimate an individual's age by comparing these facial attributes with a dataset of known age examples. However, the model's performance (accuracy) can be affected by factors such as image lighting, facial expressions, race, and makeup.

# Step 2: Methodology 
A combination of a Convolutional Neural Network (CNN) and regression is used to create an age estimation model. The simplified CNN architecture is designed for efficient deployment on mobile devices. By utilizing this compact CNN model, we ensure that the model remains computationally feasible while still capturing essential features from the images. The regression component then predicts the numerical age based on the features extracted by the CNN. This approach leverages the strengths of both a compact CNN and regression to develop a straightforward yet effective age estimation model.

## Diagram of the method chosen 
![alt-text](pics/Model_diagram.png)

# Step 3: Implementation

## Dataset 
The dataset can be found [here](https://drive.google.com/file/d/1uNA2JzKTtTaGIWtrHsrBUAg2k3eoDZHA/view?usp=drive_link). It contains facial images of people aged between 20 and 50 years. Each folder is named according to the age group it represents.

### 1. Exploratory Data Analysis (EDA)
This repository contains code for exploratory data analysis on a dataset of facial images categorized by age.

<details>
  <summary><b>1. Display sample images</b></summary><br/>

![alt-text](pics/sample_images.png)

</details>

<details>
  <summary><b>2. Create a CSV File with Labels</b></summary><br/>

![alt-text](pics/df_head.png)
</details>
<details>
  <summary><b>3. Age Distribution Analysis</b></summary><br/>

![alt-text](figs/age_distribution.png)
</details>

### 2. Dataset Splitting 
In the age distribution analysis, the dataset is imbalanced. To address this, it is better to use stratified sampling to account for the imbalance in the age feature.

<details>
  <summary><b>Stratified Sampling</b></summary><br/>

Stratified sampling works by dividing the dataset into groups based on the stratification feature (in our case, age). This approach ensures that each group is represented proportionally, which helps address imbalances in the dataset and improves the quality of the analysis.
</details>

<details>
  <summary><b>Distribution Plots for Train, Val and Test sets</b></summary><br/>

![alt-text](figs/train_test_valid_age_distri.png)

We can see that train, test and valid sets have similar age distribution, which indicates a balanced and representative dataset split.
</details>

### 3. Preprocessing and Transformation 
I have defined data preprocessing and transformations in `custom_dataset.py`, which include denoising, deblurring, image resizing, converting images to tensors, and normalizing pixel values.

### 4. Custom Dataset and DataLoader
I use `CustomDataset` to load and preprocess our data, and `DataLoader` to efficiently load, batch, and iterate over the dataset during model training and evaluation. 

### 5. Model 
I implemented a simple CNN with regression component. 

<details>
  <summary><b>SimpleCNN Model Design</b></summary><br/>

![alt-text](pics/CNN.png)

<details>
  <summary><b>Summary of Dimensions</b></summary><br/>

  1. Input: $128 \times 128 \times 3$
  2. After 1st Conv + Pool: $64 \times 64 \times 32$
  3. After 2nd Conv + Pool: $32 \times 32 \times 64$
  4. After 3rd Conv + Pool: $16 \times 16 \times 128$
  5. After 4th Conv + Pool: $8 \times 8 \times 256$
  6. Flattened Size: $8 * 8 * 256$
  7. After 1st Fully Connected: 512
  8. After 2nd Fully Connected: 1 (we are predicting one numerical value)
   
Additionally ReLU Activation is used to introduce non-linearity into the model. This is essential for enabling the model to learn complex patterns and features.

Define SimpleCNN:
```python
model = SimpleCNN(input_dim=3, output_nodes=1)
```
</details>

</details>


### 5. Training



### Improvement 
ResNet and regression are combined to create a powerful model to solve age estimation from facial images. [Residual Network(ResNet)](https://huggingface.co/docs/transformers/en/model_doc/resnet) is a deep learning model designed to help deep neural networks learn effectively and improve accuracy across various computer vision tasks. By utilizing ResNet, we can effectively capture intricate patterns and features in images, which is crucial for precise age estimation. The regression part of the model predicts the numerical age based on the features extracted by ResNet. This approach leverages the strengths of ResNet with regression to create a robust age estimation model capable of handling complex variations.


