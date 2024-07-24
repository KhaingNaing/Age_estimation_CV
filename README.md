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

## 1. Exploratory Data Analysis (EDA)
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

## 2. Dataset Splitting 
In the age distribution analysis, the dataset is imbalanced. To address this, it is better to use stratified sampling to account for the imbalance in the age feature.

<details>
  <summary><b>Stratified Sampling</b></summary><br/>

Stratified sampling works by dividing the dataset into groups based on the stratification feature (in our case, age). This approach ensures that each group is represented proportionally, which helps address imbalances in the dataset and improves the quality of the analysis. I use $80:10:10$ split for train vs valid vs test sets. The data summary is as below: \

Train set size: $23976$ \
Validation set size: $2997$ \
Test set size: $2997$
</details>

<details>
  <summary><b>Distribution Plots for Train, Val and Test sets</b></summary><br/>

![alt-text](figs/train_test_valid_age_distri.png)

We can see that train, test and valid sets have similar age distribution, which indicates a balanced and representative dataset split.
</details>

## 3. Preprocessing and Transformation 
I have defined data preprocessing and transformations in `custom_dataset.py`, which include denoising, deblurring, image resizing, converting images to tensors, and normalizing pixel values.

## 4. Custom Dataset and DataLoader
I use `CustomDataset` to load and preprocess our data, and `DataLoader` to efficiently load, batch, and iterate over the dataset during model training and evaluation. 

## 5. Model 
I implemented a simple CNN with regression component. 

<details>
  <summary><b>SimpleCNN Model Design</b></summary><br/>

![alt-text](pics/CNN.png)

<details>
  <summary><b>Summary of Dimensions</b></summary><br/>

  1. Input (RGB image): $128 \times 128 \times 3$
  2. After 1st Conv + Pool: $64 \times 64 \times 16$
  3. After 2nd Conv + Pool: $32 \times 32 \times 32$
  4. After 3rd Conv + Pool: $16 \times 16 \times 64$
  5. After 4th Conv + Pool: $8 \times 8 \times 128$
  6. Flattened Size: $8 * 8 * 128$
  7. After 1st Fully Connected: $64$
  8. After 2nd Fully Connected: $1$ (we are predicting one numerical value)

The number of filters in the convolutional layers and the number of hidden nodes in the fully connected layers have been selected to keep the total number of parameters under 1 million. ReLU activation functions are used to introduce non-linearity into the model, which is crucial for enabling the network to learn complex patterns and features.

Define SimpleCNN:
```python
model = SimpleCNN(input_dim=3, output_nodes=1, model_name="simple_cnn")
```
</details>

</details>


## 6. Training Process

This repository contains code for the training process of the simple CNN model. 

### Contents
1. [Finding Hyperparameters](#finding-hyperparameters)
2. [Training and Evaluation Loop](#training-and-evaluation-loop)
3. [Plotting Learning Curves with Matplotlib and TensorBoard](#plot-learning-curves)
4. [Save the best model checkpoint as .pth file](#save-the-best-model)

#### Finding Hyperparameters

The code is located in `hyperparameter_tunning.py`.
<details>
  <summary><b>This process involves several steps.</b>
</summary><br/>

1. **Train and Overfit the Model on a Small Subset**:
   - Train the model on a small subset of the dataset to assess its ability to learn and identify potential issues with overfitting.

2. **Experiment with Learning Rates:**
   - Train the model for a few epochs using various learning rates to identify the optimal learning rate for effective training.

3. **Optimize Hyperparameters with Weight Decay:**
   - Create a small grid search using different values of weight decay and the best learning rates identified in Step 2. Save the results to a CSV file for further analysis. This allows us to examine how weight decay affects the model’s performance.

4. **Train the Model with Optimal Hyperparameters:**
   - Train the model for a longer duration using the best hyperparameters obtained from Step 3 to achieve the best possible performance.

</details>



#### Training and Evaluation Loop

Training code is located in `train.py` and `functions.py`.

- **Training Loop**: Handles the training process, including forward and backward pass, updating model parameters, and monitoring training metrics.
- **Evaluation Loop**: Evaluates the model on a separate validation set and computes relevant evaluation metrics.

<details>
  <summary><b>Plotting Learning Curves with Matplotlib and TensorBoard</b>
</summary><br/>

Large CNN (lastest checkpoint): $parameters = 8778049, epoch = 19, loss = 6.06$

![alt-text](figs/large_cnn_loss.png)

</details>

#### Inference
The inference function is defined in `inference.py`. This function takes a pre-trained Age Estimation model, an input image and an output image path. It loads the model checkpoint, performs inference on the input image and saves the output image with the estimated age written on it.

<details>
  <summary><b>Inference Pipeline</b>
</summary><br/>

![alt-text](pics/Inference-pipeline.png)

</details>


### Future Improvement 
<details>
  <summary><b>Model</b></summary><br />

To enhance model's capability in capturing complex facial features we can implement very deep architectures, such as [Residual Network(ResNet)](https://huggingface.co/docs/transformers/en/model_doc/resnet). By utilizing deeper models, we can enhance the model's ability in capturing intricate patterns in facial images, leading to more accurate predictions.

</details>

<details>
  <summary><b>Data Transformation</b></summary><br />

In the current data preprocessing solution, in addition to standard data transformations such as resizing and normalization, denoising or smoothing techniques are applied. These techniques help the model to generalize better by focusing on larger patterns rather than fine details, which might be noise.

One potential improvement in data cleaning is to experiment with image sharpening techniques. Sharpening can enhance edges and details, potentially providing the model with more informative features to learn from. 

</details>

<details>
  <summary><b>Data Augmentation</b></summary><br />

Data augmentation techniques, such as random cropping, rotation, scaling, flipping, and color jittering, can help create a more diverse training set. These techniques improve the model's ability to generalize to new, unseen data.

</details>

<details>
  <summary><b>Advanced Loss Function</b></summary><br />

Experiment with advanced loss functions tailored to regression tasks, which might provide better performance compared to the L1 Loss (Mean Absolute Error) currently used in training.

</details>

# Steps to Run the Code:
  1. Create a conda env with python 3.8 via:\
    - ```conda create --name age_estimation python=3.8```
  2. Activate conda env:\
    - ```conda activate age_estimation```
  3. Make sure to install all the requirements: \
    - ```conda install --file requirements.txt```
  4. Prepare the dataset: \
    - Download dataset [here](https://drive.google.com/file/d/1uNA2JzKTtTaGIWtrHsrBUAg2k3eoDZHA/view?usp=drive_link)
  5. Adjust the configuration: \
    - Edit ```config.py``` to set the dataset path and configure other features as needed. 
  6. Train the simple_cnn model: \
    - ```python3 train.py```
  7. Perform inference with the best model checkpoint in ```checkpoints/``` directory: \
    - ```python3 inference.py```




