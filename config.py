import torch.cuda

"""
This file contains the configuration for the model, training, and dataset.
    1. Image Dimensions (img_width, img_height, img_size): 128 
    2. Mean and standard deviation of [0.5, 0.5, 0.5] is standard for normalizing images with pixel values in the range [0,1]
    3. epochs: 20
    4. batch_size: 64
    5. eval_batch_size: 128
"""

config = {
    "img_width": 128,
    "img_height": 128,
    "img_size": 128,
    "mean": [0.5, 0.5, 0.5], 
    "std": [0.5, 0.5, 0.5],
    "model_name": "simple_cnn",
    "root_dir": "/home/norakami/age-prediction/dataset",
    "csv_path": "/home/norakami/age-prediction/csv_dataset",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_path": "/home/norakami/age-prediction/img_test/img-35.jpg",
    "output_path": "/home/norakami/age-prediction/img_test/output.png",
    "leaky_relu": False,
    "epochs": 30,
    "batch_size": 64,
    "eval_batch_size": 128,
    "seed": 42,
    "lr": 0.0001, # Learning rate
    "wd": 0.001,  # Weight decay
    "save_interval": 1,
    "reload_checkpoint": None,
    "log_dir": "logs",
}
