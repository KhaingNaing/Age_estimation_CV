import torch.cuda

config = {
    'img_width': 128,
    'img_height': 128,
    'img_size': 128,
    'mean': [0.5, 0.5, 0.5],  # Default normalization values if not using pretrained model stats
    'std': [0.5, 0.5, 0.5],
    'model_name': 'simple_cnn',
    'root_dir': '/home/norakami/age-prediction/dataset',
    'csv_path': '/home/norakami/age-prediction/csv_dataset',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'image_path_test': '',
    # 'output_path_test': '',
    'leaky_relu': False,
    'epochs': 20,
    'batch_size': 64,
    'eval_batch_size': 128,
    'seed': 42,
    'lr': 0.0001, # Learning rate
    'wd': 0.001,  # Weight decay
    'save_interval': 1,
    'reload_checkpoint': None,
    'finetune': None,
    'weights_dir': 'weights',
    'log_dir': 'logs',
    'cpu_workers': 4,
}
