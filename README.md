# 🧠 Anomaly Detection on MVTec Bottle Dataset

This project implements an anomaly detection system using a custom autoencoder architecture with PyTorch Lightning, applied to the **MVTec AD bottle dataset**. The model is trained to detect visual defects by learning to reconstruct normal images and comparing them to test samples.

## 🚀 Features

- ⚡ Custom Autoencoder (CNN-based encoder-decoder architecture)
- 🎯 Loss function: Combination of SSIM, MSE, and L1 Loss
- 🧪 Anomaly scoring using pixel-wise reconstruction errors
- 📊 Integrated logging and visualization with Weights & Biases (W&B)
- 🧠 Image data augmentation to enhance model generalization
- ✅ Achieves ~88% validation accuracy
- 📦 Organized with PyTorch Lightning for scalable training and validation

## 🛠 Tech Stack

- **Framework:** PyTorch Lightning
- **DL Backend:** PyTorch
- **Logging:** Weights & Biases (wandb)
- **Data:** MVTec Anomaly Detection dataset (bottle category)
- **Visualization:** Matplotlib, PIL
- **Environment:** Kaggle Kernel with Python 3.11

## 🗂 Project Structure

.
├── data/ # MVTec dataset (input)
├── working/ # Outputs: Augmented images, checkpoints
├── model.py # LightningModule implementation
├── train.py # Trainer and wandb integration
├── config.py # Config dataclass
├── wandb/ # W&B run logs
└── README.md


## ⚙️ Configuration

Defined via a `Configure` dataclass:
- `data_dir`: Path to training images
- `save_dir`: Where augmented images will be stored
- `test_dir`: Test images for evaluation
- `batch_size`, `max_epochs`, `learning_rate`, etc.

## 📦 Installation & Setup

Install requirements (in Kaggle, most packages are preinstalled):

```bash
pip install torch torchvision pytorch-lightning torchmetrics wandb
