# PyTorch U-Net Brain Tumor Segmentation

Brain MRI tumor segmentation using **U-Net**, **PyTorch**, **OpenCV**, and **Albumentations**.

This project implements a simple deep learning pipeline for detecting and segmenting brain tumor regions from MRI images. It loads image/mask pairs, trains a U-Net model, evaluates predictions with overlap-based metrics, and visualizes model results.

> This project is for educational and research purposes only. It is not intended for clinical diagnosis.

---

## Features

- U-Net segmentation model implemented in PyTorch
- Brain MRI image and mask loading with OpenCV
- Albumentations-based preprocessing and augmentation
- Train / validation / test split
- BCE + Dice loss
- Dice / IoU-style evaluation
- Prediction visualization
- CUDA support when available

---

## Sample Results

### Dataset samples

![Sample Data](./sampleData.png)

### Positive tumor samples

![Positive Samples](./samplePositive.png)

---

## Dataset

This project uses the **LGG MRI Segmentation Dataset** from Kaggle:

```text
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
```

Place the dataset inside the repository like this:

```text
Pytorch-UNet-Brain-Tumor-Detection/
└── dataSets/
    ├── patient_folder_1/
    │   ├── image.tif
    │   └── image_mask.tif
    ├── patient_folder_2/
    │   ├── image.tif
    │   └── image_mask.tif
    └── ...
```

The code expects mask files with `_mask` in their filename.

---

## Installation

```bash
git clone https://github.com/mahdizynali/Pytorch-UNet-Brain-Tumor-Detection.git
cd Pytorch-UNet-Brain-Tumor-Detection
```

```bash
pip install torch torchvision numpy pandas opencv-python matplotlib albumentations scikit-learn tqdm
```

---

## Usage

Run the main script:

```bash
python main.py
```

Then choose one of the available options:

```text
1 : train a new model
2 : use the last model
0 : exit
```

After training, the model is saved as:

```text
model.pth
```

---

## Project Structure

```text
├── brain.py              # Dataset class and U-Net model
├── config.py             # Hyperparameters and device configuration
├── main.py               # Main training/testing script
├── trainer.py            # Training, loss, and evaluation logic
├── plotter.py            # Visualization utilities
├── sampleData.png        # Dataset sample image
└── samplePositive.png    # Positive tumor sample image
```

---

## Model

The model is based on the **U-Net architecture**, a popular encoder-decoder network for biomedical image segmentation.

Input shape:

```text
3 x 128 x 128
```

Output shape:

```text
1 x 128 x 128
```

The output is a binary segmentation mask showing the predicted tumor region.

---

## Configuration

Main settings can be changed in `config.py`:

```python
num_epochs = 30
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
```

You can adjust the number of epochs, batch size, dataset path, and device.

---
