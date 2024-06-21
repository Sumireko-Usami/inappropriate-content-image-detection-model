# Violence Image Classification Interface

This repository contains the `ViolenceClass` interface for classifying images as either containing violence or not. The model is based on ResNet-34 and uses PyTorch for inference.

由于数据集和模型权重文件过大，可通过对应文件夹中给出的下载链接进行下载。

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
    - [Interface Description](#interface-description)
    - [Example](#example)
- [Requirements](#requirements)

## Installation

1. Clone the repository:
    ```sh
    git https://github.com/Sumireko-Usami/inappropriate-content-image-detection-model.git
    cd inappropriate-content-image-detection-model
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Interface Description

The `ViolenceClass` interface provides a method to classify images. The main method is `classify`, which takes a batch of images as input and returns the predicted classes.

#### `ViolenceClass`

```python
class ViolenceClass:
    def __init__(self, model_checkpoint_path: str, device: str = 'cuda:0'):
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = Resnet34ViolenceClassifier.load_from_checkpoint(model_checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # 定义预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify(self, imgs : torch.Tensor) -> list:
        # 确保输入是一个torch.Tensor，并且已经在0-1范围内
        if not isinstance(imgs, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")
        
        # 如果输入是单张图像，扩展为批次维度
        if imgs.dim() == 3:
            imgs = torch.unsqueeze(imgs, 0)

        # 归一化处理
        imgs_normalized = self.normalize_images(imgs)

        # 将输入tensor移动到目标设备
        imgs_normalized = imgs_normalized.to(self.device)

        preds = []
        with torch.no_grad():
            output = self.model(imgs_normalized)
            predict = torch.softmax(output, dim=1)
            _, classes = torch.max(predict, dim=1)
            preds = classes.cpu().numpy().tolist()

        # 如果输入是单张图像，则返回预测类别索引，否则返回预测类别索引列表
        return preds[0] if len(preds) == 1 else preds
    
    def normalize_images(self, imgs: torch.Tensor) -> torch.Tensor:
        # 确保输入是在0-1范围内的torch.Tensor，并且是RGB格式
        assert imgs.dim() == 4  # 假设输入是(batch_size, channels, height, width)
        assert imgs.shape[1] == 3  # 假设输入是RGB图像

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_imgs = normalize(imgs)

        return normalized_imgs
```

### Example

Here is an example of how to use the ViolenceClass interface:

1.Ensure you have a trained model checkpoint file, e.g., resnet34_checkpoint.pth.

2.Create a script to use the ViolenceClass:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
from torch import nn
import numpy as np
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from classify import ViolenceClass
# 假设模型检查点文件在 'checkpoint/resnet34_pretrain_test-epoch=18-val_loss=0.41.ckpt'
model_checkpoint_path = 'checkpoint/resnet34_pretrain_test-epoch=18-val_loss=0.41.ckpt'

# 实例化分类器
classifier = ViolenceClass(model_checkpoint_path)

# 同时处理多个文件夹里的内容
imgs_root = "data"
folders = ["source","train"]

# 批量大小
batch_size = 16

# 预测和评估每个文件夹的数据
for folder in folders:
    img_folder = os.path.join(imgs_root, folder)
    assert os.path.exists(img_folder), f"folder: '{img_folder}' does not exist."

    # 读取指定文件夹下所有 jpg 和 png 图像路径
    img_path_list = [os.path.join(img_folder, i) for i in os.listdir(img_folder) if i.endswith((".jpg", ".png"))]

    for ids in range(0, len(img_path_list), batch_size):
        img_list = []
        for img_path in img_path_list[ids: ids + batch_size]:
            assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
            img = Image.open(img_path).convert('RGB')  # 确保图像为 RGB 模式
            img = classifier.transform(img)  # 预处理
            img_list.append(img)

        # 将 img_list 列表中的所有图像打包成一个 batch
        batch_img = torch.stack(img_list, dim=0).to(classifier.device)

        # 预测类别
        output = classifier.classify(batch_img)

        # 打印每张图像的预测结果
        for idx, img_path in enumerate(img_path_list[ids: ids + batch_size]):
            print(f"Image: {img_path}, Predicted class: {output[idx]}")


# 处理单张图片
example_image_path = 'data/adv_val1/0_adv_0.png'
example_image = Image.open(example_image_path).convert('RGB')
example_image = classifier.transform(example_image)  # 预处理

# 单张图像分类
prediction = classifier.classify(example_image)
print(f"Prediction: {prediction}")
```

3.Run the script
