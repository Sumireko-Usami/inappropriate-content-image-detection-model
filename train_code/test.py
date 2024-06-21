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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

class Resnet34ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)
        return acc
    
# 示例代码
if __name__ == "__main__":
    # 假设模型检查点文件在 'checkpoint/resnet34_pretrain_test-epoch=18-val_loss=0.41.ckpt'
    model_checkpoint_path = 'checkpoint/resnet34_pretrain_test-epoch=18-val_loss=0.41.ckpt'

    # 实例化分类器
    classifier = ViolenceClass(model_checkpoint_path)

    # 同时处理多个文件夹里的内容
    imgs_root = "data"
    folders = ["source","train","val","AIGC","val_noise","source_noise/source_noise",
               "adv_val1"]

    # 批量大小
    batch_size = 16

    # 预测和评估每个文件夹的数据
    for folder in folders:
        img_folder = os.path.join(imgs_root, folder)
        assert os.path.exists(img_folder), f"folder: '{img_folder}' does not exist."
        
        y_true = []
        y_pred = []  # 重置为[]，用于存储当前文件夹所有预测结果

        # 读取指定文件夹下所有 jpg 和 png 图像路径
        img_path_list = [os.path.join(img_folder, i) for i in os.listdir(img_folder) if i.endswith((".jpg", ".png"))]

        for ids in range(0, len(img_path_list), batch_size):
            img_list = []
            tag_list = []
            for img_path in img_path_list[ids: ids + batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
                img = Image.open(img_path).convert('RGB')  # 确保图像为 RGB 模式
                img = classifier.transform(img)  # 预处理
                img_list.append(img)
                tag_list.append(int(img_path.split("/")[-1][0]))

            # 将 img_list 列表中的所有图像打包成一个 batch
            batch_img = torch.stack(img_list, dim=0).to(classifier.device)
            
            # 预测类别
            output = classifier.classify(batch_img)

            # 打印每张图像的预测结果
            #for idx, img_path in enumerate(img_path_list[ids: ids + batch_size]):
                #print(f"Image: {img_path}, Predicted class: {output[idx]}")
                
            # 将当前批次的真实类别添加到 y_true 列表中
            y_true.extend(tag_list)
            # 将当前批次的预测类别添加到 y_pred 列表中
            y_pred.extend(output)

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 计算准确率、召回率、精确率和 F1 分数
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        folder_clean = folder.replace('/', '_') 

        # 绘制混淆矩阵图表
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for Model Resnet34 on folder {folder_clean}')
        
        plt.savefig(f"pic/confusion_matrix_model_Resnet34_folder_{folder_clean}.jpg")  # 保存混淆矩阵图片
        plt.show()
        plt.clf()  # 清除当前图形内容，准备绘制下一个混淆矩阵

        print(f"\nMetrics for Model Resnet34 on folder '{folder}':")
        print("Confusion Matrix:")
        print(cm)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)