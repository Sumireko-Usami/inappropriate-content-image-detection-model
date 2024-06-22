from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule


class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val"]
        data_root = "data/"
        self.data = [os.path.join(data_root, split, i) for i in os.listdir(data_root + split)]
        if split == "train":
            # 定义训练集的数据预处理步骤
            self.transforms = transforms.Compose([
                transforms.Resize([224, 224]),  # 调整图像大小为224x224
                transforms.RandomRotation(45),  # 随机旋转图像，旋转角度在-45到45度之间
                transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
                transforms.ColorJitter(),  # 随机更改图像的亮度、对比度、饱和度和色调
                transforms.ToTensor(),  # 将图像转换为Tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化图像，使用ImageNet的均值和标准差
            ])
        else:
            # 定义验证集或测试集的数据预处理步骤
            self.transforms = transforms.Compose([
                transforms.Resize([224, 224]),  # 调整图像大小为224x224
                transforms.ToTensor(),  # 将图像转换为Tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化图像，使用ImageNet的均值和标准差
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path)
        y = int(img_path.split("/")[-1][0])  # 获取标签值，0代表非暴力，1代表暴力
        x = self.transforms(x)
        return x, y


class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 分割数据集、应用变换等
        # 创建 training, validation数据集
        self.train_dataset = CustomDataset("train")
        self.val_dataset = CustomDataset("val")
        #self.test_dataset = CustomDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
