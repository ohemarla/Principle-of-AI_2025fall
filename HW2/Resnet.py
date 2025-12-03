# Resnet.py
# encoding: utf-8
# 为了进行图象分类，这里基于ResNet架构实现了一个深度卷积神经网络模型。
# 模型共包括多个残差块，每个残差块包含两个卷积层以及一个残差连接。
# 这里设置了4层layer，每层包含3个残差块，整体结构类似于ResNet50。
# 输出类别数为5，分别为daisy, dandelion, rose, sunflower, tulip。

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

class ResidualBlock(nn.Module):
    """
    残差块类，包含两个卷积层和一个残差连接
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False) # # 3x3卷积层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False) # # 3x3卷积层
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x) # F(x) + x 残差连接
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    """
    ResNet模型类，包含多个残差块和全连接层
    """
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # 7x7的初始卷积层，224 -> 112
        self.bn1 = nn.BatchNorm2d(64) # 批量归一化
        self.relu = nn.ReLU(inplace=True) # relu激活函数
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) # Max池化层，112 -> 56
        self.Layer1 = self._make_layer(block, 64, layers[0], stride = 1) # 第一层，两个残差块，56 -> 56
        self.Layer2 = self._make_layer(block, 128, layers[1], stride = 2) # 第二层，两个残差块，56 -> 28
        self.Layer3 = self._make_layer(block, 256, layers[2], stride = 2) # 第三层，两个残差块，28 -> 14
        self.Layer4 = self._make_layer(block, 512, layers[3], stride = 2) # 第四层，两个残差块，14 -> 7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 平均池化层，7 -> 1
        self.fc = nn.Linear(512 * block.expansion, num_classes) # 全连接层
        self._initialize_weights()  # 初始化权重
    
    def _make_layer(self, block, out_channels, blocks, stride = 1):
        """
        创建残差块层
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion   # 更新输入通道数
        for _ in range(1, blocks):  # 添加剩余的残差块
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        初始化权重函数
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # 卷积层初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming初始化
            elif isinstance(m, nn.BatchNorm2d): # 批量归一化层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 构造样本的函数
def Sample(data_path, reinforce = False, batch_size = 32):
    """
    读取对应数据，构造好样本，把子文件夹名当作图片的标签
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)   # ImageNet数据集的均值
    IMAGENET_STD = (0.229, 0.224, 0.225)    # ImageNet数据集的标准差
    if reinforce:   # 如果reinforce为True，则进行数据增强
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224), # 随机裁剪224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), # 归一化
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33)) # 随机擦除
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    dataset = datasets.ImageFolder(root = data_path, transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataloader

def Evaluate(model, data, device):
    """
    评估函数，计算模型在数据集上的准确率和损失
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss # 返回准确率和平均损失

def Train(num_classes, batch_size, lr, epochs, patience,
          train_data_path, val_data_path, save_model_path, log_dir, device):
    """
    训练函数，训练ResNet模型
    """
    writer = SummaryWriter(log_dir)  # TensorBoard日志记录器
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).to(device) # 实现ResNet34
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)  # 采用AdamW优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)   # 学习率调度器
    train_loader = Sample(train_data_path, reinforce=True, batch_size = batch_size)
    val_loader = Sample(val_data_path, batch_size = batch_size)
    best_val_accuracy = 0
    epochs_no_improve = 0
    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0  # 记录训练集损失
        correct_train = 0  # 记录训练集正确预测数
        total_train = 0    # 记录训练集总数
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()   # 计算训练集正确预测数
            loss.backward()
            optimizer.step()
            global_step += 1
        epoch_loss = running_loss / len(train_loader.dataset)   # 计算epoch的平均损失
        train_accuracy = 100 * correct_train / total_train
        writer.add_scalar('Train/Loss_Epoch', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        val_accuracy, val_loss = Evaluate(model, val_loader, device)    # 在验证集上评估
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        print(f"Epoch {epoch+1}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")
        scheduler.step()    # 根据验证集准确率更新学习率
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_model_path)  # 保存最佳模型权重
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    writer.close()  # 关闭TensorBoard记录器
    model.load_state_dict(torch.load(save_model_path))  # 加载最佳模型权重
    return model
    
    
if __name__ == "__main__":
    train_data_path = "HW2/Data/train"
    val_data_path = "HW2/Data/val"
    test_data_path = "HW2/Data/test"
    save_model_path = "HW2/Model/resnet_model.pth"
    log_dir = "HW2/Log"
    num_classes = 5
    batch_size = 64
    lr = 0.001  # 学习率
    epochs = 100    # 最大训练轮数
    patience = 15  # 早停耐心值
    criterion = nn.CrossEntropyLoss()   # 交叉熵作为损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Train(num_classes, batch_size, lr, epochs, patience,
                  train_data_path, val_data_path, save_model_path, log_dir, device)
    test_accuracy, test_loss = Evaluate(model, Sample(test_data_path, batch_size = batch_size), device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")