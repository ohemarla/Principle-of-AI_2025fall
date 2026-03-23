import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

criterion = nn.CrossEntropyLoss()


class PatchEmbedding(nn.Module):
    """
    使用卷积完成图像分块和线性映射。
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Transformer 编码器中的前馈网络。
    """
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    标准 ViT 编码器块：LN -> MHA -> residual -> LN -> MLP -> residual
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + self.dropout(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    适用于 224x224 图像分类的 Vision Transformer。
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=5,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def Sample(data_path, reinforce=False, batch_size=32):
    """
    读取对应数据，构造样本并返回 DataLoader。
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    if reinforce:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=reinforce,
    )
    return dataloader


def Evaluate(model, data, device):
    """
    评估函数，计算模型在数据集上的准确率和损失。
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
    return accuracy, avg_loss


def Train(
    num_classes,
    batch_size,
    lr,
    epochs,
    patience,
    train_data_path,
    val_data_path,
    save_model_path,
    log_dir,
    device,
):
    """
    训练函数，训练 ViT 模型。
    """
    writer = SummaryWriter(log_dir)
    model = ViT(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = Sample(train_data_path, reinforce=True, batch_size=batch_size)
    val_loader = Sample(val_data_path, batch_size=batch_size)
    best_val_accuracy = -1
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        writer.add_scalar("Train/Loss_Epoch", epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
        val_accuracy, val_loss = Evaluate(model, val_loader, device)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        print(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
        scheduler.step()
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    writer.close()
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    return model


if __name__ == "__main__":
    train_data_path = "HW2/Data/train"
    val_data_path = "HW2/Data/val"
    test_data_path = "HW2/Data/test"
    save_model_path = "HW2/Model/vit_model.pth"
    log_dir = "HW2/Log/vit"
    num_classes = 5
    batch_size = 32
    lr = 0.0003
    epochs = 100
    patience = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Train(
        num_classes,
        batch_size,
        lr,
        epochs,
        patience,
        train_data_path,
        val_data_path,
        save_model_path,
        log_dir,
        device,
    )
    test_accuracy, test_loss = Evaluate(
        model,
        Sample(test_data_path, batch_size=batch_size),
        device,
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
