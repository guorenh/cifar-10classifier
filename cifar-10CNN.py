import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 选择训练设备：优先使用 GPU（CUDA），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# ======================
# 数据路径
# ======================
# 请将这里修改为 CIFAR-10 数据所在的目录
data_root = "你的数据路径"

# ======================
# 数据增强和预处理
# ======================
# 训练集使用随机裁剪、翻转、颜色抖动来增强样本多样性
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 测试集仅进行归一化，不进行数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# ======================
# 数据集与数据加载器
# ======================
# 训练集
train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试集
test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ======================
# 模型定义
# ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分：3 个卷积块，每个块包含两个卷积层 + BN + ReLU
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 分类器部分：自适应池化 + 平铺 + 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ======================
# 模型初始化
# ======================
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失用于多分类
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

# ======================
# 早停机制
# ======================
# 如果连续多次测试损失未下降，则停止训练，避免过拟合
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, test_loss):
        if self.best_loss is None:
            self.best_loss = test_loss
        elif test_loss > self.best_loss:
            self.counter += 1
            print(f"⚠️ 过拟合预警：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = test_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=7)

# ======================
# 训练与测试指标记录
# ======================
train_acc_list = []
test_acc_list = []
test_loss_list = []

# ======================
# 训练循环
# ======================
for epoch in range(50):
    print(f"======= 第 {epoch + 1} 轮训练 =======")

    model.train()  # 切换到训练模式
    total = 0
    correct = 0
    train_loss = 0.0

    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_acc_list.append(train_acc)
    print(f"训练准确率：{train_acc:.2f}%")

    # ======================
    # 测试阶段
    # ======================
    model.eval()  # 切换到评估模式
    total = 0
    correct = 0
    test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)
    test_loss = test_loss / len(test_loader)
    test_loss_list.append(test_loss)

    print(f'测试准确率：{test_acc:.2f}%')
    print(f'测试损失：{test_loss:.4f}')

    # ======================
    # 保存最优模型
    # ======================
    if test_acc == max(test_acc_list):
        torch.save(model.state_dict(), "cifar10_best_model.pth")
        print("保存最优模型")

    # ======================
    # 早停判断
    # ======================
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print("检测到过拟合，训练停止！")
        break

# ======================
# 结果可视化
# ======================
plt.plot(train_acc_list, label='Train Accuracy', color='blue')
plt.plot(test_acc_list, label='Test Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()

