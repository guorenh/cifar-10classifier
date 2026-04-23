import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr

# ======================
# 模型定义
# ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取层：多个卷积层 + 批归一化 + ReLU 激活
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
        # 分类器层：自适应池化 + 展平 + 全连接层
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
# 设备与模型加载
# ======================
# 选择运行设备：如果当前环境支持 CUDA，则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
# 加载已训练好的模型权重
model.load_state_dict(torch.load("cifar10_best_model.pth", map_location=device))
model.eval()  # 切换到评估模式


# ======================
# 类别标签
# ======================
classes = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ======================
# 图像预处理
# ======================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


def predict(img):
    # 将 PIL 图像转换为 RGB，并进行与训练时相同的预处理
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][pred].item()

    class_name = classes[pred.item()]
    return f"类别：{class_name}", f"置信度：{prob:.2%}"


# ======================
# Gradio 用户界面
# ======================
with gr.Blocks(title="图片分类器") as demo:
    gr.Markdown(
        "# 图片分类识别程序\n仅支持 plane, car, bird, cat, deer, dog, frog, horse, ship, truck 的识别"
    )
    image = gr.Image(type="pil", label="上传图片")
    label = gr.Textbox(label="预测结果")
    prob = gr.Textbox(label="置信度")
    btn = gr.Button("开始预测")
    btn.click(predict, inputs=image, outputs=[label, prob])


if __name__ == "__main__":
    # 本地启动 Gradio Web 界面
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False
    )