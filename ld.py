import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
# 数据预处理和加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义教师模型和学生模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


teacher_model = SimpleModel()
student_model = SimpleModel()


# 广义Softmax函数
def generalized_softmax(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)


# KL散度损失计算
def kl_divergence_loss(p, q):
    return F.kl_div(F.log_softmax(q, dim=-1), p, reduction='batchmean')


# 知识蒸馏模型
class KDModel(nn.Module):
    def __init__(self, teacher_model, student_model, temperature=1.0):
        super(KDModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature

    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)  # 教师模型预测的logits
        student_logits = self.student_model(x)  # 学生模型预测的logits

        # 对每一边界框的位置应用广义Softmax
        teacher_probs = generalized_softmax(teacher_logits, self.temperature)
        student_probs = generalized_softmax(student_logits, self.temperature)

        # 计算KL散度损失
        loss = kl_divergence_loss(teacher_probs, student_probs)

        return loss


# 初始化知识蒸馏模型和优化器
kd_model = KDModel(teacher_model, student_model, temperature=2.0)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 训练过程
num_epochs = 5
for epoch in range(num_epochs):
    kd_model.train()
    running_loss = 0.0
    for data, _ in train_loader:
        optimizer.zero_grad()
        loss = kd_model(data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print('Finished Training')