
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    """ CNN处理obs 
    define: state_shape[h], state_shape[w], action_dim
    input : [batch_size, 3, h, w]
    output: [batch_size, action_dim]
    """
    def __init__(self, h, w, outputs):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(32)

        # 计算卷积层输出尺寸
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size-kernel_size+2*padding) // stride + 1
        
        # 假设初始输入为(h, w)，依次计算经过每一层卷积后的输出尺寸
        convw, convh = w, h
        for _ in range(3):  # 因为有三层卷积
            convw = conv2d_size_out(convw)
            convh = conv2d_size_out(convh)
        
        linear_input_size = convw * convh * 32  # 最终的线性层输入大小
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Qnet_back(nn.Module):
    def __init__(self, h, w, num_actions):
        super(Qnet_back, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * h * w, 128)
        self.fc2 = nn.Linear(128, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)