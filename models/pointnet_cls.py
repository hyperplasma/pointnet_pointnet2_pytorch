import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:  # 使用法向量
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)    # k为类别数（默认为40）
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1) # 计算对数概率
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # F.nll_loss(pred, target)输入一个对数概率张量和一个目标标签，不会做softmax，返回一个标量损失
        # 适用于最后一层为log_softmax的模型
        # 常用的交叉熵损失函数与此类似，唯一不同是其会做softmax
        loss = F.nll_loss(pred, target) # 分类损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)   # 特征变换正则化损失

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
