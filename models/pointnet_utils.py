import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# STN3d: T-Net 3 * 3 transform
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)    # 9 = 3 * 3
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]    # Symmetric function: max pooling
        x = x.view(-1, 1024)    # 参数拉直（展平）

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 展平的单位矩阵：np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden    # affine transformation（仿射变换）
        x = x.view(-1, 3, 3)    # batch_size * 3 * 3
        return x


# STNkd: T-Net k * k transform (k默认为64)
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # np.eye()：生成单位矩阵
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# PointNet编码器：input points (-> local embedding, if seg) -> global feature
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)   # T-Net 3 * 3 transform
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:  # 如果需要特征转换，则使用T-Net 64 * 64 transform
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()  # batch_size, 3（xyz坐标）或6（xyz坐标+法向量）, 1024（一个物体所取点的数目）
        trans = self.stn(x)
        x = x.transpose(2, 1)   # 转置，交换维度
        if D > 3:   # 如果输入点云包含法向量，则将法向量与坐标分开，法向量是第4维及之后的数据，坐标是前3维
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        # 对输入的点云进行输入转换（input transform）：对两个tensor进行矩阵乘法
        # torch.bmm()要求两个tensor的维度为[batch_size, n, m]和[batch_size, m, p]，返回的tensor维度为[batch_size, n, p]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # 将转换后的坐标与法向量拼接
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)    # 对输入的点云进行特征转换（feature transform）
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x   # 局部特征
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]    # 最大池化获取全局特征
        x = x.view(-1, 1024)
        if self.global_feat:    # 如果需要全局特征，则返回全局特征x
            return x, trans, trans_feat
        else:   # 否则返回全局特征和局部特征的拼接（torch.cat()）
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# 对特征转换矩阵做正则化（将特征变换矩阵约束为接近正交矩阵，$L_{reg}=||I-AA^T||^2_F$）
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]    # torch.eye(n, m=None, out=None)：返回一个2维单位矩阵
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))  # 正则化损失函数
    return loss
