import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

# 归一化点云：使用以centroid为中心，以1为半径的球体
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# 在球查询中计算点云中每个点与查询点之间的距离
# 输入两组点，输出两组点两两之间的距离，即n*m的矩阵
# 在训练中数据以mini-batch的形式输入，因此第1维为B
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# 按照输入的点云数据和索引，返回索引对应的数据
# 设points形状为[B, N, C]，若idx形状为[B, D1, ..., Dn]，则按idx中的维度结构将其提取为[B, D1, ..., Dn, C]
# 例如points形状为[B, 2408, 3]，idx=[5, 666, 1000, 2000]，则返回batch中第5、666、1000、2000个点组成的B * 4 * 3的点云
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# 最远点采样：从输入点云中按照所需的点的个数npoint采样出足够多的采样点，并且点与点之间的距离尽可能大
# 返回npoint个采样点在原始点云中的索引（找centroids）
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # centroids矩阵用于存储npoints个采样点的索引位置，大小为B * npoint
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance矩阵（B * N）用于存储batch中每个点与采样点之间的距离，初始化为一个很大的数
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest存储当前距离采样点最远的点的索引，随机初始化范围为0~N；初始化B个（每个batch都随机一个初始最远点）
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices用于存储当前batch中每个点所属的batch索引，初始化为0~(B-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 迭代npoint次，每次迭代中找到距离当前采样点最远的点，将其作为新的采样点
    for i in range(npoint):
        centroids[:, i] = farthest  # 将当前采样点（下标i）设为当前的最远点farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)    # 取出该中心点（centroid）的坐标
        dist = torch.sum((xyz - centroid) ** 2, -1) # 计算所有点到centroid的欧氏距离，存储于dist
        # 建立一个mask：若dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中保存的值会越来越小，其相当于记录着某batch中每个点与所有已出现采样点之间的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从distance矩阵取出的最远点为farthest，即当前的最远点，继续下一轮迭代
        farthest = torch.max(distance, -1)[1]
    return centroids


# 球查询：寻找球形邻域中的点
# 输入中radius为球半径，nsample为每个邻域中点要采样的点，xyz为所有点（[B, N, 3]），new_xyz为查询点（[B, S, 3]）
# 输出为每个样本的每个球形领域的nsample个采样点集的索引[B, S, nsample]
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx矩阵（[B, S, N]）用于存储每个查询点周围球查询中包含的点的索引
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists矩阵（[B, S, N]）记录S个中心点（new_xyz）与所有点（xyz）之间的欧几里得距离
    sqrdists = square_distance(new_xyz, xyz)
    # 将所有距离大于radius^2的点的索引直接置为N（球查询中不包含这些点），升序排序筛掉这些点，取出前nsample个点
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 当球形区域内不足nsample个点时，前nsample个点中亦会存在置为N的点，需被舍弃，可直接用第1个点来代替
    # group_first实为将第1个点复制至[B, S, K]的维度，便于后续替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask] # 替换
    return group_idx


# 采样+分组：将整个点云分散成局部的group，对每一个group都可以用PointNet单独提取局部特征
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: points sampled in farthest point sampling
        radius: search radius in local region
        nsample: how many points in each local region
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 进行最远点采样（FPS），挑出的采样点作为new_xyz：
    #   先用farthest_point_sample()最远点采样得到采样点的索引fps_idx
    #   再用index_points()根据索引从xyz中挑出采样点，作为new_xyz
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)    # 中心点（centroids）
    # 进行球查询
    #   先用query_ball_point()球查询得到npoint个球形区域中每个区域的nsample个采样点的索引，存储于idx（[B, npoint, nsample]）
    #   再用index_points()根据索引从xyz中挑出采样点，作为grouped_xyz
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # grouped_xyz减去采样点（即centroids）
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    # 若每个点上有新特征的维度，则拼接新特征与旧特征，否则直接为旧特征
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


# 直接将所有点作为一个group，即npoint=1
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# 普通的集合抽象层
# 首先通过sample_and_group()形成局部group，再对局部group中每一个点做MLP操作，最后进行局部的最大池化，得到局部的全局特征
# 其中mlp传入的是MLP的输出通道数，group_all指是否将所有点作为一个group
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # 形成局部group
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        # PointNet操作：对局部group中的每个点进行MLP操作（1x1卷积相当于全连接层）
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 局部的最大池化，得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points  # 返回采样点（centroids）和局部的全局特征


# 实现多尺度分组（MSG）方法的集合抽象层
# 其中radius_list、nsample_list、mlp_list传入的分别为不同尺度下的球查询半径列表、采样点数列表、MLP输出通道数二维列表
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        # 最远点采样
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        # new_points_list用于保存不同半径下的点云特征，最后拼接到一起
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # 球查询
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # 按照输入的点云数据和索引返回索引的点云数据
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # 拼接点特征数据和点坐标数据
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            # 最大池化，获得局部的全局特征
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 最后拼接不同半径下的点云特征
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


# Feature Propagation（分割时的上采样）主要通过线性插值（基于距离插值）和MLP完成
#   当点的个数为1时，使用repeat()直接复制成N个点
#   当点的个数大于1时，使用线性插值进行上采样
# 拼接上下采样对应点的SA层的特征，再对拼接后的每个点都做一个MLP
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:  # 直接复制
            interpolated_points = points2.repeat(1, N, 1)
        else:   # 线性插值
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)   # 距离越远的点权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 对于每个点的权重做全局归一化
            # 获得插值点
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            # 拼接上下采样前对应点SA层的特征    
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # 对拼接后每个点都做一个MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
