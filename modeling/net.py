import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM
import random


class NormalHead(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(NormalHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)


class scoring(nn.Module):
    def __init__(self, in_dim, topk_rate=0.1):
        super(scoring, self).__init__()
        self.scoring = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)
        self.topk_rate = topk_rate

    def forward(self, x):
        x = self.scoring(x)
        x = x.view(int(x.size(0)), -1)
        topk = max(int(x.size(1) * self.topk_rate), 1)
        x = torch.topk(torch.abs(x), topk, dim=1)[0]
        x = torch.mean(x, dim=1).view(-1, 1)
        return x

class AnomalyHead(scoring):
    def __init__(self, in_dim, topk=0.1):
        super(AnomalyHead, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU())

    def forward(self, x):
        feats_map_size = x.shape[-1]
        block_size = feats_map_size // 2 + 1
        indexes = list(range(feats_map_size - block_size +1))
        index = random.choice(indexes)
        block1 = x[:, :, index:index+block_size, index:index+block_size]
        indexes.remove(index)
        index = random.choice(indexes)
        block2 = x[:, :, index:index + block_size, index:index + block_size]
        indexes.remove(index)
        index = random.choice(indexes)
        block3 = x[:, :, index:index + block_size, index:index + block_size]
        indexes.remove(index)
        index = random.choice(indexes)
        block4 = x[:, :, index:index + block_size, index:index + block_size]
        indexes.remove(index)
        residual = block4 - (block3 - (block2 - block1))
        residual = self.conv(residual)
        scores = super().forward(residual)
        return scores

class Model(nn.Module):
    def __init__(self, cfg, backbone="resnet18"):
        super(Model, self).__init__()
        self.cfg = cfg
        self.feature_extractor = build_feature_extractor(backbone, cfg)
        self.in_c = NET_OUT_DIM[backbone]
        self.normalhead = NormalHead(self.in_c)
        self.anomalyhead = AnomalyHead(self.in_c, self.cfg.topk)

    def forward(self, image):
        image_pyramid = list()
        for i in range(self.cfg.total_heads):
            image_pyramid.append(list())
        for s in range(self.cfg.n_scales):
            image_scaled = F.interpolate(image, size=self.cfg.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)

            normal_scores = self.normalhead(feature)
            anomaly_scores = self.anomalyhead(feature)

            for i, scores in enumerate([normal_scores, anomaly_scores]):
                image_pyramid[i].append(scores)
        for i in range(self.cfg.total_heads):
            image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
            image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)
        return image_pyramid


