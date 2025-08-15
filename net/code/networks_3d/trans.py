import torch
from torch import nn


class Channelblock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Channelblock, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, 5, padding=2),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.GlobalAveragePooling3D = nn.AdaptiveAvgPool3d(1)

        self.function = nn.Sequential(
            nn.Linear(n_filters_out * 2, n_filters_out),
            nn.BatchNorm1d(n_filters_out),
            nn.ReLU(inplace=True),
            nn.Linear(n_filters_out,n_filters_out),
            nn.Sigmoid()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(n_filters_out * 2, n_filters_out, 1, padding=0),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )


    def forward(self, data):
        conv1 = self.convblock1(data)
        conv2 = self.convblock2(data)

        data3 = torch.cat([conv1,conv2], dim=1)
        b, c, w, h, d = conv2.shape
        data3 = self.GlobalAveragePooling3D(data3)
        data3 = data3.flatten(1)

        a = self.function(data3)
        a = a.reshape(b, c, 1, 1, 1)
        a1 = 1 - a

        y = torch.mul(conv1, a)
        y1 = torch.mul(conv2, a1)

        data_all = torch.cat([y, y1], dim=1)
        data_all = self.convblock3(data_all)

        return data_all


class Spatialblock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Spatialblock, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv3d(n_filters_out, n_filters_out, 1, padding=0),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(n_filters_out * 2, n_filters_out, 1, padding=0),
            nn.BatchNorm3d(n_filters_out),
        )

        self.function = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters_out, 1, 1, padding=0),
            nn.Sigmoid()
        )


    def expend_as(tensor, data, repnum):
        my_repeat = data.repeat(1, repnum, 1, 1, 1)

        return my_repeat

    def forward(self, data, data_all):
        data = self.convblock1(data)
        data = self.convblock2(data)

        data3 = data + data_all
        data3 = self.function(data3)

        a = self.expend_as(data3, data_all.shape[1])
        y = torch.mul(a, data_all)

        a1 = 1-data3
        a1 = self.expend_as(a1, data_all.shape[1])
        y1 = torch.mul(a1, data)

        data_pro = torch.cat([y, y1], dim=1)
        data_pro = self.convblock3(data_pro)

        return data_pro


class Attention_Block(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Attention_Block, self).__init__()
        self.conv11 = Channelblock(n_filters_in, n_filters_out)
        self.conv12 = Spatialblock(n_filters_in, n_filters_out)
        self.conv21 = Channelblock(n_filters_out, n_filters_out)
        self.conv22 = Spatialblock(n_filters_out, n_filters_out)


    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv12(x, x1)

        x3 = self.conv21(x2)
        x4 = self.conv22(x2, x3)
        return x4