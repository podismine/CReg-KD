import torch
import torch.nn as nn
import torch.nn.functional as F


class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True, mode = 0):
        super(SFCN, self).__init__()

        self.conv1 = self.conv_layer(1, channel_number[0], maxpool=True, kernel_size=3,padding=1)
        self.conv2 = self.conv_layer(channel_number[0], channel_number[1], maxpool=True, kernel_size=3,padding=1)
        self.conv3 = self.conv_layer(channel_number[1], channel_number[2], maxpool=True, kernel_size=3,padding=1)
        self.conv4 = self.conv_layer(channel_number[2], channel_number[3], maxpool=True, kernel_size=3,padding=1)
        self.conv5 = self.conv_layer(channel_number[3], channel_number[4], maxpool=True, kernel_size=3,padding=1)
        self.conv6 = self.conv_layer(channel_number[4], channel_number[-1], maxpool=False, kernel_size=1,padding=0)

        self.clf = nn.Linear(64, output_dim)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(inplace = True),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace = True)
            )
        return layer

    def forward(self, x, out_features = False, df = False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        n = x.size(0)
        x = F.avg_pool3d(x, [2, 3, 2]).view(n, -1)
        x = F.dropout(x, 0.4, training=self.training)
        x = self.clf(x)
        if df is True:
            features = x.view(-1, 64*3*3*3)
            if out_features is True:
                return x.view(x.size(0), -1), features
            else:
                return x.view(x.size(0), -1)
        else:
            out3 = F.log_softmax(x.view(x.size(0), -1), dim=1)
            if out_features is True:
                return out3, features
            else:
                return out3

