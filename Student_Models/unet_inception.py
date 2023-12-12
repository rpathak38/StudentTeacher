import torch
from torch import nn

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Register mean and std as buffers
        self.register_buffer('mean', mean.reshape(1, -1, 1, 1))
        self.register_buffer('std', std.reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        mid_channels = out_channels // 3

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels - 2 * mid_channels, kernel_size=5, stride=1, padding=2)

        # Initialize the convolutional layers with default initialization
        self.conv1.apply(self._init_weights)
        self.conv3.apply(self._init_weights)
        self.conv5.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        return torch.cat([conv1_out, conv3_out, conv5_out], dim=1)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, last=False):
        super().__init__()
        if last is False:
            self.double_conv = nn.Sequential(
                InceptionBlock(in_channels, mid_channels),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                InceptionBlock(mid_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                InceptionBlock(in_channels, mid_channels),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                InceptionBlock(mid_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                # nn.Sigmoid()
            )

        print(f"Expected: {in_channels} -> {out_channels}")

    def forward(self, x):
        return self.double_conv(x)



class UNetInception(nn.Module):
    def __init__(self, in_channels, out_channels, channel_list, means=None, stds=None):
        super().__init__()
        if means is None:
            means = torch.tensor([0.0 for _ in range(in_channels)])
        if stds is None:
            stds = torch.tensor([1.0 for _ in range(in_channels)])
        self.normalize = Normalize(mean=means, std=stds)
        self.downs = nn.ModuleList()
        curr_channel = in_channels
        for intermediate_channel in channel_list:
            self.downs.append(DoubleConv(curr_channel, intermediate_channel, intermediate_channel))
            curr_channel = intermediate_channel

        self.bottleneck = DoubleConv(curr_channel, curr_channel * 2, curr_channel)

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channel_list))):
            if i - 1 < 0:
                self.ups.append(DoubleConv(channel_list[i], channel_list[i], out_channels, last=True))
            else:
                self.ups.append(DoubleConv(channel_list[i], channel_list[i], channel_list[i - 1]))
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        x = self.normalize(x)
        pool_outs = []
        down_activations = []
        for down in self.downs:
            x = down(x)
            down_activations.append(x)

            x, indices = self.pool(x)
            pool_outs.append(indices)

        x = self.bottleneck(x)
        for index, up in enumerate(self.ups):
            x = self.unpool.forward(x, pool_outs[-index - 1])
            temp = x + down_activations[-index - 1]
            x = up(temp)
        return x
