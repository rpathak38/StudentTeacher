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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, last=False):
        super().__init__()
        if last is False:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(out_channels),
                # nn.Sigmoid()
            )

    def forward(self, x):
        # x should (N, C, H, W)
        return self.double_conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetAttn(nn.Module):
    """Unet inspired architecture.
    Using same convolutions, with output channels being equal to the number of classes. Adding instead of
    appending. Upsampling with MaxUnpooling instead of transpose convolutions.

    Attributes:
        in_channels: The number of input channels.
        out_channels: The number of output classes (including background).
        channel_list: A list representing the intermediate channels that we have. The bottleneck (bottom of the U) outputs 2 * the last channel.
    """

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
            
        self.attention_gates = nn.ModuleList()
        for i in range(len(channel_list)):
            self.attention_gates.append(AttentionGate(channel_list[i], channel_list[i], channel_list[i]//2))
            # self.attention_gates.append(AttentionGate(channel_list[i], channel_list[i], channel_list[i]))

        self.bottleneck = DoubleConv(curr_channel, curr_channel * 2, curr_channel)

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channel_list))):
            if i - 1 < 0:
                self.ups.append(DoubleConv(channel_list[i]*2, channel_list[i]*2, out_channels, last=True))
            else:
                self.ups.append(DoubleConv(channel_list[i]*2, channel_list[i]*2, channel_list[i - 1]))
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        # x = self.normalize(x)
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
            # if index < len(self.ups) - 1:
            #     attn = self.attention_gates[-index - 1](down_activations[-index - 1], x)
            #     temp = x + attn
            # else:
            #     temp = x
            attn = self.attention_gates[-index - 1](down_activations[-index - 1], x)
            temp = torch.cat((attn, x), dim=1)
            x = up(temp)
        return x
