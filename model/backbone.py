"""
Architecture crnn :
PaddlePaddleOCR input ImageShape:  3 × 32 × 100 or  3 × 48 × 192
backbone in/out: [Batch 3 32 100] == [Batch 2048 1 25]

"""
import torch
import torch.nn as nn


class ConvBnReLu(nn.Module):
    def __init__(self, in_ch, out_ch, k_size,
                 stride=1, groups=1, is_vd_mode=False,
                 act=True):
        super(ConvBnReLu, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=(k_size - 1) // 2, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rel = nn.ReLU()

    def forward(self, x):
        if self.act:
            return self.rel(self.bn(self.conv(x)))
        return self.bn(self.conv)


class BottleneckBlock(nn.Module):
    def __init__(self, in_ch,  out_ch, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBnReLu(in_ch, out_ch, k_size=1, act=True)
        self.conv1 = ConvBnReLu(in_ch=out_ch, out_ch=out_ch, k_size=3,
                                stride=stride, act=True)
        self.conv2 = ConvBnReLu(in_ch=out_ch, out_ch=out_ch * 4, k_size=1,
                                act=False)
        self.relu = nn.ReLU(inplace=True)

        if not shortcut:
            self.short = ConvBnReLu(in_ch=in_ch, out_ch=out_ch * 4, k_size=1,
                                    stride=stride)

        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.shortcut:
            identity = x
        else:
            identity = self.short(x)
        out += identity
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, layers=50):
        super(Resnet, self).__init__()
        assert layers in [18, 34, 50, 101, 152, 200]
        if layers == 18:                              depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:            depth = [3, 4, 6, 3]
        elif layers == 101:                           depth = [3, 4, 23, 3]
        elif layers == 152:                           depth = [3, 8, 36, 3]
        elif layers == 200:                           depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]
        self.layers = layers
        # head
        self.conv1 = ConvBnReLu(3, 32, 3)
        self.conv2 = ConvBnReLu(32, 32, 3)
        self.conv3 = ConvBnReLu(32, 64, 3)
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # blocks
        for block in depth:
            pass
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # x.shape Batch 3 32 100 or Batch 3 × 48 × 192
        return self.pool2d_max(self.conv3(self.conv2(self.conv1(x))))


if __name__ == "__main__":
    from torchvision.models import ResNet
    data = torch.rand(10, 3, 32, 100)
    # cbr = Resnet(50)
    # for k, v in cbr.named_parameters():
    #     print(k, "::::::::", v.shape)
    net = BottleneckBlock(64, 128, stride=1)
    for k, v in net.named_parameters():
        print(k, "::::::::", v.shape)
