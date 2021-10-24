import torch
import torch.nn as nn
from model import resnet50, SequenceEncoder, CTCHead


class CRNN_ResNet50_LSTM(nn.Module):
    def __init__(self, num_classes=120):
        super(CRNN_ResNet50_LSTM, self).__init__()
        self.backbone = resnet50()
        self.neck = SequenceEncoder(in_channels=2048)
        self.head = CTCHead(96, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from torchsummary.torchsummary import summary
    data = torch.rand(1, 3, 32, 100)
    net = CRNN_ResNet50_LSTM()
    out = net(data)
    print(out.shape)
    for k, v in net.named_parameters():
        print(k, "::::::::", v.shape)
    # print(out.shape)
    # summary(net, input_size=(3, 32, 100))
