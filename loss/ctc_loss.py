import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def forward(self, predicts, batch):
        predicts = predicts.permute(1, 0, 2)
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B)
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        loss = loss.mean()  # sum
        return loss
