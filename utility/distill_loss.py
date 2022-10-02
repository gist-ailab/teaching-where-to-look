import torch.nn as nn
import torch
import torch.nn.functional as F

class DistillLoss(nn.Module):
    def __init__(self, args, device):
        super(DistillLoss, self).__init__()
        self.device = device
        self.args = args
        self.criterion = self.cosine_loss

    def cosine_loss(self, l , h):
        l = l.view(l.size(0), -1)
        h = h.view(h.size(0), -1)
        return torch.mean(1.0 - F.cosine_similarity(l, h))

    
    def forward(self, low_feature, high_feature):
        # calculate the attention distillation
        loss_sum = 0.
        for l, h in zip(low_feature, high_feature):
            l = l.reshape(l.size(0), -1)
            h = h.reshape(h.size(0), -1)
            d_loss = self.criterion(l, h)

            loss_sum += d_loss * self.args.distill_attn_param
                
        return loss_sum
    