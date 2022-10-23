import torch
from torch import nn
import time
import torch.nn.functional as F


def at(x):
    return F.normalize(x.pow(2).mean(1))


class target(nn.Module): 
    def __init__(self, feat_type='attention'):
        super(target, self).__init__()
        self.feat_type = feat_type
        
    def forward(self, CA, SA=None):
        if self.feat_type == 'attention':
            assert SA is not None
            return [CA, SA]
        elif self.feat_type == 'self_attention':
            assert SA is None
            return at(CA)
        elif self.feat_type == 'feature':
            assert SA is None
            return CA
        else:
            raise('Select Proper Target')
    
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CAModule(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        attn_out = x
        x = self.sigmoid(x)
        return input * x, attn_out


class SAModule(nn.Module):
    '''Spatial Attention Module'''
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        attn_out = x
        x = self.sigmoid(x)
        return input * x, attn_out


class BottleNeck_IR_CBAM(nn.Module):
    '''Improved Residual Bottleneck with Channel Attention Module and Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_CBAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))

        self.channel_layer = CAModule(out_channel, 16)
        self.spatial_layer = SAModule()
        self.attention_target = target(feat_type='attention')

        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )


    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        # Target for A-SKD
        res, att_c = self.channel_layer(res)
        res, att_s = self.spatial_layer(res)
        _ = self.attention_target(att_c, att_s)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
            
        return shortcut + res


filter_list = [64, 64, 128, 256, 512]
def get_layers(num_layers):
    if num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]

class CBAMResNet(nn.Module):
    def __init__(self, num_layers, feature_dim=512, drop_ratio=0.4, mode='cbam',filter_list=filter_list):
        super(CBAMResNet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        layers = get_layers(num_layers)
        block = BottleNeck_IR_CBAM

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        self.feature_target = target(feat_type='feature')
        self.self_attention_target = target(feat_type='self_attention')
        
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x1 = self.layer1(x)
        _ = self.self_attention_target(x1)
        
        x2 = self.layer2(x1)
        _ = self.self_attention_target(x2)
        
        x3 = self.layer3(x2)
        _ = self.self_attention_target(x3)
        
        x4 = self.layer4(x3)
        _ = self.self_attention_target(x4)
        
        out = self.output_layer(x4)
        _ = self.feature_target(out)
        return out