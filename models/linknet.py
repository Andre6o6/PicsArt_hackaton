import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50

class UNetResnet(torch.nn.Module):
    def __init__(self, num_classes=1, encoder='resnet34', mode='sum', first_skip=False):
        super().__init__()
        
        if encoder=='resnet34':
            self.encoder = resnet34(pretrained=True)
            filters = 64
        else:
            self.encoder = resnet50(pretrained=True)
            filters = 256
        self.relu = nn.ReLU(inplace=True)
        
        self.up1 = DecoderBlock(filters*8, filters*4, mode)
        self.up2 = DecoderBlock(filters*4, filters*2, mode)
        self.up3 = DecoderBlock(filters*2, filters, mode)
        self.up4 = DecoderBlock(filters, 64, mode)
        
        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.last = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        blocks = []
        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        if first_skip:
            blocks.append(x)
        else:
            blocks.append(None)
            
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        blocks.append(x)
        
        x = self.encoder.layer2(x)
        blocks.append(x)
        
        x = self.encoder.layer3(x)
        blocks.append(x)
        
        x = self.encoder.layer4(x)
        
        x = self.up1(x, blocks[-1])
        
        x = self.up2(x, blocks[-2])
        
        x = self.up3(x, blocks[-3])
        
        x = self.up4(x, blocks[-4])
        
        x = self.upconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.last(x)
        x = torch.sigmoid(x)
        return x



class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mode='sum'):
        super().__init__()
        
        self.mode = mode
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        if mode == 'cat':
            inner_channels = 2*out_channels
        else:
            inner_channels = out_channels
            
        self.block = nn.Sequential(            
            nn.Conv2d(inner_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
        
    def forward(self, x, bridge=None):
        out = self.up(x)
        
        if bridge is not None:
            out = self.center_crop(out, bridge.shape[2:])
        
        shortcut = out
        
        if bridge is not None and self.mode == 'cat':   #concatenation should be here
            out = torch.cat((out, bridge), 1)          
            
        out = self.block(out)
        
        out = out + shortcut
        
        if bridge is not None and self.mode == 'sum':   #...but addition here
            out = out + bridge
           
        return out
