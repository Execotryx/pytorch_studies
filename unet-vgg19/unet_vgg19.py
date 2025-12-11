import torch
import torch.nn as nn
from torchvision.models import vgg19

class UNetVGG19(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        super().__init__()

        vgg = vgg19(weights="IMAGENET1K_V1" if pretrained else None).features
        vgg_blocks: nn.Sequential = vgg

        # Extract VGG19 encoder blocks (conv layers only, excluding pooling)
        # VGG19 structure: conv-conv-pool-conv-conv-pool-conv-conv-conv-conv-pool-conv-conv-conv-conv-pool-conv-conv-conv-conv-pool
        self.enc1 = vgg_blocks[:2]   # 64 channels (conv1_1, conv1_2)
        self.enc2 = vgg_blocks[5:7]  # 128 channels (conv2_1, conv2_2) 
        self.enc3 = vgg_blocks[10:14] # 256 channels (conv3_1, conv3_2, conv3_3, conv3_4)
        self.enc4 = vgg_blocks[19:23] # 512 channels (conv4_1, conv4_2, conv4_3, conv4_4)
        self.enc5 = vgg_blocks[28:32] # 512 channels (conv5_1, conv5_2, conv5_3, conv5_4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder blocks with correct channel dimensions
        self.up4 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec4 = self._conv_block(1024, 512)  # 512 (up) + 512 (skip) = 1024

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self._conv_block(512, 256)   # 256 (up) + 256 (skip) = 512

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self._conv_block(256, 128)   # 128 (up) + 128 (skip) = 256

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self._conv_block(128, 64)    # 64 (up) + 64 (skip) = 128

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        e1 = self.enc1(x)  # 64 channels
        x = self.pool(e1)

        e2 = self.enc2(x)  # 128 channels
        x = self.pool(e2)

        e3 = self.enc3(x)  # 256 channels
        x = self.pool(e3)

        e4 = self.enc4(x)  # 512 channels
        x = self.pool(e4)

        e5 = self.enc5(x)  # 512 channels (bottleneck)
        
        # Decoder path with skip connections
        x = self.up4(e5)  # Upsample to match e4 size
        x = torch.cat([x, e4], dim=1)  # Concatenate skip connection
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)  # Apply final activation
        return x