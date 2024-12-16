import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    Double convolution block used in the U-Net architecture.

    According to the original U-Net paper (Ronneberger et al., 2015),
    each stage of the contracting and expanding paths uses two successive
    3x3 convolutions, each followed by ReLU (or similar) activation. 
    Here, we're also using BatchNorm2d, which was not in the original 
    paper but is a common modern improvement.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Note: We use padding=1 to maintain spatial dimensions 
        # (this slightly deviates from the 'valid' convolution in the original paper 
        # but is commonly used for convenience without significantly affecting performance).
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Implementation.

    The U-Net architecture (Ronneberger et al., 2015) consists of 
    a contracting path (encoder) to capture context and a symmetric 
    expanding path (decoder) that enables precise localization. 
    Skip connections between mirrored layers in the two paths help 
    the network localize features better.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        :param in_channels:  Number of input channels to the network (e.g. 1 for grayscale, 3 for RGB).
        :param out_channels: Number of output segmentation classes (e.g. 1 for binary segmentation).
        :param features:     List defining the number of feature maps at each level of the U-Net.
        """
        super(UNet, self).__init__()
        
        # Contracting path layers (encoder)
        self.down_layers = nn.ModuleList()
        # Expanding path layers (decoder)
        self.up_layers = nn.ModuleList()
        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Build contracting path
        # Each DoubleConv is followed by a max-pool (except for the last).
        # The original paper uses 'valid' convolutions (no padding),
        # but here we use padded convolutions for simpler shape management.
        prev_channels = in_channels
        for feature in features:
            self.down_layers.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck part (bottom of the "U")
        # In the original U-Net paper, after descending the contracting path,
        # there is a bottom layer that also consists of two 3x3 convolutions.
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Build expanding path
        # Each step in the decoder consists of a transposed convolution (for upsampling)
        # followed by the concatenation of the corresponding feature map from the contracting path,
        # and then two 3x3 convolutions (DoubleConv).
        for feature in reversed(features):
            self.up_layers.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.up_layers.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution: reduces the feature maps to out_channels
        # (e.g., for binary segmentation, out_channels=1).
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Skip connections will store outputs from each stage of the contracting path
        # to be used in the expanding path.
        skip_connections = []

        # Contracting path
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the order of skip connections so that they match up with the up_layers
        skip_connections = skip_connections[::-1]

        # Expanding path
        # Note: up_layers consists of pairs -> (ConvTranspose2d, DoubleConv)
        for idx in range(0, len(self.up_layers), 2):
            # Transposed convolution for upsampling
            x = self.up_layers[idx](x)
            # Corresponding skip connection from the contracting path
            skip_connection = skip_connections[idx // 2]

            # Ensure x is the same spatial size as skip_connection 
            # (in case of any size mismatch due to rounding in down/up sampling).
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # Concatenate skip connection along channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # DoubleConv on the concatenated features
            x = self.up_layers[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    """Quick test to check output shape consistency."""
    x = torch.rand((3, 1, 161, 161))  # (batch_size=3, channels=1, height=161, width=161)
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", preds.shape)
    assert preds.shape == x.shape, "Output shape does not match input shape"
    print("Test passed!")


if __name__ == '__main__':
    test()
