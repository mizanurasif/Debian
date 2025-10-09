class DecoderBlock3DNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction_rate: int = 2,
        negative_slope: float = 0.01
    ):
        super().__init__()

        self.Transpose = nn.ConvTranspose3d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=2, stride=2)

        # Projection 1x1
        self.proj =  nn.Conv3d(in_channels*3, in_channels, kernel_size=1, bias=False)

        self.norm_act = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

        # --- New Fusion: Dilated Convs ---
        self.branch1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, bias=False)
        
        self.conv_block = nn.Sequential(
            nn.InstanceNorm3d(in_channels*3, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv3d(in_channels*3, out_channels, kernel_size=3, padding=1, bias=False)
        )
 
        self.res_block = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, skip_unet: torch.Tensor | None, skip_swin: torch.Tensor | None, down_feat: torch.Tensor):
        branches = []
        down_feat = self.Transpose(down_feat)

        branches = [skip_unet, skip_swin, down_feat]
        x = torch.cat(branches, dim=1)
        proj=self.proj(x)
        norm_poj=self.norm_act(proj) 
        # dilated conv branches
        b1 = self.branch1(norm_poj)
        b2 = self.branch2(norm_poj)
        b3 = self.branch3(norm_poj)

        # concat dilated outputs
        x2 = torch.cat([b1, b2, b3], dim=1)

        # norm + activation + reduce back
        out = self.conv_block(x2)
        res = self.res_block(down_feat)

        return out+res
