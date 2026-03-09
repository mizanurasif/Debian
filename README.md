class DecoderBlockDynamic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction_rate: int = 2,
        negative_slope: float = 0.,
        num_groups:int = 4,
        add_dropout:bool = False,
    ):
        super().__init__()

        self.Transpose = nn.ConvTranspose3d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=2, stride=2)

        # Projection 1x1
        #self.proj =  nn.Conv3d(in_channels*2, in_channels, kernel_size=1, bias=False)
        self.conv_block1 = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=in_channels*2),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                DynamicConvolution3D(nof_kernels=4,reduce=1,in_channels=in_channels*2,out_channels=in_channels*2,kernel_size=3,padding=1,bias=False)
            )
        # --- New Fusion: Dilated Convs ---
        #self.branch1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        #self.branch2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        #self.branch3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, bias=False)
        
        self.conv_block2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels*2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            DynamicConvolution3D(nof_kernels=4,reduce=1,in_channels=in_channels*2,out_channels=out_channels,kernel_size=3,padding=1,bias=False)
        )
 
        self.res_block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels*2),
            DynamicConvolution3D(nof_kernels=4,reduce=1,in_channels=in_channels*2,out_channels=out_channels,kernel_size=3,padding=1,bias=False)
        )

    def forward(self, skip_unet: torch.Tensor | None, down_feat: torch.Tensor):
        branches = []
        down_feat = self.Transpose(down_feat)
        branches = [skip_unet, down_feat]
        x = torch.cat(branches, dim=1)
        x2 = self.conv_block1(x) 
        out = self.conv_block2(x2)
        res = self.res_block(x)

        return out+res
