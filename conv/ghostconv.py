class FusedMBConvN(nn.Module):
    
    def __init__(
        self,
        n_in, # In_channels
        n_out, # out_channels
        k_size = 3, # kernel_size
        stride = 1,
        expansion_factor = 4,
        reduction_factor = 4, # SqueezeExcitation Block
        survival_prob = 0.8 # StochasticDepth Block
    ):
        super(FusedMBConvN, self).__init__()
        
        reduced_dim = int(n_in//4)
        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1)//2
        
        self.use_residual = (n_in == n_out) and (stride == 1)
        #self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded_dim, k_size = 1)
        self.conv = ConvBnAct(n_in, expanded_dim,
                              k_size, stride = stride,
                              padding = padding, groups = 1
                             )
        #self.se = SqueezeExcitation(expanded_dim, reduced_dim)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = nn.Identity() if (expansion_factor == 1) else ConvBnAct(expanded_dim, n_out, k_size = 1, act = False)
        
    def forward(self, x):
        
        residual = x.clone()
        #x = self.conv(x)
        x = self.conv(x)
        #x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            x += residual
            
        return x




### Squeeze Excitation Block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


### DWConv + BN + ReLU
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )
    

### Ghost Bottleneck

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)