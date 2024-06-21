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