import torch
import torch.nn as nn
import torch.nn.functional as F

def fill_shells_torch(n=200, n_shells=100, values=None, device=None, dtype=torch.float32):
    """
    Hard shell assignment. Differentiable w.r.t. `values`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if values is None:
        values = torch.arange(n_shells, device=device, dtype=dtype)
    else:
        values = torch.as_tensor(values, device=device, dtype=dtype)
        n_shells = len(values)

    assert values.numel() == n_shells, "Need exactly one value per shell"

    # centered coordinates
    grid = torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0
    x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")
    r = torch.sqrt(x * x + y * y + z * z)

    # edges in radius
    r_max = r.max()
    shell_edges = torch.linspace(0.0, r_max, n_shells + 1, device=device, dtype=dtype)

    # bucketize gives bin in [0..n_shells]; subtract 1 -> [ -1 .. n_shells-1 ]
    shell_idx = torch.bucketize(r, shell_edges, right=False) - 1
    shell_idx = shell_idx.clamp(0, n_shells - 1).long()

    # index values
    box = values[shell_idx]
    return box



class ConvBlock(nn.Module):
    """Convolution blocks of the form specified by `seq`.

    `seq` types:
    'C': convolution specified by `kernel_size` and `stride`
    'B': normalization (to be renamed to 'N')
    'A': activation
    'U': upsampling transposed convolution of kernel size 2 and stride 2
    'D': downsampling convolution of kernel size 2 and stride 2
    """
    def __init__(self, in_chan, out_chan=None, mid_chan=None,
            kernel_size=3, stride=1, seq='CBA'):
        super().__init__()

        if out_chan is None:
            out_chan = in_chan

        self.in_chan = in_chan
        self.out_chan = out_chan
        if mid_chan is None:
            self.mid_chan = max(in_chan, out_chan)
        else: self.mid_chan = mid_chan
            
        self.kernel_size = kernel_size
        self.stride = stride

        self.norm_chan = in_chan
        self.idx_conv = 0
        self.num_conv = sum([seq.count(l) for l in ['U', 'D', 'C']])

        layers = [self._get_layer(l) for l in seq]

        self.convs = nn.Sequential(*layers)

    def _get_layer(self, l):
        if l == 'U':
            in_chan, out_chan = self._setup_conv()
            return nn.ConvTranspose3d(in_chan, out_chan, 2, stride=2)
        elif l == 'D':
            in_chan, out_chan = self._setup_conv()
            return nn.Conv3d(in_chan, out_chan, 2, stride=2)
        elif l == 'C':
            in_chan, out_chan = self._setup_conv()
            return nn.Conv3d(in_chan, out_chan, self.kernel_size,
                    stride=self.stride, padding='same')
        elif l == 'B':
            return nn.BatchNorm3d(self.norm_chan)
#             return nn.InstanceNorm3d(self.norm_chan, affine=True, track_running_stats=True)
#             return nn.InstanceNorm3d(self.norm_chan)
        elif l == 'A':
            return nn.LeakyReLU()
        else:
            raise ValueError('layer type {} not supported'.format(l))

    def _setup_conv(self):
        self.idx_conv += 1

        in_chan = out_chan = self.mid_chan
        if self.idx_conv == 1:
            in_chan = self.in_chan
        if self.idx_conv == self.num_conv:
            out_chan = self.out_chan

        self.norm_chan = out_chan

        return in_chan, out_chan

    def forward_old(self, x):
        return self.convs(x)

    
    def forward(self, x, t_emb=None):
        h = x
        for layer in self.convs:
            if isinstance(layer, nn.BatchNorm3d):
                h = layer(h)
                if t_emb!=None:
                    # inject BEFORE activation
                    h = h + t_emb
            else:
                h = layer(h)
        return h

    
    
    
class UNet_new(nn.Module):
    def __init__(self, in_chan, out_chan, cond_chan, bypass=False, base_channels=32, box_side=200, spherical_noise=False, **kwargs):
        super().__init__()
        self.box_side = box_side
        self.spherical_noise = spherical_noise
        total_in = cond_chan
        # self.time_emb = TimestepEmbedding(base_channels)              # embedding
        # self.time_proj = nn.Linear(base_channels * 4, base_channels)  # project embedding to match layer outputs

        self.conv_l0 = ConvBlock(total_in, base_channels, seq='CACBA')
        self.down_l0 = ConvBlock(base_channels, seq='DBA')
        self.conv_l1 = ConvBlock(base_channels, seq='CBACBA')
        self.down_l1 = ConvBlock(base_channels, seq='DBA')

        self.conv_c = ConvBlock(base_channels, seq='CBACBA')

        self.up_r1 = ConvBlock(base_channels, seq='UBA')
        self.conv_r1 = ConvBlock(base_channels*2, base_channels, base_channels, seq='CBACBA') # change middle channel here (reduce by a factor of 2)
        self.up_r0 = ConvBlock(base_channels, seq='UBA')
        self.conv_r0 = ConvBlock(base_channels*2, out_chan, seq='CAC')

        self.bypass = in_chan == out_chan

        if self.spherical_noise == True:
            self.extra_output = nn.Parameter(
            torch.full((self.box_side//2,), 0.1e2, dtype=torch.float32),
            requires_grad=True
            )
        else:   
            self.extra_output = nn.Parameter(
            torch.full((self.box_side**3,), 0.1e2, dtype=torch.float32),
            requires_grad=True
            )
        # self.extra_head = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),  # (B, C, 1,1,1)
        #     nn.Flatten(),             # (B, C)
        #     nn.Linear(base_channels, base_channels),
        #     nn.SiLU(),
        #     nn.Linear(base_channels, box_side**3),
        # )

    def forward(self, x):
        # Time embedding
#         if self.bypass:
#             x0 = x

        # x = torch.cat([x, cond], dim=1)
    
        # Encoder
        y0 = self.conv_l0(x)
        x = self.down_l0(y0)
        

        y1 = self.conv_l1(x)
        x = self.down_l1(y1)

                      # (N, C, 1,1,1)
            
        # Bottleneck
        x = self.conv_c(x)
#         x = x + t_emb  # inject
        # extra = self.extra_head(x)   # (B, box_side**3)


        # Decoder
        x = self.up_r1(x)
#         y1 = narrow_by(y1, 4)
        x = torch.cat([y1, x], dim=1) # Skip connection
        x = self.conv_r1(x)

        x = self.up_r0(x)
#         y0 = narrow_by(y0, 2)
        x = torch.cat([y0, x], dim=1) # Skip connection
        x = self.conv_r0(x)

#         if self.bypass:
#             x += x0
        # logdiag = out[:,1:2].clamp(-20,10)
        # diag_flat = torch.exp(logdiag).flatten(1)
        # return mean, diag_flat

        # return x, abs(extra)


        if self.spherical_noise == True:
            box = fill_shells_torch(n=self.box_side, values = self.extra_output)
            box = torch.fft.fftshift(box)
            box = box.flatten()
            box = torch.nn.functional.softplus(box) + 1e-6
            box = box.unsqueeze(0).expand(x.size(0), -1)
            return x, box
        else:
            extra = torch.nn.functional.softplus(self.extra_output) + 1e-6
            extra = extra.unsqueeze(0).expand(x.size(0), -1)
            # extra = abs(self.extra_output).unsqueeze(0).expand(x.size(0), -1)
            return x, extra
        # # self.extra_output.unsqueeze(0).expand(x.size(0), -1)
        # out = x
        # box1 = out[:, 0:1 , :, :, :]
        # # # box2 must flatten to (B, D*H*W)
        # box2 = out[:, 1, :, :, :].reshape(x.size(0), -1)   # (B, box_side**3)
        # # print('box2 shape:', box2.shape)
        # # print(box2[:10])
        
        # # return box1, torch.exp(box2)
        # # # return box1, box2
        # logvar = out[:, 1, :, :, :].reshape(x.size(0), -1)                   # (B,1,D,H,W)
        # logvar = logvar.clamp(-20.0, 10.0)            # IMPORTANT
        # sigma  = torch.exp(0.5 * logvar)              # (B,1,D,H,W)

        # # flatten for your loss
        # sigma_flat = sigma.reshape(sigma.size(0), -1) # (B, D*H*W)

        # return box1, sigma_flat/ 100



class UNet_new_Res(nn.Module):
    def __init__(self, in_chan, out_chan, cond_chan, bypass=False, base_channels=32, box_side=200, spherical_noise=False, **kwargs):
        super().__init__()
        self.box_side = box_side
        self.spherical_noise = spherical_noise
        total_in = cond_chan
        # self.time_emb = TimestepEmbedding(base_channels)              # embedding
        # self.time_proj = nn.Linear(base_channels * 4, base_channels)  # project embedding to match layer outputs

        self.conv_l0 = ConvBlock(total_in, base_channels, seq='CACBA')
        self.down_l0 = ConvBlock(base_channels, seq='DBA')
        self.conv_l1 = ConvBlock(base_channels, seq='CBACBA')
        self.down_l1 = ConvBlock(base_channels, seq='DBA')

        self.conv_c = ConvBlock(base_channels, seq='CBACBA')

        self.up_r1 = ConvBlock(base_channels, seq='UBA')
        self.conv_r1 = ConvBlock(base_channels*2, base_channels, base_channels, seq='CBACBA') # change middle channel here (reduce by a factor of 2)
        self.up_r0 = ConvBlock(base_channels, seq='UBA')
        self.conv_r0 = ConvBlock(base_channels*2, out_chan, seq='CAC')

        self.bypass = in_chan == out_chan

        if self.spherical_noise == True:
            self.extra_output = nn.Parameter(
            torch.full((self.box_side//2,), 0.1e2, dtype=torch.float32),
            requires_grad=True
            )
        else:   
            self.extra_output = nn.Parameter(
            torch.full((self.box_side**3,), 0.1e2, dtype=torch.float32),
            requires_grad=True
            )


    def forward(self, x):
        # Time embedding
#         if self.bypass:
#             x0 = x

        # x = torch.cat([x, cond], dim=1)
    
        # Encoder
        y0 = self.conv_l0(x)
        x = self.down_l0(y0)
        

        y1 = self.conv_l1(x)
        x = self.down_l1(y1)

                      # (N, C, 1,1,1)
            
        # Bottleneck
        x = self.conv_c(x)
#         x = x + t_emb  # inject
        # extra = self.extra_head(x)   # (B, box_side**3)


        # Decoder
        x = self.up_r1(x)
#         y1 = narrow_by(y1, 4)
        x = torch.cat([y1, x], dim=1) # Skip connection
        x = self.conv_r1(x)

        x = self.up_r0(x)
#         y0 = narrow_by(y0, 2)
        x = torch.cat([y0, x], dim=1) # Skip connection
        x = self.conv_r0(x)

#         if self.bypass:
#             x += x0
        # logdiag = out[:,1:2].clamp(-20,10)
        # diag_flat = torch.exp(logdiag).flatten(1)
        # return mean, diag_flat

        # return x, abs(extra)


        if self.spherical_noise == True:
            box = fill_shells_torch(n=self.box_side, values = self.extra_output)
            box = torch.fft.fftshift(box)
            box = box.flatten()
            box = torch.nn.functional.softplus(box) + 1e-6
            box = box.unsqueeze(0).expand(x.size(0), -1)
            return x, box
        else:
            extra = torch.nn.functional.softplus(self.extra_output) + 1e-6
            extra = extra.unsqueeze(0).expand(x.size(0), -1)
            # extra = abs(self.extra_output).unsqueeze(0).expand(x.size(0), -1)
            return x, extra
        # # self.extra_output.unsqueeze(0).expand(x.size(0), -1)
        # out = x
        # box1 = out[:, 0:1 , :, :, :]
        # # # box2 must flatten to (B, D*H*W)
        # box2 = out[:, 1, :, :, :].reshape(x.size(0), -1)   # (B, box_side**3)
        # # print('box2 shape:', box2.shape)
        # # print(box2[:10])
        
        # # return box1, torch.exp(box2)
        # # # return box1, box2
        # logvar = out[:, 1, :, :, :].reshape(x.size(0), -1)                   # (B,1,D,H,W)
        # logvar = logvar.clamp(-20.0, 10.0)            # IMPORTANT
        # sigma  = torch.exp(0.5 * logvar)              # (B,1,D,H,W)

        # # flatten for your loss
        # sigma_flat = sigma.reshape(sigma.size(0), -1) # (B, D*H*W)

        # return box1, sigma_flat/ 100




class Simple3DConvNet(nn.Module):
    def __init__(self, BOX_SIDE=200):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding='same')
        )
        
        # This is your global learnable vector (not dependent on input)
#         self.extra_output = nn.Parameter(torch.randn(50*50*50))  # e.g. 50-dimensional
        self.extra_output = nn.Parameter(
        torch.full((BOX_SIDE**3,), 0.1e2, dtype=torch.float32),
        requires_grad=True
    )


    def forward(self, x, cdn=None):
        batch_size = x.size(0)
        # Expand extra_output to shape (B, N)
        extra = F.softplus(self.extra_output) + 1e-6
        diag_values = extra.unsqueeze(0).expand(x.size(0), -1)
        # diag_values = self.extra_output.unsqueeze(0).expand(batch_size, -1)
        return self.model(x), diag_values
