import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal


class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y

    
class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out    
 

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DiffeomorphicTransform(nn.Module):
    def __init__(self, size, mode='bilinear', time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode
        self.time_step = time_step

    def forward(self, velocity):
        flow = velocity / (2.0 ** self.time_step)
        # 1.0 flow
        for _ in range(self.time_step):
            new_locs = self.grid + flow
            shape = flow.shape[2:]
            # Need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)
                new_locs = new_locs[..., [1, 0]]
            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)
                new_locs = new_locs[..., [2, 1, 0]]
            flow = flow + nnf.grid_sample(flow, new_locs, align_corners=True, mode=self.mode)
        return flow

    
class Lagrangian_flow(nn.Module):
    """
    [Lagrangian_flow] is a class representing the computation of Lagrangian flow (v12, v13, v14, ...) from inter frame
    (INF) flow filed (u12, u23, u34, ...)
    v12 = u12
    v13 = v12 + u23 o v12 ('o' is a warping)
    v14 = v13 + u34 o v13
    ...
    """

    def __init__(self, vol_size):
        """
        Instiatiate Lagrangian_flow layer
            :param vol_size: volume size of the atlas
        """
        super(Lagrangian_flow, self).__init__()

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, inf_flow):
        """
        Pass input x through forward once
            :param inf_flow: inter frame motion field
        """
        shape = inf_flow.shape
        seq_len = shape[0]
        lag_flow = torch.zeros(shape, device=inf_flow.device)
        lag_flow[0, ::] = inf_flow[0,::]
        for k in range (1, seq_len):
            src = inf_flow[k, ::]
            sum_flow = lag_flow[k-1:k, ::]
            src_x = src[0, ::]
            src_x = src_x.unsqueeze(0)
            src_x = src_x.unsqueeze(0)
            src_y = src[1, ::]
            src_y = src_y.unsqueeze(0)
            src_y = src_y.unsqueeze(0)
            lag_flow_x = self.spatial_transform(src_x, sum_flow)
            lag_flow_y = self.spatial_transform(src_y, sum_flow)
            lag_flow[k, ::] = sum_flow + torch.cat((lag_flow_x, lag_flow_y), dim=1)

        return lag_flow


class Lagrangian_motion_estimate_net(nn.Module):
    """
    [lagrangian_motion_estimate_net] is a class representing the architecture for Lagrangian motion estimation on a time
     sequence which is based on probabilistic diffeomoprhic VoxelMorph with a full sequence of lagrangian motion
     constraints. You may need to modify this code (e.g., number of layers) to suit your project needs.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, int_steps=7):
        """
        Instiatiate lagrangian_motion_estimate_net model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
            :param int_steps: the number of integration steps
        """
        super(Lagrangian_motion_estimate_net, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow_mean = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        self.flow_log_sigma = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow_mean.weight = nn.Parameter(nd.sample(self.flow_mean.weight.shape))
        self.flow_mean.bias = nn.Parameter(torch.zeros(self.flow_mean.bias.shape))
        self.flow_log_sigma.weight = nn.Parameter(nd.sample(self.flow_log_sigma.weight.shape))
        self.flow_log_sigma.bias = nn.Parameter(torch.zeros(self.flow_log_sigma.bias.shape))

        self.diffeomorph_transform = DiffeomorphicTransform(size=vol_size, mode='bilinear',time_step=int_steps)
        self.spatial_transform = SpatialTransformer(vol_size)
        self.lag_flow = Lagrangian_flow(vol_size)
        self.lag_regular = True

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)

        flow_mean = self.flow_mean(x)
        flow_log_sigma = self.flow_log_sigma(x)

        # reparamterize
        std = torch.exp(0.5*flow_log_sigma)
        z = flow_mean + std * torch.rand_like(std)

        # bi-directional INF flows
        inf_flow = self.diffeomorph_transform(z)
        neg_inf_flow = self.diffeomorph_transform(-z)

        # image warping
        y_src = self.spatial_transform(src, inf_flow)
        y_tgt = self.spatial_transform(tgt, neg_inf_flow)

        flow_param = torch.cat((flow_mean, flow_log_sigma), dim=1)

        if self.lag_regular:
            # Lagrangian flow
            lag_flow = self.lag_flow(inf_flow)
            # Warp the reference frame by the Lagrangian flow
            src_0 = src[0, ::]
            shape = src.shape  # seq_length (batch_size), channel, height, width
            seq_length = shape[0]
            src_re = src_0.repeat(seq_length, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
            src_re = src_re.contiguous()
            lag_y_src = self.spatial_transform(src_re, lag_flow)
            return y_src, y_tgt, lag_y_src, flow_param, inf_flow, neg_inf_flow, lag_flow
        else:
            return  y_src, y_tgt, flow_param, inf_flow, neg_inf_flow

