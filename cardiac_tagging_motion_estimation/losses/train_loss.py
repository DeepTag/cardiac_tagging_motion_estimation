import torch
import torch.nn.functional as F
import numpy as np
import math

"""
VM_diffeo_loss function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class VM_diffeo_loss(torch.nn.Module):
    """
    N-D main loss for VoxelMorph_diffeomorphism MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        super(VM_diffeo_loss, self).__init__()

        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape

    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner

        return filt

    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # prepare tf filter
        z = torch.ones([1] + sz)

        filt_tf = torch.from_numpy(self._adj_filt(ndims)).float()
        # filt_tf = filt_tf.cuda()
        strides = [1] * ndims

        win = filt_tf.shape
        pad_no = math.floor(win[0] / 2)
        padding = [pad_no] * ndims

        D = conv_fn(z.permute(0, 3, 1, 2), filt_tf.permute(3, 2, 0, 1), stride=strides, padding=padding)

        return D

    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        ndims = len(list(y_pred.size())) - 2

        y_pred1 = y_pred.permute(0, 2, 3, 1) #Batch, x, y, Channel

        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            # r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y_pred1.permute(d, *range(d), *range(d + 1, ndims + 2))
            df = y[1:, ...] - y[:-1, ...]
            sm += torch.mean(df * df)

        return 0.5 * sm / ndims

    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(list(y_pred.size())) - 2
        mean = y_pred[:, 0:ndims, ::]
        log_sigma = y_pred[:, ndims:, ::]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            shape = y_true.shape()
            self.flow_vol_shape = (shape[2], shape[3])

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D.cuda() * torch.exp(log_sigma) - log_sigma
        sigma_term = torch.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma ** 2) * torch.mean((y_true - y_pred) ** 2)

    def weighted_loss(self, warped_grid, fixed_img):
        """ weighted loss """
        s = warped_grid.shape
        one_matrix = torch.ones(s).cuda()
        reversed_grid = one_matrix - warped_grid
        # reversed_grid = reversed_grid.cuda()
        return torch.mean(reversed_grid*fixed_img)

    def gradient_loss(self, s, penalty='l2'):
        # s is the deformation_matrix of shape (seq_length, channels=2, height, width)
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-3):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 2
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)
        # conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        # win_size = np.prod(self.win)
        win_size = torch.from_numpy(np.array([np.prod(self.win)])).float()
        win_size = win_size.cuda()
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc0 = cross * cross / (I_var * J_var + self.eps)
        cc = torch.clamp(cc0, 0.001, 0.999)

        # return negative cc.
        return -1.0 * torch.mean(cc)
