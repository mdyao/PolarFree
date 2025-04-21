
import torch.nn as nn
from torchvision import models

#########################################################################################################################################

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class PhaseLoss(nn.Module):
    def __init__(self, loss_weight = 1, epsilon=1e-8):
        """
        初始化 PhaseLoss 类。
        
        :param epsilon: 防止除零或其他数值不稳定的小常数
        """
        super(PhaseLoss, self).__init__()
        self.epsilon = epsilon
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        # 转换为灰度图，确保张量在 0-1 范围内
        gray_img1 = torch.mean(img1, dim=1, keepdim=True) 
        gray_img2 = torch.mean(img2, dim=1, keepdim=True) 
        
        # 计算 FFT，加上 epsilon 提高数值稳定性
        fft_img1 = torch.fft.fft2(gray_img1 + self.epsilon)
        fft_img2 = torch.fft.fft2(gray_img2 + self.epsilon)
        
        # 获取相位信息并归一化到 [-π, π]
        phase_img1 = torch.angle(fft_img1)
        phase_img2 = torch.angle(fft_img2)

        # 计算相位差的 MSE 损失
        loss = F.mse_loss(phase_img1, phase_img2)

        return self.loss_weight * loss


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1) * ssim_map.mean()
    else:
        return (-1) * ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


###########################################################################################################################

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        self.vgg = models.vgg19(pretrained=False)  # .to(device)  # .cuda()
        self.vgg.load_state_dict(torch.load('/ailab/user/yaomingde/workspace/ideas/polarization_reflect/data/v2/pretrained_model/vgg19-dcbb9e9d.pth'))
        self.vgg.eval()
        self.vgg_pretrained_features = self.vgg.features

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        # indices = sorted(indices)
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)
        return out

# class Vgg19(nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg = models.vgg19(pretrained=False)
#         vgg.load_state_dict(torch.load('//gdata2/zhuyr/VGG/vgg19-dcbb9e9d.pth'))
#         vgg.eval()
#         vgg_pretrained_features = vgg.features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(3):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(3, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGLoss(nn.Module):
    def __init__(self, loss_weight: float = 1) -> None:
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

        self.loss_weight = loss_weight
    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # loss = self.criterion(x_vgg, y_vgg.detach())
        for i in range(len(x_vgg)):
            loss += self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.loss_weight * loss


############################################################################################################################3


class GradientLoss(nn.Module):
    """Gradient Histogram Loss"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.bin_num = 64
        self.delta = 0.2
        self.clip_radius = 0.2
        assert(self.clip_radius>0 and self.clip_radius<=1)
        self.bin_width = 2*self.clip_radius/self.bin_num
        if self.bin_width*255<1:
            raise RuntimeError("bin width is too small")
        self.bin_mean = np.arange(-self.clip_radius+self.bin_width*0.5, self.clip_radius, self.bin_width)
        self.gradient_hist_loss_function = 'L2'
        # default is KL loss
        if self.gradient_hist_loss_function == 'L2':
            self.criterion = nn.MSELoss()
        elif self.gradient_hist_loss_function == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.KLDivLoss()

    def get_response(self, gradient, mean):
        # tmp = torch.mul(torch.pow(torch.add(gradient, -mean), 2), self.delta_square_inverse)
        s = (-1) / (self.delta ** 2)
        tmp = ((gradient - mean) ** 2) * s
        return torch.mean(torch.exp(tmp))

    def get_gradient(self, src):
        right_src = src[:, :, 1:, 0:-1]     # shift src image right by one pixel
        down_src = src[:, :, 0:-1, 1:]      # shift src image down by one pixel
        clip_src = src[:, :, 0:-1, 0:-1]    # make src same size as shift version
        d_x = right_src - clip_src
        d_y = down_src - clip_src

        return d_x, d_y

    def get_gradient_hist(self, gradient_x, gradient_y):
        lx = None
        ly = None
        for ind_bin in range(self.bin_num):
            fx = self.get_response(gradient_x, self.bin_mean[ind_bin])
            fy = self.get_response(gradient_y, self.bin_mean[ind_bin])
            fx = torch.cuda.FloatTensor([fx])
            fy = torch.cuda.FloatTensor([fy])

            if lx is None:
                lx = fx
                ly = fy
            else:
                lx = torch.cat((lx, fx), 0)
                ly = torch.cat((ly, fy), 0)
        # lx = torch.div(lx, torch.sum(lx))
        # ly = torch.div(ly, torch.sum(ly))
        return lx, ly

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.get_gradient(output)
        target_gradient_x, target_gradient_y = self.get_gradient(target)

        output_gradient_x_hist, output_gradient_y_hist = self.get_gradient_hist(output_gradient_x, output_gradient_y)
        target_gradient_x_hist, target_gradient_y_hist = self.get_gradient_hist(target_gradient_x, target_gradient_y)
        # loss = self.criterion(output_gradient_x_hist, target_gradient_x_hist) + self.criterion(output_gradient_y_hist, target_gradient_y_hist)
        loss = self.criterion(output_gradient_x,target_gradient_x)+self.criterion(output_gradient_y,target_gradient_y)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        #diff = x.to('cuda:0') - y.to('cuda:0')
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class CharbonnierLoss1(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-4):
        super(CharbonnierLoss1, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        #diff = x.to('cuda:0') - y.to('cuda:0')
        diff = x - y
        loss = torch.mean((diff * diff) + (self.eps*self.eps))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.kernel = self.kernel #.to(device)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        #loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))

        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        #diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss


class fftLoss_old_version(nn.Module):
    def __init__(self):
        super(fftLoss_old_version, self).__init__()

    def forward(self, x, y):
        #diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        diff = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False) - torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)
        #torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss



class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()
    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)
        return loss

def sigmoid_mse_loss(input_logits, target_logits):
    """Takes sigmoid on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = F.sigmoid(input_logits)
    target_sigmoid = F.sigmoid(target_logits)

    # num_classes = input_logits.size()[1]
    return F.mse_loss(input_sigmoid, target_sigmoid, reduction='mean')


class TVLoss(nn.Module):
    def __init__(self, loss_weight: float = 1) -> None:
        """Total Variation Loss

        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.loss_weight * (tv_h + tv_w) / (batch_size * c * h * w)


#   train_out_fft = torch.rfft(train_output[2], signal_ndim=2, normalized=False, onesided=False)
#   train_labels_fft = torch.rfft(labels, signal_ndim=2, normalized=False, onesided=False)

#maoyy_stark_1.0

if __name__ == '__main__':
    loss =TVLoss()
    input1 = torch.rand(2,3,64,64).cuda()
    input2 = torch.rand(2,3,64,64).cuda()
    out = loss( input2)
    for i in range(10):
        print('qqqqqq', out)