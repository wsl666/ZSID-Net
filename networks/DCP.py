import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Guided image filtering for grayscale images
class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):  # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """

        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter(torch.ones(p.size()))

        if I.is_cuda:
            N = N.cuda()

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b


class DCPDehazeGenerator(nn.Module):
    """Create a DCP Dehaze generator"""
    def __init__(self, win_size=5, r=15, eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()

        self.guided_filter = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega = 0.95

    def get_dark_channel(self, img, neighborhood_size):
        shape = img.shape
        if len(shape) == 4:
            img_min,_ = torch.min(img, dim=1)

            padSize = np.int(np.floor(neighborhood_size/2))
            if neighborhood_size % 2 == 0:
                pads = [padSize, padSize-1 ,padSize ,padSize-1]
            else:
                pads = [padSize, padSize ,padSize ,padSize]

            img_min = F.pad(img_min, pads, mode='constant', value=1)
            dark_img = -F.max_pool2d(-img_min, kernel_size=neighborhood_size, stride=1)
        else:
            raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

        dark_img = torch.unsqueeze(dark_img, dim=1)
        return dark_img

    def atmospheric_light(self, img, dark_img):
        num,chl,height,width = img.shape
        topNum = np.int(0.01*height*width)

        A = torch.Tensor(num,chl,1,1)
        if img.is_cuda:
            A = A.cuda()

        for num_id in range(num):
            curImg = img[num_id,...]
            curDarkImg = dark_img[num_id,0,...]

            _, indices = curDarkImg.reshape([height*width]).sort(descending=True)
            #curMask = indices < topNum

            for chl_id in range(chl):
                imgSlice = curImg[chl_id,...].reshape([height*width])
                A[num_id,chl_id,0,0] = torch.mean(imgSlice[indices[0:topNum]])

        return A

    def forward(self, x):
        if x.shape[1] > 1:
            # rgb2gray
            guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
        else:
            guidance = x
        # rescale to [0,1]

        guidance = (guidance + 1)/2
        guidance = torch.unsqueeze(guidance, dim=1)
        imgPatch = (x + 1)/2

        num,chl,height,width = imgPatch.shape

        # dark_img and A with the range of [0,1]
        dark_img = self.get_dark_channel(imgPatch, self.neighborhood_size)
        A = self.atmospheric_light(imgPatch, dark_img)

        map_A = A.repeat(1,1,height,width)
        # make sure channel of trans_raw == 1
        trans_raw = 1 - self.omega*self.get_dark_channel(imgPatch/map_A, self.neighborhood_size)

        # get initial results
        T_DCP = self.guided_filter(guidance, trans_raw)
        J_DCP = (imgPatch - map_A)/T_DCP.repeat(1,3,1,1) + map_A
        J_DCP = torch.clamp(J_DCP,0,1)

        A_DCP = map_A

        return J_DCP, T_DCP, A_DCP