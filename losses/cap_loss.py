import torch
from skimage.color import rgb2hsv
from torch import nn

loss_mse = nn.MSELoss()

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def caploss(J):

    hsv = np_to_torch(rgb2hsv(torch_to_np(J).transpose(1, 2, 0)))
    cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
    cap_loss = loss_mse(cap_prior, torch.zeros_like(cap_prior))

    return cap_loss



if __name__ =="__main__":

    x=torch.randn(1,3,256,256).cuda()
    bsloss = caploss(x)
    print(bsloss)



