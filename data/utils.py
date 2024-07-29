from torch import nn
from torch.nn import init
import random
import time
import datetime
import sys
from PIL import Image
import torch
from visdom import Visdom
import numpy as np
import pandas as pd
import cv2

def synthesize_fog(J, t, A=None):
    """
    Synthesize hazy image base on optical model
    I = J * t + A * (1 - t)
    """

    if A is None:
        A = 1

    return J * t + A * (1 - t)

def reverse_fog_asm(I, t, A=1, t0=0.01):
    """
    Recover haze-free image using hazy image and depth
    J = (I - A) / max(t, t0) + A
    """
    t_clamp = torch.clamp(t, t0, 1)
    J = (I-A) / t_clamp + A

    return torch.clamp(J, 0, 1)

def reverse_fog_easm(I, t, A=1, t0=0.01):
    """
    Recover haze-free image using hazy image and depth
    J = (I - A) / max(t, t0) + A
    ef = enhancement factor
    """
    t_clamp = torch.clamp(t, t0, 1)
    ef = torch.log(t_clamp) / torch.log(torch.min(t_clamp))
    J = (I-A) / (t_clamp * ef) + A / ef

    return torch.clamp(J, 0, 1)


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


def get_dark_channel(I, w):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc

def mask_sky(hazy_image, dehaze_image, clear_image):

    dc = get_dark_channel(hazy_image, w=15)
    dc_shaped = dc.repeat(1, 3, 1, 1)

    onemask = torch.ones_like(dc_shaped)
    zeromask = torch.zeros_like(dc_shaped)

    nosky = torch.where(dc_shaped > 0.60, zeromask, onemask)
    onlysky = torch.where(dc_shaped > 0.60, onemask, zeromask)

    nosky_hazy = nosky * hazy_image
    nosky_dehaze = nosky * dehaze_image
    nosky_clear = nosky * clear_image

    onlysky_hazy = onlysky * hazy_image
    onlysky_dehaze = onlysky * dehaze_image

    return nosky_hazy, nosky_dehaze, nosky_clear, onlysky_hazy, onlysky_dehaze

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

#反归一化，保存图片用
def tensor_image(tensor):

    image = (tensor * 0.5) + 0.5

    return image

def tensor2image(tensor):
    image_tensor=tensor.clone()
    # image = 127.5*(image_tensor[0].detach().cpu().float().numpy() + 1.0)
    image = 255.0*(image_tensor[0].detach().cpu().float().numpy())
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log_train(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        # batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        # batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        # sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


    def log_val(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        if losses == None:
            pass
        else:
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name]
                else:
                    self.losses[loss_name] += losses[loss_name]

                if (i+1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        # batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        # batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        # sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

    def log_zs(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        # sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        if losses == None:
            pass
        else:
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name]
                else:
                    self.losses[loss_name] += losses[loss_name]

                # if (i+1) == len(losses.keys()):
                #     sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
                # else:
                #     sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        # batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        # batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        # sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


#把生成器生成的数据放进队列，以队列的形式作为对判别器的输入
class ReplayBuffer():
    def __init__(self,max_size=50):
        assert (max_size > 0)
        self.max_size = max_size
        self.data = []

    def push_and_pop(self,data):
        to_return = []
        for element in  data.data:
            element = torch.unsqueeze(element,0)
            if len(self.data)<self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1)>0.5:
                    i = random.randint(0,self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return torch.cat(to_return)
        # (n*batch_size, feature_size) 沿着第一维度拼接

#学习率衰减的函数
class LambdaLR():
    def __init__(self,n_epochs,offset,decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0)
        self.n_epochs=n_epochs
        self.offset=offset
        self.decay_start_epoch=decay_start_epoch

    def step(self,epoch):
        return 1.0-max(0,epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



#初始化模型参数
def weights_init(m,init_type='normal', init_gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)

#Set requies_grad=Fasle for all the networks to avoid unnecessary computations
def set_requires_grad(nets, requires_grad=False):
        """
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    #计算梯度惩罚损失
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- haze images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix haze and fake data or not [haze | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'haze':   # either use haze images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients1 = torch.autograd.grad(outputs=disc_interpolates[0], inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates[0].size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients2 = torch.autograd.grad(outputs=disc_interpolates[1], inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates[1].size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients1 = gradients1[0].view(real_data.size(0), -1)  # flat the data
        gradients2 = gradients2[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty1 = (((gradients1 + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        gradient_penalty2 = (((gradients2 + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps

        return gradient_penalty1 + gradient_penalty2

    else:

        return 0.0, None


def print_log(epoch, num_epochs, one_epoch_time, val_psnr, val_ssim):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, val_psnr, val_ssim))

    # --- Write the training logs --- #
    with open('logs/train_log.txt', 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}],Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, val_psnr, val_ssim), file=f)


