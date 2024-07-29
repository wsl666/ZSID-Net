# Perceptual fusion
import cv2
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torchvision import transforms

def fuse_images(real_I, rec_J, refine_J):
    """
    real_I, rec_J, and refine_J: Images with shape hxwx3
    """
    # realness features
    mat_RGB2YMN = np.array([[0.299,0.587,0.114],
                            [0.30,0.04,-0.35],
                            [0.34,-0.6,0.17]])

    recH,recW,recChl = rec_J.shape
    rec_J_flat = rec_J.reshape([recH*recW,recChl])
    rec_J_flat_YMN = (mat_RGB2YMN.dot(rec_J_flat.T)).T
    rec_J_YMN = rec_J_flat_YMN.reshape(rec_J.shape)

    refine_J_flat = refine_J.reshape([recH*recW,recChl])
    refine_J_flat_YMN = (mat_RGB2YMN.dot(refine_J_flat.T)).T
    refine_J_YMN = refine_J_flat_YMN.reshape(refine_J.shape)

    real_I_flat = real_I.reshape([recH*recW,recChl])
    real_I_flat_YMN = (mat_RGB2YMN.dot(real_I_flat.T)).T
    real_I_YMN = real_I_flat_YMN.reshape(real_I.shape)

    # gradient features
    rec_Gx = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    rec_Gy = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    rec_GM = np.sqrt(rec_Gx**2 + rec_Gy**2)

    refine_Gx = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    refine_Gy = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    refine_GM = np.sqrt(refine_Gx**2 + refine_Gy**2)

    real_Gx = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    real_Gy = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    real_GM = np.sqrt(real_Gx**2 + real_Gy**2)

    # similarity Calculation
    rec_S_V = (2*real_GM*rec_GM+160)/(real_GM**2+rec_GM**2+160) # GM
    rec_S_M = (2*rec_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(rec_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    rec_S_N = (2*rec_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(rec_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    rec_S_R = (rec_S_M*rec_S_N).reshape([recH,recW]) # ChromMN

    refine_S_V = (2*real_GM*refine_GM+160)/(real_GM**2+refine_GM**2+160) # GM
    refine_S_M = (2*refine_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(refine_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    refine_S_N = (2*refine_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(refine_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    refine_S_R = (refine_S_M*refine_S_N).reshape([recH,recW]) # ChromMN

    rec_S = rec_S_R*np.power(rec_S_V, 0.4)
    refine_S = refine_S_R*np.power(refine_S_V, 0.4)


    fuseWeight = np.exp(rec_S)/(np.exp(rec_S)+np.exp(refine_S))
    fuseWeightMap = fuseWeight.reshape([recH,recW,1]).repeat(3,axis=2)

    fuse_J = rec_J*fuseWeightMap + refine_J*(1-fuseWeightMap)

    return fuse_J


def tensor2image(tensor):
    image_tensor=tensor.clone()
    # image = 127.5*(image_tensor[0].detach().cpu().float().numpy() + 1.0)
    image = 255.0*(image_tensor[0].detach().cpu().float().numpy())
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))

    image = np.transpose(image, (1, 2, 0))

    return image.astype(np.uint8)


def Perceptual_Fusion(real_I, rec_J, refine_J):

    real_I = tensor2image(real_I)  # [0, 255], np.float
    rec_J = tensor2image(rec_J) / 255.  # [0, 1]
    refine_J = tensor2image(refine_J) / 255.  # [0, 1]

    result_J = fuse_images(real_I, rec_J * 255., refine_J * 255.) / 255.  # [0, 1], np.float

    result_J = np.transpose(result_J, (2, 0, 1))

    result_J = torch.unsqueeze(torch.from_numpy(result_J), dim=0).float()  # [0, 1], tensor

    return result_J



if __name__ == "__main__":
    # 读取图像
    image_path1 = "GT_452.png"
    image_path2 = "GT_452_j.png"
    image_path3 = "GT_452_asm.png"

    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image3 = Image.open(image_path3)

    # 使用transforms.ToTensor()将图像转换为张量，并进行归一化
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor1 = transform(image1)
    image_tensor2 = transform(image2)
    image_tensor3 = transform(image3)

    image_tensor1 = torch.unsqueeze(image_tensor1, dim=0)
    image_tensor2 = torch.unsqueeze(image_tensor2, dim=0)
    image_tensor3 = torch.unsqueeze(image_tensor3, dim=0)

    fusion = Perceptual_Fusion(image_tensor1,image_tensor2,image_tensor3)
    # print(fusion)
    torchvision.utils.save_image(fusion,"tensor_f.png")
    # fusion = (fusion*255.).astype(np.uint8)
    #
    # save_image(fusion, "f.png")





