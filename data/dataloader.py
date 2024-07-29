import glob
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tf
import os

random.seed(42)


def rotate(img,rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index==1:
        return img.rotate(90)
    if rotate_index==2:
        return img.rotate(180)
    if rotate_index==3:
        return img.rotate(270)
    if rotate_index==4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class TrainDataloader(Dataset):

    def __init__(self, haze_path, CLAHE_path, is_rotate=True, transform=None, model="train"):

        self.transform = tf.Compose(transform)
        self.model = model
        self.is_rotate = is_rotate

        self.haze_path = os.path.join(haze_path,"*.*")
        self.CLAHE_path = os.path.join(CLAHE_path,"*.*")

        self.list_haze = sorted(glob.glob(self.haze_path))
        self.list_CLAHE = sorted(glob.glob(self.CLAHE_path))

        print("Total {} examples:".format(model), max(len(self.list_haze), len(self.list_CLAHE)))


    def __getitem__(self, index):

        haze = self.list_haze[index % len(self.list_haze)]
        CLAHE = self.list_CLAHE[index % len(self.list_CLAHE)]

        name = os.path.basename(haze)

        haze = Image.open(haze).convert("RGB")
        CLAHE = Image.open(CLAHE).convert("RGB")

        if self.is_rotate:

            rotate_index = random.randrange(0, 8)

            haze = rotate(haze, rotate_index)
            CLAHE = rotate(CLAHE, rotate_index)

        haze = self.transform(haze)
        CLAHE = self.transform(CLAHE)


        return haze, CLAHE, name

    def __len__(self):

        return max(len(self.list_haze),len(self.list_CLAHE))



class TestDataloader(Dataset):

    def __init__(self, haze_path, clear_path, transform=None, model="test"):

        self.transform = tf.Compose(transform)
        self.model = model

        self.haze_path = os.path.join(haze_path,"*.*")
        self.clear_path = os.path.join(clear_path,"*.*")

        self.list_haze = sorted(glob.glob(self.haze_path))
        self.list_clear = sorted(glob.glob(self.clear_path))

        print("Total {} examples:".format(model), max(len(self.list_haze), len(self.list_clear)))


    def __getitem__(self, index):

        haze = self.list_haze[index % len(self.list_haze)]
        clear = self.list_clear[index % len(self.list_clear)]

        name = os.path.basename(haze)

        haze = Image.open(haze).convert("RGB")
        clear = Image.open(clear).convert("RGB")

        haze = self.transform(haze)
        clear = self.transform(clear)

        return haze, clear, name

    def __len__(self):

        return max(len(self.list_haze),len(self.list_clear))
