import numpy as np
from torchvision.transforms import functional as transform_F

class TwoCrops(object):
    def __call__(self, img):
        h, w = img.size

        return transform_F.crop(img, 0, 0, h, h), transform_F.crop(img, w - h, 0, h, h)

def hard_merge_masks(img1, img2, full_h=320):
    b,h,w = img1.shape

    new = np.zeros((b, full_h, w), np.float32)
    new[:, :full_h // 2 + 1, :] = img1[:, :full_h // 2 + 1, :]
    new[:, full_h // 2 + 1:, :] = img2[:, -(full_h // 2 - 1):, :]

    return new

def merge_masks(img1, img2, full_h=320):
    b,h,w = img1.shape
    a = full_h - h

    new = np.zeros((b, full_h, w), np.float32)
    new[:, :h, :] += img1[:, :, :]
    new[:, a:, :] += img2[:, :, :]
    new[:, a:h, :] /= 2

    return new

def merge_mirrored(img, mirror):
    return 0.5*(img[:,:,:] + mirror[:,:,::-1])
