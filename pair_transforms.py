from PIL import Image
import random

from torchvision.transforms import functional as transform_F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target = None):
        if target is not None:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target

        for t in self.transforms:
            img = t(img)
        return img

class RandomResizedCrop(object):
    def __init__(self, size, scale = (0.5, 1.0), ratio = (0.75, 1.33)):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, target = None):
        w, h = img.size
        min_dim = min(w,h)
        
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        cw, ch = int(min_dim*scale/ratio), int(min_dim*scale*ratio)
        
        if cw > w:
            cw = w
        if ch > h:
            ch = h
            
        crop_size = (cw, ch)
        
        if target is not None:
            crop_img, crop_target = RandomCrop(crop_size)(img, target)
            return Resize(self.size)(crop_img, crop_target)
        else:
            return Resize(self.size)(RandomCrop(crop_size)(img))
    
class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        if h - th > 0:
            i = random.randint(0, h - th)
        else:
            i = 0
            
        if w - tw > 0:
            j = random.randint(0, w - tw)
        else:
            j = 0
        return i, j, th, tw

    def __call__(self, img, target = None):
        i, j, h, w = self.get_params(img, self.size)

        if target is not None:
            return transform_F.crop(img, i, j, h, w), transform_F.crop(target, i, j, h, w)
        else:
            return transform_F.crop(img, i, j, h, w)

        
class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR, interpolation_tg = Image.NEAREST):
        self.size = size
        self.interpolation = interpolation
        self.interpolation_tg = interpolation_tg

    def __call__(self, img, target = None):
        if target is not None:
            return transform_F.resize(img, self.size, self.interpolation), transform_F.resize(target, self.size, self.interpolation_tg)
        return transform_F.resize(img, self.size, self.interpolation)
        

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            if target is not None:
                return transform_F.hflip(img), transform_F.hflip(target)
            else:
                return transform_F.hflip(img)

        if target is not None:
            return img, target
        return img

        
class ToTensor(object):
    def __call__(self, pic, pic2=None):
        if pic2 is not None:
            return transform_F.to_tensor(pic), transform_F.to_tensor(pic2)
        return transform_F.to_tensor(pic)
