import os
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, images_dir, images_name, target_dir=None,
                 pair_transforms=None, tensor_transforms=None, color_transforms=None):
        
        self.images_dir = images_dir
        self.target_dir = target_dir
        self.images_name = images_name
        self.tensor_transforms = tensor_transforms
        self.pair_transforms = pair_transforms
        self.color_transforms = color_transforms
                           
        print('{} images'.format(len(self.images_name)))

    def __len__(self):
        return len(self.images_name)
               
    def __getitem__(self, idx):
        img_filename = os.path.join(self.images_dir, self.images_name[idx])
        img = Image.open(img_filename)
        
        if self.target_dir:
            mask_name = self.images_name[idx].split('.')[0] + '.png'    #.jpg --> .png
            mask_filename = os.path.join(self.target_dir, mask_name)
            mask = Image.open(mask_filename)
        else:
            mask = []
            
        #Augmentation
        # color
        if self.color_transforms is not None:
            img = self.color_transforms(img)
            
        # crop, resize, rotate
        if self.pair_transforms is not None:
            if mask:
                img, mask = self.pair_transforms(img, target=mask)
            else:
                img = self.pair_transforms(img)
        
        # normalize
        if self.tensor_transforms is not None:
            img = self.tensor_transforms(img)
        
        return {'img': img, 'mask': mask}