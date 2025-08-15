import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class Kvasir_SEG(Dataset):
    """ Kvasir_SEG Dataset """

    def __init__(self,config,transform,keyword):
        self.data_dir = config.root_path
        self.transform = transform
        self.sample_list = []
        self.image_list = os.listdir(self.data_dir+ '/images')
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if keyword == 'train':
            self.image_list = self.image_list[:700]
        if keyword == 'val':
            self.image_list = self.image_list[700:900]
        if keyword == 'test':
            self.image_list = self.image_list[900:]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image = cv2.imread(self.data_dir + '/images/' + image_name,-1)
        label = cv2.imread(self.data_dir + '/masks/' + image_name,-1)

        if image.shape[0] < 512:
            num = 512 - image.shape[0]
            image = np.pad(image, [(num//2, num - num//2),(0,0),(0,0)], mode='constant', constant_values=0)
            label = np.pad(image, [(num//2, num - num//2),(0,0),(0,0)], mode='constant', constant_values=0)
        if image.shape[1] < 512:
            num = 512 - image.shape[1]
            image = np.pad(image, [(0,0),(num//2, num - num//2),(0,0)], mode='constant', constant_values=0)
            label = np.pad(image, [(0,0),(num//2, num - num//2),(0,0)], mode='constant', constant_values=0)

        end1 = np.random.randint(image.shape[0] - 511)
        end2 = np.random.randint(image.shape[1] - 511)

        image = image[end1:end1+512,end2:end2+512,:]
        label = label[end1:end1+512,end2:end2+512,0]

        if self.transform:
            sample = self.transform(image=image, label=label)

        image = sample['image'].transpose(2,0,1)
        label = sample['label'][np.newaxis,...]
        label = np.where(label > 127, 1, 0)
        sample = {'label': label, 'image': image, 'idx':idx}

        return sample
