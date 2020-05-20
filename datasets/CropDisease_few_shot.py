# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *

identity = lambda x:x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


        d = ImageFolder(CropDisease_path + "/dataset/train/")

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(38)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(CropDisease_path + "/dataset/train/")


        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
    

        for key, item in self.sub_meta.items():
            print (len(self.sub_meta[key]))

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SetDataset2:
    def __init__(self, batch_size, sub_meta, transform):

        for key, item in sub_meta.items():
            print (len(sub_meta[key]))
        self.cl_list = range(38)
        seed = 10
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        import random
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = False,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        
        for cl in self.cl_list:
            random.shuffle(sub_meta[cl]) ## add back the seeded randomness ## same across datasets
            sub_dataset = SubDataset2(sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.d = d

    def __getitem__(self,i):
        img = self.d[self.sub_meta[i]]
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class SubDataset2:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.sub_meta = sub_meta

    def __getitem__(self,i):
        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class EpisodicBatchSampler2(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes
    
    def generate_perm(self):
        self.perms = []
        for i in range(self.n_episodes):
          self.perms.append(torch.randperm(self.n_classes)[:self.n_way])

        return self.perms

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.perms[i] ##this is fixed in later iterations

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.1, Contrast=0.1, Color=0.05)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size, scale=(0.6, 0.9))
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = False, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class SetDataManager2(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager2, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.cl_list = range(38)
        self.sub_meta = {}
        d = ImageFolder(CropDisease_path + "/dataset/train/")
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
        

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, num_aug = 4): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(False)
        
        
        

        dataset = SetDataset2(self.batch_size, self.sub_meta, transform)
        dataset2 = SetDataset2(self.batch_size, self.sub_meta, transform)
        
        sampler = EpisodicBatchSampler2(len(dataset), self.n_way, self.n_eposide )  
        perms = sampler.generate_perm() ##permanent samples

        data_loader_params = dict(batch_sampler = sampler, shuffle = False, num_workers = 1, pin_memory = True)       
        
        dataset_list = [dataset] + [dataset2]## for checking randomness later
        for i in range(num_aug):
          transform2 = TransformLoader(self.image_size).get_composed_transform(True)
          dataset2 = SetDataset2(self.batch_size, self.sub_meta, transform2)
          dataset_list.append(dataset2)
        dataset_chain = ConcatDataset(dataset_list)
        
        data_loader = torch.utils.data.DataLoader(dataset_chain, **data_loader_params)
       
        return data_loader

if __name__ == '__main__':

    train_few_shot_params   = dict(n_way = 5, n_support = 5) 
    base_datamgr            = SetDataManager(224, n_query = 16)
    base_loader             = base_datamgr.get_data_loader(aug = True)

    cnt = 1
    for i, (x, label) in enumerate(base_loader):
        if i < cnt:
            print(label)
        else:
            break

