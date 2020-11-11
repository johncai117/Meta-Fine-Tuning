# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path=ChestX_path+"/Data_Entry_2017.csv", \
        image_path = ChestX_path+"/images/"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)        

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Transform image to tensor
        #img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (self.img_path +  single_image_name, single_image_label)

    def __len__(self):
        return self.data_len



identity = lambda x:x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


        d = CustomDatasetFromImages()

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class SimpleDataset2:
    def __init__(self, transform, d, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


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
        self.cl_list = range(7)


        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = CustomDatasetFromImages()

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        #for key, item in self.sub_meta.items():
            #print (len(self.sub_meta[key]))
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
       
        for cl in self.cl_list:
            #print (cl)
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SetDataset2:
    def __init__(self, batch_size, dat, transloader, num_aug=4):

        self.sub_meta = {}
        self.cl_list = range(7)
        self.num_aug = num_aug

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for i, (data, label) in enumerate(dat):
            self.sub_meta[label].append(data)

        #for key, item in self.sub_meta.items():
            #print (len(self.sub_meta[key]))

        seed = 11

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
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        
        for cl in self.cl_list:
            sub_dataset = SubDataset2(self.sub_meta[cl], cl, transform = transloader, num_aug = self.num_aug)       
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, dat, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
  

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class SubDataset2:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), num_aug = 4, target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.num_aug = num_aug
        ## load two none-transformed versions for consistency
        self.transform_list = [transform.get_composed_transform_noaug] + [transform.get_composed_transform_noaug]
        for i in range(self.num_aug):
          transform_ = transform.get_composed_transform_aug
          self.transform_list.append(transform_)
        self.target_transform = target_transform

    def __getitem__(self,i):
        path = self.sub_meta[i]
        img_as_img = Image.open(path).resize((256, 256)).convert('RGB')
        img_as_img.load()
        img_aug_list = []
        for j in range(self.num_aug + 2): ## need the plus 2
          img_transform_func = self.transform_list[j]
          img_func = img_transform_func()
          img_aug_list.append(img_func(img_as_img))
        target = self.target_transform(self.cl)
        target_list = [target] * (self.num_aug + 2)
        out = list(zip(img_aug_list, target_list))
        #print(hello)
        return out

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
            yield self.perms[i]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.1, Contrast=0.1, Color=0.001)):
        
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)

        if transform_type=='RandomSizedCrop':
            return method(self.image_size, scale=(0.6, 0.95)) 

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
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_composed_transform_aug(self):
        transform_list = ['RandomSizedCrop', 'ImageJitter', 'ToTensor', 'Normalize']
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_composed_transform_noaug(self):
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
        self.dat = CustomDatasetFromImages()

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, num_aug = 4): #parameters that would change on train/val set
        dataset = SetDataset2(self.batch_size, self.dat, self.trans_loader, num_aug)

        sampler = EpisodicBatchSampler2(len(dataset), self.n_way, self.n_eposide )  
        perms = sampler.generate_perm()

        data_loader_params = dict(batch_sampler = perms, num_workers = 0, pin_memory = True)       
    
        
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
   
       
        return data_loader


if __name__ == '__main__':

    base_datamgr            = SetDataManager(224, n_query = 16, n_support = 5)
    base_loader             = base_datamgr.get_data_loader(aug = True)

