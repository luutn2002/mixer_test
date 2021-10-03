from torch import max, rot90
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes, ImageFolder
from torchvision.io import read_image
from glob import glob
from torchvision.transforms import Resize, Compose, AutoAugment
from torchvision.transforms import AutoAugmentPolicy
from torchvision.transforms.functional import hflip, vflip
from numpy.random import uniform, seed, shuffle
from numpy import pi, arange, floor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import PILtoTensor
import os

class MedicalDatasetLoader(Dataset):
    def __init__(self, config):
        
        self.img_dir = config.img_dir
        self.mask_dir = config.mask_dir
        self.image_only_transform = config.image_only_transform
        self.mask_only_transform = config.mask_only_transform
        self.enable_parallel_transform = config.enable_parallel_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def parallel_transform(self, image, mask):
        
        mask = ((mask > 127.5)*1).byte()
        
        #resize
        trans = Resize((224, 224))
        image = trans(image)
        mask = trans(mask)
        
        '''
        # Random horizontal flipping
        if uniform() > 0.5:
            image = hflip(image)
            mask = hflip(mask)

        # Random vertical flipping
        if uniform() > 0.5:
            image = vflip(image)
            mask = vflip(mask)
        '''
        
        return image, mask

    def __getitem__(self, idx):
        
        img_list = glob(self.img_dir + '*.jpg')
        mask_list = glob(self.mask_dir + '*.jpg')
        img_path = img_list[idx]
        mask_path = mask_list[idx]
        image = read_image(img_path)
        mask = read_image(mask_path)
        
        if self.enable_parallel_transform:
            image, mask = self.parallel_transform(image, mask)
        if self.image_only_transform:
            image = self.image_only_transform(image)
        if self.mask_only_transform:
            mask = self.mask_only_transform(mask)

        if max(image).item() > 1:
            image = image/255
        
        return image, mask
    
def Medical_Train_Test_Loader(dataset, config):
    
    dataset_size = dataset.__len__()
    
    indices = arange(dataset_size)
    split = int(floor(config.validation_split * dataset_size))
    
    if config.shuffle_dataset:
        seed(config.random_seed)
        shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, 
                              batch_size=config.batch_size,
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, 
                                   batch_size=config.batch_size,
                                   sampler=valid_sampler)
    
    return train_loader, validation_loader

def get_cityscapes_loader(config, load_val=False):
    combination_trans = Compose([
                                ToTensor(),
                                Resize((224, 224))
                                ])

    mask_trans = Compose([
                        PILtoTensor(),
                        Resize((224, 224))
                        ])
    
    if load_val:
        val_data = Cityscapes(config.data_dir,
                                        split='val', 
                                        mode=config.data_mode,
                                        target_type=config.data_target_type,
                                        transform = combination_trans,
                                        target_transform=mask_trans)
        
        val_loader = DataLoader(val_data, 
                            batch_size=config.batch_size)
        
        return val_loader
    
    else:    
        train_data = Cityscapes(config.data_dir,
                                    split='train', 
                                    mode=config.data_mode,
                                    target_type=config.data_target_type,
                                    transform=combination_trans,
                                    target_transform=mask_trans)

        test_data = Cityscapes(config.data_dir,
                                    split='test', 
                                    mode=config.data_mode,
                                    target_type=config.data_target_type,
                                    transform = combination_trans,
                                    target_transform=mask_trans)

    
        train_loader = DataLoader(train_data, 
                              batch_size=config.batch_size,
                              shuffle=True, pin_memory=True)
    
        test_loader = DataLoader(test_data, 
                            batch_size=config.batch_size,
                            pin_memory=True)
        
        return train_loader, test_loader
    
def get_imagenet_loader(config):
    
    train_transform = Compose([
        #AutoAugment(AutoAugmentPolicy.IMAGENET),
        ToTensor(),
        Resize((224, 224))
    ])
    
    test_transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    
    train_data = ImageFolder(config.train_data_dir, transform=train_transform)
    train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    
    val_data = ImageFolder(config.val_data_dir, transform=test_transform)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, pin_memory=True)

    return train_data_loader, val_data_loader