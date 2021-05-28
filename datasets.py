import os
import numpy as np
import pandas
import monai
import PIL
import torch
import pandas as pd
from math import ceil
from monai.transforms import (AddChannel, Compose, RandRotate90, Resize, 
                              ScaleIntensity, ToTensor, LoadImage, RandRotate, 
                              RandFlip, RandZoom)

class Loader():
    """Loader for different image datasets with built in split function and download if needed.
    
    Functions:
        load_IXIT1: Loads the IXIT1 3D brain MRI dataset.
        load_MedNIST: Loads the MedNIST 2D image dataset.
    """
    
    ixi_train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), RandRotate90()])
    ixi_test_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96))])
    
    mednist_train_transforms = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(),
                                        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True), 
                                        RandFlip(spatial_axis=0, prob=0.5), 
                                        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)])
    mednist_test_transforms = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity()])
    
    
    @staticmethod
    def load_IXIT1(download: bool = False, train_transforms: object = ixi_train_transforms, 
                   test_transforms: object = ixi_test_transforms, test_size: float = 0.2, 
                   val_size: float = 0.0, sample_size: float = 0.01, shuffle: bool = True):
        """Loads the IXIT1 3D Brain MRI dataset.
        
        Consists of ~566 images of 3D Brain MRI scans and labels (0) for male and (1) for female.
        
        Args:
            download (bool): If true, then data is downloaded before loading it as dataset.
            train_transforms (Compose): Specify the transformations to be applied to the training dataset.
            test_transforms (Compose): Specify the transformations to be applied to the test dataset.
            sample_size (float): Percentage of available images to be used.
            test_size (float): Precantage of sample to be used as test data.
            val_size (float): Percentage of sample to be used as validation data.
            shuffle (bool): Whether or not the data should be shuffled after loading.
        """
        # Download data if needed
        if download:
            data_url = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar'
            compressed_file = os.sep.join(['Data', 'IXI-T1.tar'])
            data_dir = os.sep.join(['Data', 'IXI-T1'])

            # Data download
            monai.apps.download_and_extract(data_url, compressed_file, './Data/IXI-T1')

            # Labels document download
            labels_url = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls'
            monai.apps.download_url(labels_url, './Data/IXI.xls')
            
        # Get all the images and corresponding Labels
        images = [impath for impath in os.listdir('./Data/IXI-T1')]

        df = pd.read_excel('./Data/IXI.xls')

        data = []
        labels = []
        for i in images:
            ixi_id = int(i[3:6])
            row = df.loc[df['IXI_ID'] == ixi_id]
            if not row.empty:
                data.append(os.sep.join(['Data', 'IXI-T1', i]))
                labels.append(row.iat[0, 1] - 1) # Sex labels are 1/2 but need to be 0/1

        data, labels = data[:int(len(data) * sample_size)], labels[:int(len(data) * sample_size)]
        
        # Make train test validation split
        train_data, train_labels, test_data, test_labels, val_data, val_labels = _split(data, labels, 
                                                                                        test_size, val_size)
        
        # Construct and return Datasets
        train_ds = IXIT1Dataset(train_data, train_labels, train_transforms, shuffle)
        test_ds = IXIT1Dataset(test_data, test_labels, test_transforms, shuffle)
        
        if val_size == 0:
            return train_ds, test_ds
        else:
            val_ds = IXIT1Dataset(val_data, val_labels, test_transforms, shuffle)
            return train_ds, test_ds, val_ds
        
    
    @staticmethod
    def load_MedNIST(download: bool = False, train_transforms: object = mednist_train_transforms, 
                   test_transforms: object = mednist_test_transforms, test_size: float = 0.2, 
                   val_size: float = 0.0, sample_size: float = 0.01, shuffle: bool = True):
        """Loads the MedNIST 2D image dataset.
        
        Consists of ~60.000 2D images from 6 classes: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT.
        
        Args:
            download (bool): If true, then data is downloaded before loading it as dataset.
            train_transforms (Compose): Specify the transformations to be applied to the training dataset.
            test_transforms (Compose): Specify the transformations to be applied to the test dataset.
            sample_size (float): Percentage of available images to be used.
            test_size (float): Precantage of sample to be used as test data.
            val_size (float): Percentage of sample to be used as validation data.
            shuffle (bool): Whether or not the data should be shuffled after loading.
        """
        
        root_dir = './Data'
        resource = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
        md5 = "0bc7306e7427e00ad1c5526a6677552d"

        compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
        data_dir = os.path.join(root_dir, "MedNIST")
            
        if download:
            monai.apps.download_and_extract(resource, compressed_file, root_dir, md5)

        # Reading image filenames from dataset folders and assigning labels
        class_names = sorted(x for x in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, x)))
        num_class = len(class_names)

        image_files = [
            [
                os.path.join(data_dir, class_names[i], x)
                for x in os.listdir(os.path.join(data_dir, class_names[i]))
            ]
            for i in range(num_class)
        ]
        
        image_files = [images[:int(len(images) * sample_size)] for images in image_files]
        
        # Constructing data and labels
        num_each = [len(image_files[i]) for i in range(num_class)]
        data = []
        labels = []

        for i in range(num_class):
            data.extend(image_files[i])
            labels.extend([i] * num_each[i])
            
        if shuffle:
            indicies = np.arange(len(data))
            np.random.shuffle(indicies)
            
            data = [data[i] for i in indicies]
            labels = [labels[i] for i in indicies]
        
        # Make train test validation split
        train_data, train_labels, test_data, test_labels, val_data, val_labels = _split(data, labels, 
                                                                                        test_size, val_size)
        
        # Construct and return datasets
        train_ds = MedNISTDataset(train_data, train_labels, train_transforms, shuffle)
        test_ds = MedNISTDataset(test_data, test_labels, test_transforms, shuffle)
        
        if val_size == 0:
            return train_ds, test_ds
        else:
            val_ds = MedNISTDataset(val_data, val_labels, test_transforms, shuffle)
            return train_ds, test_ds, val_ds

        
class IXIT1Dataset(torch.utils.data.Dataset):
    """IXI-T1 Dataset
    
    Consists of ~566 images of 3D Brain MRI scans and labels (0) for male and (1) for female.

    Functions:
        as_tensor(): Will return the dataset as a two torch tensors. Data and Labels.
    """
    def __init__(self, data, labels, transforms, shuffle):
        self.ds = monai.data.ImageDataset(image_files=data, labels=labels, transform=transforms, shuffle=shuffle)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def as_tensor(self):
        data = torch.Tensor([data[0] for data in self.ds])
        labels = torch.Tensor([data[1] for data in self.ds])
        return data, labels
    

class MedNISTDataset(torch.utils.data.Dataset):
    """ Loads the MedNISTDataset
    
    Consists of ~60.000 2D images from 6 classes: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT.
    
    Functions:
        as_tensor(): Will return the dataset as a two torch tensors. Data and Labels.
    """
    
    def __init__(self, data, labels, transforms, shuffle):
        self.transforms = transforms
        self.shuffle=shuffle
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transforms(self.data[index]), self.labels[index]
    
    def as_tensor(self):
        data = [self.transforms(self.data[i]) for i in range(len(self.data))]
        return torch.Tensor(data), torch.Tensor(self.labels)
    
    
def _split(data, labels, test_size, val_size):
    """Helper function for train-test-validation split."""
    length = len(data)
        
    train_data = data[:-int(length*(test_size + val_size))]
    train_labels = labels[:-int(length*(test_size + val_size))]

    test_data = data[-int(length*(test_size + val_size)):ceil(length - length*(val_size))]
    test_labels = labels[-int(length*(test_size + val_size)):ceil(length - length*(val_size))]

    val_data = data[-int(length*val_size):]
    val_labels = labels[-int(length*val_size):]

    return train_data, train_labels, test_data, test_labels, val_data, val_labels