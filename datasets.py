import os
import numpy
import pandas
import monai
import PIL
import torch
import pandas as pd
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor



class IXIT1Dataset(torch.utils.data.Dataset):
    """ Loads the IXI-T1 Dataset
    Args:
    download (bool): Wether or not the Dataset should be downloaded.
    transformers (object): Takes a Compose object which contains
       the tranforms to be applied to the data.
    sample_size (float): Percentage of the full Dataset to be used.

    Functions:
    as_tensor(): Will return the dataset as a two torch tensors. Data and Labels.
    """
    standard_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), RandRotate90()])
    
    def __init__(self, download: bool = False, transforms: object = standard_transforms, 
                 sample_size: float = 0.01, shuffle: bool = True):
        if download:
            data_url = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar'
            compressed_file = os.sep.join(['Data', 'IXI-T1.tar'])
            data_dir = os.sep.join(['Data', 'IXI-T1'])

            # Data download
            monai.apps.download_and_extract(data_url, compressed_file, './Data/IXI-T1')

            # Labels document download
            labels_url = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls'
            monai.apps.download_url(labels_url, './Data/IXI.xls')

        self.ds=None
        self.shuffle=shuffle
        self.sample_size=sample_size
        self.transforms=transforms

        self._load_data()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def _load_data(self):
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

        data, labels = data[:int(len(data) * self.sample_size)], labels[:int(len(data) * self.sample_size)]

        self.ds = monai.data.ImageDataset(image_files=data, labels=labels, transform=self.transforms, shuffle=self.shuffle)

    def as_tensor(self):
        data = torch.Tensor([data[0] for data in self.ds])
        labels = torch.Tensor([data[1] for data in self.ds])
        return data, labels
