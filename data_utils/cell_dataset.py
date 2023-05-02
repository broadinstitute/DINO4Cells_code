import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage import io
import numpy as np
from tqdm import tqdm
from skimage import exposure
from torchvision.transforms import ToTensor
import pandas as pd

import torch.nn.functional as F

class dino_dataset(torch.utils.data.Dataset):
    """
    Class build on top of PyTorch dataset
    used to load training, validation and test datasets

    inputs:
    - dataframe (required): a pandas dataframe conatining at least
    three columns, namely, Path, Class_ID, Batch_ID.
    - root_dir: a string to be added as a prefix to the "Path"
    - transform (optional): any image transforms. Optional.
    - class_dict (optional): a dictionary that converts strings
    in Class_ID column to numerical class labels
    - class_dict (optional): a dictionary that converts strings
    in Batch_ID column to numerical class labels

    returns:
    - a three-tuple with the image tensor (N,C,W,H) or (C,W,H),
    a numerical class id, and a numerical batch id.
    The id's are numerical values inferred from the
    class and batch dictionaries. An np.nan is returned
    if the keys are unavailable in the dictionaries.
    """
    
    def __init__(self, dataframe, root_dir='/home/ubuntu/data/CellNet_data/Hirano3D_v2.0/data/', transform=None,
                 label_dicts=None, RGBmode=False, training=True):
        
        if not isinstance(dataframe, pd.DataFrame):
            dataframe=pd.read_csv(dataframe)
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.RGBmode = RGBmode
        self.train = training
        
        # store unique class and batch IDs
        self.classes = [cls for cls in self.dataframe.Class_ID.unique()]
        self.batches = [bt for bt in self.dataframe.Batch_ID.unique()]
        if label_dicts  is not None:
            print('label_dict received!')
            self.class_dict, self.batch_dict = label_dicts['class_dict'], label_dicts['batch_dict']
        else: 
            self.class_dict = {v: k for k, v in enumerate(self.classes)}
            self.batch_dict = {v: k for k, v in enumerate(self.batches)}
            self.label_dicts = {'class_dict': self.class_dict, 'batch_dict': self.batch_dict}
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        im = io.imread(self.root_dir + row["Path"])
        im = exposure.rescale_intensity(im, in_range='image', out_range=(0, 255))
        im = np.float32(im)
        im = np.divide(im, 255)
        im = ToTensor()(im)
        
        if self.RGBmode: im = im[:3,...]
        if self.transform is not None:
            im = self.transform(im)
        
        if self.train:
            return (im,
                    torch.tensor(self.class_dict[row["Class_ID"]]))        
        else:
            return (im,
                    row['Class_ID'],
                    row['Batch_ID'])
                
               
#                 torch.tensor(self.class_dict[row["Class_ID"]] if row["Class_ID"] in
#                              self.class_dict.keys() else 100)
#                 ,
#                 torch.tensor(self.batch_dict[row["Batch_ID"]] if row["Batch_ID"] in
#                              self.batch_dict.keys() else 100),
#                 )
