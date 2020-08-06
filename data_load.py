import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image



class ShapeDimensionDataset(Dataset):
    """Datset for 2D shape dimensions."""


    # TODO: add a method that returns un-scaled dimensions ...
    
    def __init__(self, csv_file, root_dir, transform=None, target_cols=None):
        """
        Args:
            - csv_file (string): Path to the csv file with scaled dimensions.
            - root_dir (string): Directory with all the images.
            - transform (callable, optional): Optional transform to be applied
              on a sample.
            - target_cols: list of strings, column names of scaled targets in pandas
        """
        self.shape_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_cols = target_cols


    def __len__(self):
        return len(self.shape_data)


    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.shape_data.filename.iloc[idx])
        
        # torchvision transform want PIL images
        image = Image.open(image_name)
        # convert to grayscale
        image = image.convert(mode="L")

        scaled_dims = self.shape_data.loc[idx, self.target_cols].values 
        scaled_dims = scaled_dims.astype('float').reshape(-1, 1)

        # need to transform the image only -> length etc. does not change.
        # this is true as long as no part of the cross section is cropped, 
        # and the target parameters are invariant to translation, rotation, 
        # scaling.
        # RandomCrop should not be used - it could cut away some of the image 
        if self.transform:
            image = self.transform(image)

        return image, scaled_dims

