
import os
import pickle

import torch
from torchvision.transforms import Compose

import numpy as np
from PIL import Image

from utils import *


class WrapperDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 item_list,
                 average_embedding_batch,
                 trans,
                 embeddings) -> None:
        """ Helper class for training and validation dataset.

        Args:
            item_list (list): List of all items.
            trans (Compose): The transformation to use on loaded images.
        """

        self.item_list = item_list
        self.trans = trans
        self.average_embedding_batch = average_embedding_batch
        self.embeddings = embeddings

    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, x):
        location, score, mean_embedding_idx = self.item_list[x]
        img_name = location.split("/")
        img_name = img_name[-3] +"/"+ img_name[-2] +"/"+ img_name[-1]
        img = Image.open(location).convert("RGB")
        embedding = self.embeddings[img_name] 
        return (self.trans(Image.open(location).convert("RGB")), torch.tensor(score), embedding, mean_embedding_idx)


def construct_datasets(dataset_args: Arguments, trans: Compose) -> Tuple[WrapperDataset, WrapperDataset]:
    """ Constructs the train and validation datasets for training the eDifFIQA approach.

    Args:
        dataset_args (Arguments): Arguments of the dataset.
        trans (Compose): Transformation of images used for training.

    Returns:
        Tuple[Dataset, Dataset]: Returns the tuple of trainig and validation datasets.
    """

    # Read all quality attribute files
    with open(f"./training_data/training_items-{dataset_args.model_name}.pkl", "rb") as pkl_in:
        training_items = pickle.load(pkl_in)

    with open(f"./training_data/average_embedding_batch-{dataset_args.model_name}.pkl", "rb") as pkl_in:
        average_embedding_batch = pickle.load(pkl_in)
        
    # Prepare training items
    training_items = list(map(lambda x: (os.path.join(dataset_args.image_loc, x[0]), x[1], x[2]), training_items))
    training_items = list(filter(lambda x: os.path.isfile(x[0]), training_items))
    training_items.sort(key=lambda x: x[1])
    # Normalize prior qualities to 0-1
    min_score, max_score = min(training_items, key=lambda x: x[1])[1], max(training_items, key=lambda x: x[1])[1]
    training_items = list(map(lambda x: (x[0], (x[1] - min_score) / (max_score - min_score), x[2]), training_items))
    
    with open(f"/media/sda/ziga-work/datasets/vggface2/{dataset_args.backbone_model}-embeddings.pkl", "rb") as pkl_in:
        embeddings = {k: torch.from_numpy(v) for k, v in pickle.load(pkl_in).items()}

    # Split data into a training and validation set
    test_indices = np.arange(0, len(training_items), 1./dataset_args.val_split)
    val_items = [training_items[int(k)] for k in test_indices]
    train_items = list(set(training_items).difference(set(val_items)))

    return  WrapperDataset(train_items,
                           average_embedding_batch, 
                           trans,
                           embeddings), \
            WrapperDataset(val_items, 
                           average_embedding_batch,
                           trans,
                           embeddings)


class ImageDataset():

    def __init__(self, 
                 image_loc, 
                 trans) -> None:
        """ Helper class that loads all images from a given directory.

        Args:
            image_loc (str): The location of the directory containing the desired images.
            trans (Compose): Transformations used on loaded images.
        """
        self.image_loc = image_loc
        self.trans = trans

        self.items = []
        for (dir, subdirs, files) in os.walk(self.image_loc):
            self.items.extend([os.path.join(dir, file) for file in files if isimagefile(file)])

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, x):
        path = self.items[x]
        return (path, self.trans(Image.open(path).convert("RGB")))