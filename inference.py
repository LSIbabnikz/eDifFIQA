
import os
import pickle
import argparse

import torch 
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import *
from dataset import *

from model.ediffiqa import EDIFFIQA_CONF


@torch.no_grad()
def inference(args):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Load the training FR model and construct the transformation
    conf_loc, weight_loc = EDIFFIQA_CONF[args.model]
    model, trans = construct_full_model(conf_loc)
    model.load_state_dict(torch.load(weight_loc))
    model.to(args.base.device).eval()

    # Construct the Image Dataloader 
    dataset = ImageDataset(args.dataset.loc, trans)
    dataloader = DataLoader(dataset, **args_to_dict(args.dataloader.params, {}))

    # Predict quality scores 
    quality_scores = {}
    for (name_batch, img_batch) in tqdm(dataloader, 
                                        desc=" Inference ", 
                                        disable=not args.base.verbose):

        img_batch = img_batch.to(args.base.device)
        preds = model(img_batch).detach().squeeze().cpu().numpy()
        quality_scores.update(dict(zip(name_batch, preds)))

    # Save to designated location
    with open(os.path.join(args.base.save_path, f"quality-scores.pkl"), "wb") as pkl_out:
        pickle.dump(quality_scores, pkl_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
        help=' Location of the eDifFIQA inference configuration. '
    )
    args = parser.parse_args()
    arguments = parse_config_file(args.config)

    inference(arguments)