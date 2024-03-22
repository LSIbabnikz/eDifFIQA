
import os
import argparse

from utils import *

import torch
from tqdm import tqdm

def prepare_data(beta: float,
                 label_location : str,
                 embedding_location : str,
                 save_location : str,
                 save_name: str) -> None:
    """ Generates necessary data for extended learning proposed by eDifFIQA

    Args:
        beta (float): Percent of images considered "high" quality.
        label_location (str): Location of baseline quality scores.
        embedding_location (str): Location of embeddings.
        save_location (str): Location in which the generated data will be stored.
        save_name (str): Name of the saved files.
    """

    assert os.path.exists(embedding_location), f" Given location for embeddings does not exist! : {embedding_location}"
    assert os.path.exists(label_location), f" Given location for quality labels does not exist! : {label_location}"
    assert os.path.exists(save_location),  f" Given save location does not exist! : {save_location}"

    # Load embeddings of images
    embeddings : dict = load_pickle(embedding_location)

    # Load quality labels of images
    quality_scores : list = load_pickle(label_location)
    quality_scores = dict(quality_scores)

    # Join quality scores of each identity
    identity_quality_scores : dict = {}
    for path, quality in tqdm(quality_scores.items(), " Combining quality scores and image locations "):
        # Obtain identity info from file path
        identity = path.rsplit("/", 1)[0]                             
        identity_quality_scores.setdefault(identity, []).append((path, quality))      

    # Calculate quality threshold for selected "high" quality images
    quality_threshold = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)[int(beta*len(quality_scores.items()))][1]

    average_embeddings = {}
    # Calculate the mean high quality embedding for all training identities
    for identity, paths_qualities in tqdm(list(identity_quality_scores.items()), " Calculating mean embeddings "):
        # Sort all files of given identity by quality scores
        sorted_paths_qualities = sorted(paths_qualities, key=lambda x:x[1], reverse=True)           
        # Select images over the quality threshold
        chosen = [(path, quality) for path, quality in sorted_paths_qualities if quality >= quality_threshold]
        # Incase of too few training samples discard the identity 
        if len(chosen) < 10:
            del identity_quality_scores[identity]
            continue
        # Calculate the average embedding of the chosen samples
        average_embedding = torch.tensor([])
        for item, _ in chosen:
            average_embedding = torch.cat((average_embedding, torch.from_numpy(embeddings[item]).unsqueeze(0)), dim=0)
        average_embedding = torch.mean(average_embedding, dim=0)
        average_embeddings[identity] = average_embedding

    # Combine average embeddings of all identites
    mapping = {}
    reverse_mapping = {}
    average_embedding_batch = torch.tensor([])
    for i, (identity, average_embedding) in enumerate(tqdm(average_embeddings.items(), " Combining average embeddings ")):
        mapping[identity] = i
        reverse_mapping[i] = identity
        average_embedding_batch = torch.cat((average_embedding_batch, average_embedding.unsqueeze(0)), dim=0)

    # Make new training data list
    training_items = []
    for identity, paths_qualities in tqdm(identity_quality_scores.items(), " Creating training list "):
        for path, quality in paths_qualities:
            training_items.append((path, quality, mapping[identity]))

    # Store data needed for training model
    dump_pickle(os.path.join(save_location, f"average_embedding_batch-{save_name}.pkl"), average_embedding_batch)
    dump_pickle(os.path.join(save_location, f"training_items-{save_name}.pkl"), training_items)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--beta", type=float, default=0.2)
    parser.add_argument("-ll", "--label_location", type=str, default="./training_data/vggface2-qs.pkl")
    parser.add_argument("-el", "--embedding_location", type=str, default="./training_data/cosface-embeddings.pkl")
    parser.add_argument("-sl", "--save_location", type=str, default="./training_data")
    parser.add_argument("-sn", "--save_name", type=str, default="ediffiqa")
    args = parser.parse_args()

    prepare_data(
        args.beta,
        args.label_location,
        args.embedding_location,
        args.save_location,
        args.save_name
    )