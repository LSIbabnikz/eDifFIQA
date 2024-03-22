
import os
import yaml
import random
import pickle
import importlib
from typing import Tuple
from collections import OrderedDict

import numpy as np

import torch
from torchvision.transforms import Compose


def seed_all(seed: int) -> None:
    """ Seeds python.random, torch and numpy,

    Args:
        seed (int): Desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(module: str, weights: str) -> torch.nn.Module:
    """ Loads the model architecture using specified module and weights location

    Args:
        module (str): Name of the module and function which loads the model architecture.
        weights (str): Location of the pretrained model weights.

    Returns:
        torch.nn.Module: Pretrained FR model.
    """

    module_name, func_name = module.rsplit(".", 1)

    module = importlib.import_module(module_name)
    importlib.reload(module)

    model_func = getattr(module, func_name)
    model : torch.nn.Module = model_func()

    if weights != "":
        model.load_state_dict(torch.load(weights))

    return model


class Arguments:
    """ Empty class definition for constructing any type of arguments from configs.
    """
    pass


def convert_single_layer(config: dict, arguments_obj: Arguments):
    """ Converts a single layer (level) of a config dictionary into the Arguments class

    Args:
        config (dict): Dictionary of provided configuration.
        arguments_obj (Arguments): Argument class of the parent configuration.

    Returns:
        Arguments: Argument class of current configuration.
    """

    for key, value in config.items():
        if isinstance(value, dict):
            setattr(arguments_obj, key, convert_single_layer(value, Arguments()))
        else:
            setattr(arguments_obj, key, value)
    
    return arguments_obj


def parse_config_file(config_loc: str) -> dict:
    """ Constructs an argument class from the configuration located at <config_loc>.

    Args:
        config_loc (str): Location of the configuration file.

    Returns:
        Argument: Argument class 
    """
    
    assert os.path.exists(config_loc), f" Given config path ({config_loc}) does not exist!"
    with open(config_loc, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    arguments = convert_single_layer(config, Arguments())
    return arguments


def construct_transformation(transformation_arguments: Arguments) -> Compose:
    """ Constructs a composition of transformations given by the transformation arguments.

    Args:
        transformation_arguments (Arguments): Arguments of the transformation.

    Returns:
        Compose: Torchvision Compose object of given transformations.
    """

    transforms_list = []
    idx = 1
    while True:
        try:
            trans_args = getattr(transformation_arguments, f"trans_{idx}")
            module_name, function_name = trans_args.module.rsplit(".", 1)
            module = importlib.import_module(module_name)
            trans_function = getattr(module, function_name)
            if hasattr(trans_args, "params"):
                transforms_list.append(trans_function(**vars(getattr(trans_args, "params"))))
            else:
                transforms_list.append(trans_function())
        except AttributeError:
            break
        idx += 1

    return Compose(transforms_list)


def construct_full_model(config_loc: str) -> Tuple[torch.nn.Module, Compose]:
    """ Construct a torch.nn.Module model and its transform from the config provided in config_loc.

    Args:
        config_loc (str): Location of the config file containing information about the model.

    Returns:
        Tuple[torch.nn.Module, Compose]: Returns the constructed model and given transformation.
    """

    args = parse_config_file(config_loc)

    base_model_args = args.base_model
    head_args = args.mlp
    ediffiqa_args = args.ediffiqa

    base_model = load_model(base_model_args.module, base_model_args.weights)
    trans = construct_transformation(base_model_args.transformations)
 
    module_name, function_name = head_args.module.rsplit(".", 1)
    module = importlib.import_module(module_name)
    head_function = getattr(module, function_name)
    model_head = head_function(**vars(getattr(head_args, "params")))

    module_name, function_name = ediffiqa_args.module.rsplit(".", 1)
    module = importlib.import_module(module_name)
    ediffiqa_function = getattr(module, function_name)
    ediffiqa_model = ediffiqa_function(
        backbone_model=base_model,
        quality_head=model_head,
        **vars(getattr(ediffiqa_args, "params"))
        )

    return ediffiqa_model, trans


def load_module(module_args: Arguments, call=True):
    """ Loads module defined by given arguments: name given by using <module: 'name'> and parameters by using <params: ...>.

    Args:
        module_args (Arguments): Arguments from which to construct the desired module.
        call (Bool): Changes the return value, if call=True calls the function before returning results, otherwise not.

    Returns:
        func: Returns the returned values called by the desired module function call or the callable function if call=False.
    """

    module_name, function_name = module_args.module.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function_ = getattr(module, function_name)
    if call:
        if hasattr(module_args, "params"):
            return function_(**vars(getattr(module_args, "params")))
        else:
            return function_()
    else:
        return function_


def construct_optimizer(optimizer_args: Arguments, model: torch.nn.Module):
    """ Constructs an torch optimizer class from the given arguments using the provided model.

    Args:
        optimizer_args (Arguments): Arguments of the optimizer.
        model (torch.nn.Module): Model for which to construct the optimizer.

    Returns:
        torch.optim.Optimizer: The constructed optimizer class.
    """

    optimizer_cls = load_module(optimizer_args, call=False)

    optimizer_groups = []
    idx = 1
    while True:
        try:
            group_args = getattr(optimizer_args, f"group_{idx}")
            group_params = vars(group_args)
            group_params["params"] = getattr(model, group_params["params"]).parameters()
            if 'limit' in group_params:
                group_params["params"] = list(group_params["params"])[-group_params["limit"]:]
                group_params.pop("limit")
            optimizer_groups.append(group_params)
        except AttributeError:
            break
        idx += 1

    return optimizer_cls(optimizer_groups)


def args_to_dict(arguments: Arguments, dictionary: dict) -> dict:
    """ Converters multi-level Arguments to multi-level dictionary objects (for printing and such).

    Args:
        arguments (Arguments): Arguments to be converted.
        dictionary (dict): Empty dictionary.

    Returns:
        dict: Converted arguments into dictionary form.
    """

    for key, value in vars(arguments).items():
        if isinstance(value, Arguments):
            dictionary[key] = args_to_dict(value, {})
        else:
            dictionary[key] = value
        
    return dictionary


def isimagefile(loc: str, exts=[".png", ".jpg", ".jpeg", ".tiff", ".bmp"]) -> bool:
    """ Helper function to determine if files are indeed images.

    Args:
        loc (str): Location of file to be checked.
        exts (list, optional): Possible file types for image files. Defaults to [".png", ".jpg", ".jpeg", ".tiff", ".bmp"].

    Returns:
        bool: Returns True if file is image type or False otherwise.
    """
    for ext in exts:
        if loc.endswith(ext):
            return True
    return False


def load_pickle(loc: str):
    """ Helper function for loading in pickled data.

    Args:
        loc (str): Location of the pickle file.
    """
    with open(loc, "rb") as pkl_in:
        data = pickle.load(pkl_in)
    return data

def dump_pickle(loc: str, data):
    """ Helper function for dumping data to pickled files.

    Args:
        loc (str): Location of the pickle file.
        data (): Data to be pickled.
    """
    with open(loc, "wb") as pkl_out:
        pickle.dump(data, pkl_out)


if __name__ == "__main__":

    ...