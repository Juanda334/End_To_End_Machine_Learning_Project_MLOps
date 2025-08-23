import os
import yaml
import json
import joblib
import base64
from typing import Any
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from End_to_End_ML_Pipeline import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object

    Args:
        path_to_yaml (Path): Path to the yaml file

    Raises:
        e: BoxValueError if the yaml file is empty

    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("The yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """Creates a list of directories

    Args:
        path_to_directories (list): List of directories to be created
        verbose (bool, optional): Whether to log the creation of directories. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok = True)
        if verbose:
            logger.info(f"Directory created at: {path}")    
    
@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary to a json file

    Args:
        path (Path): Path to the json file
        data (dict): Data to be saved
    """
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent = 4)
    logger.info(f"json file: {path} saved successfully")
    
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a json file and returns a ConfigBox object

    Args:
        path (Path): Path to the json file

    Returns:
        ConfigBox: ConfigBox object
    """
    with open(path, 'r') as json_file:
        content = json.load(json_file)
    logger.info(f"json file: {path} loaded successfully")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """Saves data to a binary file using joblib

    Args:
        data (Any): Data to be saved
        path (Path): Path to the binary file
    """
    with open(path, 'wb') as bin_file:
        joblib.dump(data, bin_file)
    logger.info(f"binary file: {path} saved successfully")
    
@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file using joblib

    Args:
        path (Path): Path to the binary file

    Returns:
        Any: Data loaded from the binary file
    """
    with open(path, 'rb') as bin_file:
        data = joblib.load(bin_file)
    logger.info(f"binary file: {path} loaded successfully")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of the file in KB

    Args:
        path (Path): Path to the file

    Returns:
        str: Size of the file in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    logger.info(f"file: {path} has size: {size_in_kb} KB")
    return f"{size_in_kb} KB"

def decode_image(image_string, fileName):
    imgdata = base64.b64decode(image_string)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()
        
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        image_data = f.read()
        return base64.b64encode(image_data) 