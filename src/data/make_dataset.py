import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Union

import librosa
import numpy as np

from src.logger import get_logger
from src.utils import read_config, parse_data_config

ROOT_DIR = Path(__file__).parent / '../..' # project root
logging_level = logging.DEBUG

def process_raw_data(raw_data: np.array, sample_rate: int, params: dict) -> None:
    """Extract MFCCs from input audio file, given as a time series.

    Args:
        raw_data (np.array): Time series audio.
        sample_rate (int): Sampling rate of audio.
        params (dict): ...
            n_mfcc (int): Number of coefficients to extract from raw data.
            n_ftt (int): ...
            hop_length (int): ...
            aggregate (bool): ...
    """
    
    max_len = params.pop("padding")
    aggregate = params.pop("aggregate_over_t")
    
    if params.get("n_fft", None) is None:
        params["n_fft"] = int(sample_rate * 0.032) # 32 ms window
    if params.get("hop_length", None) is None:
        params["hop_length"] = int(sample_rate * 0.01) # 10 ms hop

    mfcc = librosa.feature.mfcc(
        y=raw_data,
        sr=sample_rate,
        fmin=20,
        fmax=int(sample_rate/2), # Nyquist freq.
        **params
    )
    if aggregate:
        mfcc = np.mean(mfcc.T, axis=0)  # aggregate over t (time)
    else:
        # pad or truncate
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
    
    return mfcc

def get_data(data_path: Path) -> list:
    """Find all audio files in specified path.

    Args:
        data_path (Path): Path containing raw audio files.

    Returns:
        list: List of paths of all audio files in the dataset.
    """
    
    logger = get_logger(__name__, logging_level)
    
    # avoids inefficient nested loops
    filepaths = [
        Path(os.path.join(root, file))
        for root, dirs, files in os.walk(str(data_path)) if not dirs # take only leaf nodes
        for file in files
    ]
    if not filepaths:
        logger.warning(f"Could not find any files in specified path {data_path}")
        
    return filepaths

# def process_data(filepaths: list, n_mfcc: int, resampling_rate: int, show_progress: bool = True) -> list:
def process_data(filepaths: list, params: dict, show_progress: bool = True) -> list:
    """Dataset generator.
    Process raw audio files by extracting MFCCs and labels, and make pairs of (features, label).

    Args:
        filepaths (list): List of paths to all raw audio files in the dataset.
        params (dict): ...
            n_mfcc (int): Number of MFCCs to extract from audio files.
            resampling_rate (int): Resampling rate for input audio.
        show_progress (bool, optional): Option for logging the progress of the data processing. Defaults to True.

    Returns:
        list: List of (features, label) pairs.
    """
        
    logger = get_logger(__name__, logging_level)
    
    resampling_rate = params.pop("resampling_rate")
    
    dataset = []    # tuples of (input_features, label)
    n_files = len(filepaths)
    progress = int(n_files * 0.2) # Log progress in 20% completion increments
    count = 0
    for filepath in filepaths:
        # storing the speaker info may be useful if we wanted to
        # split the data in a specific way to avoid imbalance
        speaker, filename = filepath.parent.name, filepath.stem
        # label is expected to be at the beggining of filename, followed by underscore
        label = int(filename.split("_")[0])
        # reading files one by one seems inefficient,
        # there might be an efficient way of batching them
        raw_data, sample_rate = read_file(filepath, resampling_rate)
        extracted_features = process_raw_data(raw_data, sample_rate, params.copy())
        
        dataset.append((extracted_features, label))
        
        count +=1
        if count % progress == 0 and show_progress:
            pctg = (count / n_files)*100
            logger.info(f"Current progress: {pctg:.2f}%. Processed [{count}/{n_files}] audio files.")
            
    logger.info("Finished processing audio files.")
        
    return dataset

def read_file(filepath: Path, sr: int) -> tuple:
    """Load an audio file as a time series.

    Args:
        filepath (Path): Raw audio file path.
        sr (int): Target sampling rate for the audio file.

    Returns:
        tuple: Pair of time series and audio's sampling rate.
    """
    
    logger = get_logger(__name__, logging_level)
    
    assert filepath.is_file()
    try:
        raw_data, sample_rate = librosa.load(filepath, sr=sr)
    except FileNotFoundError as err:
        logger.error(f"Audio file not found in the specified path {filepath}\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error when trying to read audio file\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    
    return raw_data, sample_rate

def save_dataset(dataset: list, name: str) -> None:
    """Save dataset to disk.

    Args:
        dataset (list): Dataset to be saved.
        name (str): Dataset name.
    """
    
    logger = get_logger(__name__, logging_level)

    if not dataset:
        logger.warning(f"Dataset {name} is empty. Cannot save an empty dataset.")
        sys.exit(1)
        
    new_dataset_path = ROOT_DIR / ("data/processed/" + name + ".npz")
    x, y = zip(*dataset)

    try:
        np.savez(new_dataset_path, x=x, y=y)
        logger.info(f"Dataset succesfully saved to {new_dataset_path}.")
    except Exception as err:
        logger.error(f"Unexpected error when trying to save processed data\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)

# def main(path: Path, n_mfcc: int, resampling_rate: int, name: str) -> None:
# def main(path: Path, params: dict, name: str) -> None:
def main(path: Path, config: Union[Path, None], name: str) -> None:
    """Generate a dataset suitable for supervised learning. Extract MFCCs from audio files, and save processed dataset to disk. 

    Args:
        path (Path): Raw dataset path.
        params (dict): Dict of parameters for the feature extraction.
            n_mfcc (int): Number of Mel-frequency cepstral coefficients (MFCCs) to extract from raw audio files.
            resampling_rate (int): Resampling rate for input audio.
        name (str): Dataset name.
    """
    
    logger = get_logger(__name__, logging_level)
    
    # 1. Process raw data
    logger.info(f"Processing raw data from {path}")
    data_config = read_config(config) if config is not None else {}
    params = parse_data_config(data_config)
    filepaths = get_data(path)
    dataset = process_data(filepaths, params)
    
    # 2. Save processed data for future use
    logger.info(f"Saving processed dataset {name}")
    save_dataset(dataset, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process raw input data.')
    parser.add_argument('--path', '-p', type=str, required=True, help='Path to raw data. Please specify path from project root. Usage example: data/raw/')
    parser.add_argument('--config', type=str, help='Path to config file. Please specify path from project root. Usage example: config/data.yml')
    parser.add_argument('--name', type=str, default="mydataset", help='Dataset name.')
    parser.add_argument('--verbose', '-v', action="store_true", help='Log informational messages.')
    args = parser.parse_args()
    
    path = ROOT_DIR / args.path
    config = ROOT_DIR / args.config if args.config is not None else None
    name = args.name
    
    if not args.verbose:
        logging_level = logging.ERROR
        
    main(path, config, name)