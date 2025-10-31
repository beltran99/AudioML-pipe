from pathlib import Path
import argparse
import librosa
import logging
import os
import numpy as np

ROOT_DIR = Path(__file__).parent / '../..' # project root

def process_raw_data(raw_data, sample_rate, n_mfcc):

    extracted_features = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc=n_mfcc)
    extracted_features = np.mean(extracted_features.T, axis=0)
    
    return extracted_features

def load_data(data_path, n_mfcc):
    
    dataset = []    # tuples of (input_features, label)
    
    # avoids inefficient nested loops
    filepaths = [
        Path(os.path.join(root, file))
        for root, dirs, files in os.walk(str(data_path)) if not dirs # take only leaf nodes
        for file in files 
    ]
    for filepath in filepaths:
        # storing the speaker info may be useful if we wanted to
        # split the data in a specific way to avoid imbalance
        speaker, filename = filepath.parent.name, filepath.stem
        # label is expected to be at the beggining of filename, followed by underscore
        label = int(filename.split("_")[0])
        raw_data, sample_rate = read_file(filepath)
        extracted_features = process_raw_data(raw_data, sample_rate, n_mfcc)
        
        dataset.append((extracted_features, label))
        
    return dataset

def read_file(filepath):
    assert filepath.is_file()
    try:
        raw_data, sample_rate = librosa.load(filepath)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    
    return raw_data, sample_rate

def save_dataset(dataset, name='mydataset'):
    new_dataset_path = ROOT_DIR / ("data/processed/" + name + ".npz")
    x, y = zip(*dataset)

    try:
        np.savez(new_dataset_path, x=x, y=y)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process raw input data.')
    parser.add_argument('--path', '-p', type=str, default= ROOT_DIR / ("data/raw/") , help='Path to raw data. Please specify path from project root.')
    parser.add_argument('--mfcc', type=int, default=40, help='Number of Mel-frequency cepstrum coefficients to extract from the input data.')
    parser.add_argument('--verbose', '-v', action="store_true", help='Log informational messages.')
    args = parser.parse_args()
    
    path = args.path
    n_mfcc = args.mfcc
    
    logging_level = logging.DEBUG
    if not args.verbose:
        logging_level = logging.CRITICAL
        
    dataset = load_data(path, n_mfcc)
    save_dataset(dataset)