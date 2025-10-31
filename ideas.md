# Ideas

## Problem definition
Implement an Audio Classification Model to identify spoken digits 0-9.

- Input: audio files
  - recording of a digit from 0-9 (assumes only one digit per recording)
  - assumes a popular audio codec, e.g., WAV, MP3, FLAC.
  - assumes audio is mono (otherwise take first channel)
- Output: predicted digit class from 0-9

## Pipeline
### Training pipeline
1. Reads raw data and processes it
2. Builds model
3. Trains model on given train/eval split
4. Saves trained model
 
### Inference pipeline
1. Reads raw data and processes it
2. Loads trained model
3. Performs inference and reports results

## Model
1. Baseline: simple, fully-connected neural network that receives MFCCs as input
2. Convolutional network: receives mel-scaled spectrograms as input

## Dataset
- 50 English recordings per digit (0-9) of 60 speakers (50x10x60 = 30000 recordings)
- Diverse range of accents (around 70% have a German accent), country of origin, and age (22 to 61 years old)
- The genders represented in the dataset are unbalanced, with around 80% being men: 48 men, 12 women.
- Reference: [Hugging Face Dataset Card](https://huggingface.co/datasets/gilkeyio/AudioMNIST)