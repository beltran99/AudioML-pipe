# AudioML-pipe

## Features
- Full machine learning pipeline with data preprocessing, and model training and evaluation.
- Model and training loop implemented in PyTorch.
- Examples of configuration files in [config/data/](config/data/) and [config/models/](config/models/).
- Examples of already processed datasets in [data/processed/](data/processed/), as well as test audio files [data/test/](data/test/).
- Examples of already trained models in [models/](models/).

## Requirements
- Python 3.10.8
- virtualenv 20.0.17

## Installation
Clone the repository into your working directory
```bash
git clone https://github.com/beltran99/AudioML-pipe.git
```

Navigate to the repository folder
```bash
cd AudioML-pipe
```

Install virtualenv if you don't have it installed already
```bash
pip install virtualenv
```

Create a new virtual environment called env
```bash
virtualenv env
```

Activate the environment
```bash
source env/bin/activate # in Linux
```

Installed the required packages in the virtual environment
```bash
pip install -r requirements.txt
```

## Example usage
### Create a dataset
```bash
python3 -m src.data.make_dataset --data-path data/raw/ --config-path config/data/data_20f_agg.yml --name 20f_agg -v
```

### Train a model
```bash
python3 -m src.pipeline.train --data-path data/processed/20f_agg.npz --config-path config/models/config_20f.yml -r -p -v
```

### Test the trained model
```bash
python3 -m src.pipeline.inference --model models/mymodel_20_coeff_v010 --data-path data/test/ --config config/data/data_20f_agg.yml -r -v
```