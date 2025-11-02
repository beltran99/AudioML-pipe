import sys
from collections.abc import Iterator
from pathlib import Path

import torch.optim as optim
import yaml

from src.logger import get_logger
from src.models.models import DEFAULT_FCNN, DEFAULT_OPT


def read_config(config_path: Path) -> dict:
    """Read YAML file containing experiment configuration. Read model configuration and optimizer configuration.

    Args:
        config_path (Path): Path to the YAML config file.

    Returns:
        dict: Loaded configuration.
    """
    logger = get_logger(__name__)
    logger.info(f"Reading model config")
    if config_path is None:
        return {}
    
    try:
        stream = open(config_path)
        config = yaml.safe_load(stream)
    except FileNotFoundError as err:
        logger.error(f"File not found in the specified path {config_path}\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)        
    except yaml.YAMLError as exc:
        logger.error(f"Error when reading YAML file {config_path}\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error when trying to read YAML config\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
        
    return config

def parse_model_config(model_config: dict) -> dict:
    """Parse model configuration.

    Args:
        model_config (dict): Raw model config from YAML file.

    Returns:
        dict: Parsed configuration that can serve as parameters to instantiate the model class.
    """
    
    logger = get_logger(__name__)
    logger.info(f"Parsing model config")
    
    # read model layers from config file
    input_size = model_config.get("model", {}).get("layers", {}).get("input_size", {})
    hidden_sizes = model_config.get("model", {}).get("layers", {}).get("hidden_sizes", {})
    output_size = model_config.get("model", {}).get("layers", {}).get("output_size", {})

    # read activation function from config file
    activation_fn = model_config.get("model", {}).get("activation", {})

    return dict(input_size = DEFAULT_FCNN["input_size"] if not input_size else input_size,
                hidden_sizes = DEFAULT_FCNN["hidden_sizes"] if not hidden_sizes else hidden_sizes,
                output_size = DEFAULT_FCNN["output_size"] if not output_size else output_size,
                activation_fn = DEFAULT_FCNN["activation_fn"] if not activation_fn else activation_fn)
    
def parse_optimizer_config(model_params: Iterator, opt_config: dict):
    """Parse optimizer configuration and create an optimizer instance.

    Args:
        model_params (Iterator): Iterator over model parameters.
        opt_config (dict): Optimizer raw config.

    Returns:
        Optimizer object.
    """
    
    logger = get_logger(__name__)
    logger.info(f"Parsing optimizer config")
    
    opt_name = opt_config.get("type", DEFAULT_OPT["type"])
    
    if opt_name and hasattr(optim, opt_name):   # opt type is a torch.optim class
        
        opt_params = opt_config.get("params", {})
        if 'lr' in opt_params:
            opt_params['lr'] = float(opt_params['lr'])  # YAML resolves scientific notation to str
            
        opt_fn = getattr(optim, opt_name)
        try:
            optimizer = opt_fn(model_params, **opt_params)
        except Exception as err:
            logger.error(f"Unexpected error when trying to parse optimizer config\n\tError details: {err=}, {type(err)=}")
            sys.exit(1)
            
    return optimizer