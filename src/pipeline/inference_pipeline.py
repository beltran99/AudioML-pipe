from pathlib import Path
import argparse
import logging
import json
import torch

from src.models.models import FCNN

ROOT_DIR = Path(__file__).parent / '../..' # project root

def main(saved_model_path: Path):
    
    # Load model params
    params_path = saved_model_path.parent / (saved_model_path.stem + "_params.json")
    with open(params_path) as f:
        input_params = json.load(f)
        
    # Load model weights and biases
    saved_model = FCNN(**input_params)
    saved_model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    
    # Set to inference mode
    saved_model.eval()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build an inference pipeline.')
    parser.add_argument('--model', required=True, type=str, help='Path to trained model. Please specify path from project root. Usage example: models/mymodel.pth')
    parser.add_argument('--verbose', '-v', action="store_true", help='Log informational messages.')
    args = parser.parse_args()
    
    model_path = ROOT_DIR / args.model
    
    logging_level = logging.DEBUG
    if not args.verbose:
        logging_level = logging.CRITICAL
        
    main(model_path)