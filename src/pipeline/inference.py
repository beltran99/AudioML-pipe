import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data.make_dataset import get_data, process_data
from src.logger import get_logger
from src.utils import read_config, parse_data_config
from src.models.models import FCNN, CNN

logging_level = logging.DEBUG
ROOT_DIR = Path(__file__).parent / '../..' # project root

def main(
    saved_model_path: Path,
    test_data_path: Path,
    data_config_path: Path,
    make_report: bool) -> None:
    """Create and execute an inference pipeline.
        1. Read and process test data.
        2. Load trained model.
        3. Perform inference.
        
        Optional:
        4. Create a report of testing results.

    Args:
        saved_model_path (Path): Path to trained model.
        test_data_path (Path): Path to test data.
        data_config_path (Path): Path to config file with parameters for the data processing.
        make_report (bool): Option to make a report with the testing results.
    """
    
    logger = get_logger(__name__, logging_level)
        
    # 1. Load and process test data
    filepaths = get_data(test_data_path)
    data_config = read_config(data_config_path) if data_config_path is not None else {}
    params = parse_data_config(data_config)
    processed_data = process_data(filepaths, params, show_progress=False)
    
    x_test, y_test = zip(*processed_data)
    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = np.array(y_test, dtype=np.int32)
    
    # 2. Load saved model
    # Load model params
    model_name = saved_model_path.stem
    params_path = saved_model_path / (model_name + "_params.json")
    try:
        f = open(params_path)
        input_params = json.load(f)
    except FileNotFoundError as err:
        logger.error(f"File not found in the specified path {params_path}\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error when trying to read saved model parameters\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
        
    # Load model state
    # Create instance of model with saved params and load saved state
    saved_model_path = saved_model_path / (model_name + ".pth")
    model_type = input_params.pop("model_type")
    device = torch.device("cpu")
    try:
        if model_type == "FCNN":
            saved_model = FCNN(**input_params)
        else:
            n_in = x_test.shape[1]
            saved_model = CNN(n_in).to(device)
        saved_model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    except FileNotFoundError as err:
        logger.error(f"File not found in the specified path {saved_model_path}\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error when trying to load saved model state\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    
    # 3. Perform inference
    # Set to inference mode
    saved_model.eval()
    with torch.no_grad():
        if model_type == "CNN":
            x_test = x_test.unsqueeze(1).float()  # (batch, 1, 13, 150)
        predictions = saved_model(x_test)
    
    # Calculate test accuracy
    y_pred = predictions.argmax(dim=1).float().numpy()  # convert softmax prob to label
    test_correct = (y_pred == y_test).sum()
    test_acc = test_correct / len(y_test)
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # 4. Report results
    if make_report:
        report_dir = ROOT_DIR / ("reports/" + model_name)
        report_dir.mkdir(parents=True, exist_ok=True)
        cm_report_path = report_dir / ("cm.png")
        
        logger.info(f"Saving model test reports to {report_dir}")
        try:
            # Create Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=range(10))
            # Plot Confusion Matrix
            disp = ConfusionMatrixDisplay(cm).plot()
            plt.title("Model Confusion Matrix at inference time")
            plt.savefig(cm_report_path)
            logger.info(f"Test report saved succesfully to {cm_report_path}")
        except Exception as err:
            logger.error(f"Unexpected error when trying to save test report\n\tError details: {err=}, {type(err)=}")
            sys.exit(1)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build an inference pipeline.')
    parser.add_argument('--model', required=True, type=str, help='Path to trained model. Please specify path from project root. Usage example: models/mymodel/')
    parser.add_argument('--data-path', required=True, type=str, help='Path to test data. Please specify path from project root. Usage example: data/test/')
    parser.add_argument('--data-config', type=str, help='Path to config file. Please specify path from project root. Usage example: config/data/data.yml')
    parser.add_argument('--report', '-r', action="store_true", help='Generate report with test results.')
    parser.add_argument('--verbose', '-v', action="store_true", help='Log informational messages.')
    args = parser.parse_args()
    
    model_path = ROOT_DIR / args.model
    data_path = ROOT_DIR / args.data_path
    data_config = ROOT_DIR / args.data_config if args.data_config is not None else None
    make_report = args.report
    
    if not args.verbose:
        logging_level = logging.CRITICAL
        
    main(model_path, data_path, data_config, make_report)