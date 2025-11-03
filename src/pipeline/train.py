import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data import make_dataset
from src.logger import get_logger
from src.models.models import FCNN, CNN
from src.utils import (parse_model_config, parse_optimizer_config, parse_training_loop_config, read_config)

logging_level = logging.DEBUG
ROOT_DIR = Path(__file__).parent / '../..' # project root    


def train(
        model: nn.Module,
        x: np.array,
        y: np.array,
        opt_config: dict,
        train_config: dict
    ) -> tuple:
    """Training loop. Train a model on given data with specified configuration.

    Args:
        model (nn.Module): Model to be trained.
        x (np.array): Feature variables from dataset.
        y (np.array): Target variable from dataset.
        opt_config (dict): Optimizer configuration.
        train_config (dict): Training loop configuration.

    Returns:
        tuple: Trained model and reported metrics.
    """
    
    logger = get_logger(__name__, logging_level)
    
    # Important data check
    # Check that the input data has the same shape as the input layer of the model
    # if x.shape[1] != model.input_size:
    #     logger.warning(f"Invalid configuration: input data shape must match model's input layer size\n\tInput data shape: {x.shape}, Model input layer size: {model.input_size}")
    #     sys.exit(1)
    
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=train_config['eval_split'], random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders for batching
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    batch_size = train_config['batchsize']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # reshuffle data at every epoch
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    optimizer = parse_optimizer_config(model.parameters(), opt_config)
    # loss_fn = nn.CrossEntropyLoss() # could also be parametrized
    loss_fn = nn.NLLLoss() # could also be parametrized
    
    # Training history dict
    history = {
        "train_loss": {},
        "train_acc": {},
        "val_loss": {},
        "val_acc": {},
    }
    
    # Training loop
    early_stop = train_config['earlystop']
    epochs = train_config['epochs']
    for epoch in range(epochs):
        ### Training step ###
        model.train() # set the model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            X_batch, y_batch = batch
            
            # X_batch = X_batch.unsqueeze(1).float()  # (batch, 1, 13, 150)

            # Forward pass
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad() # clear previous gradients
            loss.backward() # compute gradients
            optimizer.step() # update parameters

            # Update training loss
            train_loss += loss.item()
            
            # Calculate training accuracy
            predicted_labels = predictions.argmax(dim=1).float().numpy() # convert softmax prob to label
            train_correct += (predicted_labels == y_batch.numpy()).sum()
            train_total += y_batch.size()[0]

        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        ### END Training step ###

        ### Validation step ###
        model.eval() # set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # disable gradient computation for evaluation
            for batch in val_loader:
                X_batch, y_batch = batch
                # X_batch = X_batch.unsqueeze(1).float()  # (batch, 1, 13, 150)
                predictions = model(X_batch)
                val_loss += loss_fn(predictions, y_batch).item()

                # Calculate validation accuracy
                predicted_labels = predictions.argmax(dim=1).float().numpy() # convert softmax prob to label
                val_correct += (predicted_labels == y_batch.numpy()).sum()
                val_total += y_batch.size()[0]

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        ### END Validation step ###
        
        history['train_loss'][epoch+1] = avg_train_loss
        history['train_acc'][epoch+1] = train_acc
        history['val_loss'][epoch+1] = avg_val_loss
        history['val_acc'][epoch+1] = val_acc
        
        logger.info(
            f"Epoch {epoch+1:>1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        if early_stop and len(history['val_loss']) > 1 and avg_val_loss > history['val_loss'][epoch]:
            logger.info(f"Early stopping: Val. loss has increased from {history['val_loss'][epoch]:.4f} to {avg_val_loss:.4f}!")
            break
        
    return model, history

def main(
        data_path: Path,
        config_path: Union[Path, None],
        is_persistent: bool,
        make_report: bool) -> None:
    """Create and execute a training pipeline.
        1. Load processed dataset from disk.
        2. Build model with specified parameters.
        3. Train model.
        
        Optional:
        4. Save trained model.
        5. Create a report of training results.

    Args:
        data_path (Path): Processed dataset path.
        config_path (Union[Path, None]): Optional. Path to config file containing
        configuration of model, optimizer, and training loop. If not provided, the default
        configuration will be considered.
        is_persistent (bool): Option to save the trained model.
        make_report (bool): Option to make a report with the training results.
    """
    
    logger = get_logger(__name__, logging_level)
    
    # 1. Load processed dataset
    logger.info(f"Loading processed dataset from {data_path}")
    x = y = None
    try:
        npzfile = np.load(data_path)
        x = npzfile['x']
        y = npzfile['y']
    except FileNotFoundError as err:
        logger.error(f"File not found in the specified path {data_path}\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error when trying to read dataset\n\tError details: {err=}, {type(err)=}")
        sys.exit(1)
    
    # 2. Build model
    logger.info(f"Building model")
    config = read_config(config_path)
    model_params = parse_model_config(config)
    train_loop_config = parse_training_loop_config(config)
    
    device = torch.device("cpu")
    model = FCNN(**model_params).to(device)
    # model = CNN().to(device)
    logger.info(f"Model built succesfully!\n{model}")
    
    # 3. Train model
    logger.info(f"Loading training configuration")
    opt_config = config.get("optimizer", {})
    trained_model, history = train(model, x, y, opt_config, train_loop_config)
    
    # 4. Save trained model
    model_name = config.get("name", "mymodel")
    model_version = config.get("version", "")
    model_version = str(model_version).replace(".", "") if model_version else ""
    model_name = model_name + "_v" + model_version if model_version else model_name
    
    if is_persistent:   
        model_dir = ROOT_DIR / ("models/" + model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / (model_name + ".pth")
        model_params_path = model_dir / (model_name + "_params.json")
        
        logger.info(f"Saving trained model to {model_dir}")
        try:
            torch.save(trained_model.state_dict(), model_path)
            logger.info(f"Model state saved succesfully to {model_path}")
            
            f = open(model_params_path, 'w', encoding='utf-8')
            json.dump(model_params, f, ensure_ascii=False, indent=4)
            logger.info(f"Model parameters saved succesfully to {model_params_path}")
        except Exception as err:
            logger.error(f"Unexpected error when trying to save trained model\n\tError details: {err=}, {type(err)=}")
            sys.exit(1)
            
    # 5. Make training results report
    if make_report:
        report_dir = ROOT_DIR / ("reports/" + model_name)
        report_dir.mkdir(parents=True, exist_ok=True)
        loss_report_path = report_dir / ("loss.png")
        acc_report_path = report_dir / ("accuracy.png")
        
        logger.info(f"Saving model training reports to {report_dir}")
        try:
            # Plot loss evolution
            plt.plot(history["train_loss"].keys(), history["train_loss"].values(), label="Train")
            plt.plot(history["val_loss"].keys(), history["val_loss"].values(), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Model loss evolution")
            plt.legend(loc="upper right")
            plt.savefig(loss_report_path)
            logger.info(f"Training loss report saved succesfully to {loss_report_path}")
        except Exception as err:
            logger.error(f"Unexpected error when trying to save training loss report\n\tError details: {err=}, {type(err)=}")
            sys.exit(1)
        
        try:
            # Plot accuracy evolution
            plt.clf()
            plt.plot(history["train_acc"].keys(), history["train_acc"].values(), label="Train")
            plt.plot(history["val_acc"].keys(), history["val_acc"].values(), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Model accuracy evolution")
            plt.legend(loc="upper right")
            plt.savefig(acc_report_path)
            logger.info(f"Training accuracy report saved succesfully to {acc_report_path}")
        except Exception as err:
            logger.error(f"Unexpected error when trying to save training accuracy report\n\tError details: {err=}, {type(err)=}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a training pipeline.')
    parser.add_argument('--data-path', required=True, type=str, help='Path to processed data. Please specify path from project root. Usage example: data/processed/mydataset.npz')
    parser.add_argument('--config-path', type=str, help='Path to experiment config file, with model and optimizer parameters. Please specify path from project root. Usage example: experiments/config.yml')
    parser.add_argument('--report', '-r', action="store_true", help='Generate report with train results.')
    parser.add_argument('--persistent', '-p', action="store_true", help='Save trained model for later use.')
    parser.add_argument('--verbose', '-v', action="store_true", help='Log informational messages.')
    args = parser.parse_args()
    
    if not args.data_path.endswith(".npz"):
        parser.error("Expected dataset in --data-path in .npz format")
        
    data_path = ROOT_DIR / args.data_path
    config_path = ROOT_DIR / args.config_path if args.config_path is not None else None
    make_report = args.report
    is_persistent = args.persistent
    
    if not args.verbose:
        logging_level = logging.ERROR
        
    main(data_path, config_path, is_persistent, make_report)
    