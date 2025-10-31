from pathlib import Path
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data import make_dataset
from src.models.models import FCNN
from src.pipeline.pipeline import Pipeline

ROOT_DIR = Path(__file__).parent / '../..' # project root

class TrainPipeline(Pipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.is_persistent = False
        
    def parse_config(self):
        super().parse_config()
        
        assert 'train' in self.config
        
        # parse dataset params
        
        # parse model params
        if 'model' in self.config['train']:
            if 'activation' in self.config['train']['model']:
                cls_name = self.config['train']['model']['activation'].get('type', 'ReLU')
                params = self.config['train']['model']['activation'].get('params', {})

                if not hasattr(nn, cls_name):
                    raise ValueError(f"Unknown torch activation function: {cls_name}")

                cls = getattr(nn, cls_name)
                self.config['train']['model']['activation']['parsed'] = cls(**params)

            
        
        # parse optimizer params
        
        # parse training loop params
        
        # parse persistence params
        if 'persistence' in self.config['train']:
            pass
        
    def get_data(self):
        pass
    
    

def train(model, x, y):
    
    y = y.astype(np.float32)    
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders for batching
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # reshuffle data at every epoch
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch in train_loader:
            X_batch, y_batch = batch

            # Forward pass
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update parameters

            running_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch in val_loader:
                X_batch, y_batch = batch
                predictions = model(X_batch)
                val_loss += loss_fn(predictions, y_batch).item()

                # Calculate accuracy
                predicted_labels = (predictions > 0.5).float().numpy()
                predicted_labels = [np.where(x)[0][0] if len(np.where(x)[0]) else -1 for x in predicted_labels]
                
                y_batch = y_batch.numpy()
                correct += (predicted_labels == y_batch).sum()
                total += len(y_batch)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
    return model

def main(config_path: Path, data_path: Path, raw_data: bool):
    
    pipe = TrainPipeline(config_path)
    
    # 1. Load data and process it if needed
    x = y = None
    if raw_data:
        dataset = make_dataset.load_data(data_path)
        x, y = zip(*dataset)
        make_dataset.save_dataset(dataset) # save processed data for reuse
    else:
        npzfile = np.load(data_path)
        x = npzfile['x']
        y = npzfile['y']
        
    # 2. Build model
    input_size = 40
    hidden_sizes = [64, 128, 128, 64]
    output_size = 10
    # model = FCNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size).to(torch.device("cpu"))
    model = FCNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    print(model)
    
    # 3. Train model
    trained_model = train(model, x, y)
    
    # 4. Save trained model
    torch.save(trained_model.state_dict(), ROOT_DIR / 'models/mymodel.pth')
    params = dict(
        input_size = input_size,
        hidden_sizes = hidden_sizes,
        output_size = output_size
    )
    with open(ROOT_DIR / 'models/mymodel_params.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a training pipeline.')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file. Please specify path from project root. Usage example: config.yml')
    parser.add_argument('--raw-data-path', type=str, help='Path to raw data. Please specify path from project root. Usage example: data/raw/')
    parser.add_argument('--processed-data-path', type=str, help='Path to processed data. Please specify path from project root. Usage example: data/processed/mydataset.npz')
    parser.add_argument('--verbose', '-v', action="store_true", help='Log informational messages.')
    args = parser.parse_args()
    
    data_path = None
    raw_data = False    
    if args.raw_data_path is None and args.processed_data_path is None:
        parser.error("At least one of --raw-data-path and --processed-data-path is required.")
    elif args.raw_data_path is not None and args.processed_data_path is None:
        # data is raw
        raw_data = True
        data_path = ROOT_DIR / args.raw_data_path
    elif args.raw_data_path is None and args.processed_data_path is not None:
        # data is processed
        data_path = ROOT_DIR / args.processed_data_path
    else:
        parser.error("Please specify only one of --raw-data-path and --processed-data-path.")
        
    config_path = ROOT_DIR / args.config
    
    logging_level = logging.DEBUG
    if not args.verbose:
        logging_level = logging.CRITICAL
        
    main(config_path, data_path, raw_data)
    