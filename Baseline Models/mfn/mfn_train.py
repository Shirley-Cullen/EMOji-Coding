from graph_mfn import GraphMFN3
from mfn import MFN3
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from mosei_dataset import MOSEIDataset
import os
from datetime import datetime
import tqdm
import random

TARGET_INDEX = {
    "sentiment": 0,
    "happy": 1,
    "sad": 2,
    "anger": 3,
    "surprise": 4,
    "disgust": 5,
    "fear": 6,
}

def r_squared(y_pred, y_true):
    """
    Compute the coefficient of determination (r^2 score)
    """
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

def mae(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Absolute Error (MAE) between predictions and labels.
    
    Args:
        preds (torch.Tensor): 1D tensor of predictions.
        labels (torch.Tensor): 1D tensor of ground truth values.
        
    Returns:
        torch.Tensor: A scalar tensor containing the MAE.
    """
    # Ensure both tensors are 1D
    if preds.ndim != 1 or labels.ndim != 1:
        raise ValueError("Both preds and labels must be 1D tensors.")
    return torch.mean(torch.abs(preds - labels))

def pearson_corr(pred, label):
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    corr_matrix = np.corrcoef(pred, label)
    # Extract Pearson's r value (off-diagonal element)
    return corr_matrix[0, 1]

def class_from_string(class_path: str, *args, **kwargs):
    # Expecting class_path format "module_name.ClassName", returns the class object, not initiated object
    module_name, class_name = class_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls

def general_epoch(model, dataloader, target, loss_fn, optimizer=None, device='cpu'):
    """
    Performs one epoch of training or evaluation for regression.
    
    Args:
        model (torch.nn.Module): the model to train/evaluate.
        dataloader (DataLoader): DataLoader for the current dataset (train or validation).
        loss_fn (callable): the loss function.
        optimizer (torch.optim.Optimizer or None): optimizer. If None, evaluation mode.
        device (str): device to run the computation on.
    
    Returns:
        epoch_loss (float): average loss over the epoch.
        epoch_r (float): r-squared value over the epoch.
    """
    if optimizer is not None:
        model.train()  # training mode
    else:
        model.eval()   # evaluation mode
    model = model.to(dtype=torch.float32)

    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0
    target_index = TARGET_INDEX[target]
    gradient_explosion_count = 0
    num_nans = 0

    for *inputs, labels in tqdm.tqdm(dataloader): 
        inputs = [inputs[1], inputs[3], inputs[4]]
        inputs = torch.cat(inputs, dim=2).to(device, dtype=torch.float32)
        labels = labels[:, target_index].to(device, dtype=torch.float32) # regression targets

        outputs = model(inputs)
        num_nans += torch.isnan(outputs).sum().item()
        loss = loss_fn(outputs, labels.unsqueeze(1)) / inputs.shape[0]  # assuming model outputs [batch, 1]
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            except RuntimeError:
                gradient_explosion_count += 1
                continue
            optimizer.step()
        
        if not torch.isnan(loss).item():
            running_loss += loss.item() * inputs.size(0)
        all_preds.append(outputs.squeeze())
        all_labels.append(labels)
        total_samples += inputs.size(0)
    
    if optimizer is not None:
        print(f"Number of batches with gradient explosion in this epoch: {gradient_explosion_count}/{len(dataloader)}")
    else:
        print(f"Number of NaN prediction outputs during test/validation: {num_nans}/{len(dataloader.dataset)}")
    epoch_loss = running_loss / total_samples
    all_preds = torch.nan_to_num(torch.cat(all_preds), nan=0.0)
    all_labels = torch.cat(all_labels)
    epoch_r = pearson_corr(all_preds, all_labels)
    epoch_mae = mae(all_preds, all_labels)
    epoch_r2 = r_squared(all_preds, all_labels)
    
    return epoch_loss, epoch_r, epoch_mae, epoch_r2


def train_model(model, train_loader, valid_loader, loss_fn, optimizer, workdir, target,
                device='cpu', num_epochs=25, patience=5):
    """
    Train the regression model with mini-batch training, per-epoch validation, and early stopping.
    
    Args:
        model (torch.nn.Module): the model to train.
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
        loss_fn (callable): loss function.
        optimizer (torch.optim.Optimizer): optimizer.
        device (str): device to use.
        num_epochs (int): maximum number of epochs.
        patience (int): epochs to wait before early stopping.
    
    Returns:
        model: the trained model (best state).
    """
    best_val_loss = float('inf')
    epochs_without_improve = 0
    train_loss_trace, train_r_trace, val_loss_trace, val_r_trace = [], [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_loss, train_r, train_mae, train_r2 = general_epoch(model, train_loader, target, loss_fn, optimizer, device)
        train_loss_trace.append(train_loss)
        train_r_trace.append(train_r)
        print(f"Train Loss: {train_loss:.4f} r: {train_r:.4f} MAE: {train_mae:.4f} R^2: {train_r2}")
        
        # Validation phase
        val_loss, val_r, val_mae, val_r2 = general_epoch(model, valid_loader, target, loss_fn, optimizer=None, device=device)
        val_loss_trace.append(val_loss)
        val_r_trace.append(val_r)
        print(f"Val   Loss: {val_loss:.4f} r: {val_r:.4f} MAE: {val_mae:.4f} R^2: {val_r2}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            print("Validation loss improved, saving model...")
        else:
            epochs_without_improve += 1
            print(f"No improvement for {epochs_without_improve} epoch(s).")
        torch.save(model.state_dict(), f"{workdir}/epoch{epoch}_valloss{val_loss:.2f}_valmae{val_mae:.2f}.pt")
        
        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

        print("-" * 30)

    return model


# Example usage for regression:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-checkpoint", type=str, default=None, help="If supplied, will run test dataset on this checkpoint")
    parser.add_argument("--target", type=str, default="sentiment", help="Prediction target (sentiment, fear, happy, etc.)")

    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to continue training on")
    parser.add_argument("--model-class", type=str, required=True, help="e.g. mfn.MFN3")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max-seqlen", type=int, default=50, help="Max sequence length fot LSTM network")
    parser.add_argument("--check-nan", action="store_true", help="Enable check for NaN in forward and backward")
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    print(args)
    assert args.target in TARGET_INDEX
    data_path = "tensors_short.pkl" if args.dry_run else "tensors.pkl"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.test_checkpoint is not None:
        model = class_from_string(args.model_class)()
        checkpoint = torch.load(args.test_checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        test_loss, test_r, test_mae, test_r2 = general_epoch(
            model,
            DataLoader(MOSEIDataset(data_path, "test"), batch_size=64, shuffle=False),
            args.target,
            nn.L1Loss(),
            optimizer=None,
            device=device
        )
        print(f"Test   Loss: {test_loss:.4f} r: {test_r:.4f} MAE: {test_mae:.4f} R^2: {test_r2:.4f}")
        exit()

    print("Loading datasets ...")
    train_dataset = MOSEIDataset(data_path, "train")
    valid_dataset = MOSEIDataset(data_path, "val")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    if args.checkpoint is None:
        model = class_from_string(args.model_class)(max_seqlen=args.max_seqlen)
    else:
        model = class_from_string(args.model_class)(max_seqlen=args.max_seqlen)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
    model.to(device)

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    workdir = f"workdir_{model.__class__.__name__}_{args.target}_lr{args.lr}_wd{args.weight_decay}_{datetime.now().strftime('%m%d%H%M%S')}"
    os.mkdir(workdir)

    torch.autograd.set_detect_anomaly(args.check_nan, check_nan=args.check_nan)
    train_model(model, train_loader, valid_loader, loss_fn, optimizer, workdir, args.target,
                device=device, num_epochs=args.epochs, patience=5)
    
    print("Training session finished. Exiting ...")
