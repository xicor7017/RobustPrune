import os
import time
import torch
import random
import pathlib
import torch.nn as nn
from hydra.utils import get_original_cwd

from dataset import get_dataloaders

from model import MLP
from transformer_model import TransformerPrunableEncoder

def overfit(cfg):
    if cfg.model.type.lower() == "mlp":
        model   = MLP(hidden_dims=cfg.model.hidden_dims)    
    else:
        model    = TransformerPrunableEncoder(
            input_dim=2,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_feedforward=32,
            num_classes=5,
            max_seq_len=1
        )   

    train_data, test_data, miss_data, all_train, _ = get_dataloaders(
                                                                        cfg.dataset.train_samples,
                                                                        cfg.dataset.test_samples,
                                                                        cfg.dataset.miss_samples,
                                                                        batch_size=cfg.model.batch_size, 
                                                                        biased=cfg.dataset.biased,
                                                                        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    print()
    print("Dataset type: {}".format("Biased" if cfg.dataset.biased else "Unbiased / Uniform"), end="\n\n")
    time.sleep(2)

    for epoch in range(10000):
        # Test
        with torch.no_grad():
            test_accuracy = 0
            for data, label in test_data:
                label          = label
                pred_probs     = model(data)
                pred           = torch.argmax(pred_probs, -1)
                test_accuracy += (pred == label).float().mean().item()
            test_accuracy = test_accuracy

            train_accuracy = 0
            for data, label in train_data:
                label           = label
                pred_probs      = model(data)
                pred            = torch.argmax(pred_probs, -1)
                train_accuracy += (pred == label).float().mean().item()
            train_accuracy = train_accuracy

            miss_accuracy = 0
            for data, label in miss_data:
                label          = label
                pred_probs     = model(data)
                pred           = torch.argmax(pred_probs, -1)
                miss_accuracy += (pred == label).float().mean().item()
            miss_accuracy = miss_accuracy

        test_accuracy  = round(test_accuracy, 3)
        miss_accuracy  = round(miss_accuracy, 3)
        train_accuracy = round(train_accuracy, 3)
        print(f"\033[F\033[KEpoch: {epoch} | Test: {test_accuracy} | Train: {train_accuracy} | Miss: {miss_accuracy}", end=" ")
        if train_accuracy == 1.0 and miss_accuracy == 1.0: # Overfitting successful
            break

        # Train
        for _ in range(100):
            for data, label in all_train:
                data = data
                label = label
                pred_probs = model(data)
                loss = criterion(pred_probs, label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss  = round(loss.item(), 3)
        print(f" | Loss: {loss}")
        
    # Save the model
    cwd = get_original_cwd()
    if not os.path.exists(f"{cwd}/saved_models"):
        os.makedirs(f"{cwd}/saved_models")

    datatype = "biased" if cfg.dataset.biased else "unbiased"
    torch.save(model.state_dict(), f"{cwd}/saved_models/overfitted_{datatype}.pt")
    print("Finished")

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    overfit()