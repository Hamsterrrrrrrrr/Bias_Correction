import torch
import torch.nn as nn
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        
        self.layer1 = nn.Linear(1, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.layer5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.layer6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.layer7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.layer8 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):

        x1 = self.relu(self.bn1(self.layer1(x)))
        x2 = self.relu(self.bn2(self.layer2(x1)))
        x3 = self.relu(self.bn3(self.layer3(x2)))
        
        mid = self.relu(self.bn4(self.layer4(x3)))
        
        x = self.relu(self.bn5(self.layer5(mid)))
        x = x + x3
        x = self.relu(self.bn6(self.layer6(x)))
        x = x + x2
        x = self.relu(self.bn7(self.layer7(x)))
        x = x + x1
        x = self.layer8(x)  
        
        return x

def train_ann(model, train_loader, val_loader, criterion, optimizer, num_epochs, checkpoint_path, patience=100, scheduler_patience=5):
    best_val_loss = float('inf')
    counter = 0
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < 2*best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                break
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f'Total training time: {hours}h {minutes}m {seconds}s')
    
    return total_time


def evaluate_ann(model, test_loader):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.extend(outputs.numpy())
    
    return np.array(all_predictions)

