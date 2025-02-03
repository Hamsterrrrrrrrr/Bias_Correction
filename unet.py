import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.enc_conv1 = self.conv_block(in_channels=1, out_channels=32)
        self.enc_conv2 = self.conv_block(in_channels=32, out_channels=64)
        self.enc_conv3 = self.conv_block(in_channels=64, out_channels=128)
        self.enc_conv4 = self.conv_block(in_channels=128, out_channels=256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck_conv = self.conv_block(in_channels=256, out_channels=512)
        
        self.upconv4 = self.conv_transpose_block(in_channels=512, out_channels=256)
        self.dec_conv4 = self.conv_block(in_channels=512, out_channels=256)
        
        self.upconv3 = self.conv_transpose_block(in_channels=256, out_channels=128)
        self.dec_conv3 = self.conv_block(in_channels=256, out_channels=128)
        
        self.upconv2 = self.conv_transpose_block(in_channels=128, out_channels=64)
        self.dec_conv2 = self.conv_block(in_channels=128, out_channels=64)
        
        self.upconv1 = self.conv_transpose_block(in_channels=64, out_channels=32)
        self.dec_conv1 = self.conv_block(in_channels=64, out_channels=32)
        
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def conv_transpose_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):

        x1 = self.enc_conv1(x)   
        x2 = self.pool(x1)
        x2 = self.enc_conv2(x2)    
        x3 = self.pool(x2)
        x3 = self.enc_conv3(x3)     
        x4 = self.pool(x3)
        x4 = self.enc_conv4(x4)     
        x5 = self.pool(x4)
        
        x5 = self.bottleneck_conv(x5)  
        
        u4 = self.upconv4(x5)
        u4 = F.interpolate(u4, size=x4.size()[2:], mode='bilinear', align_corners=True)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec_conv4(u4)
        
        u3 = self.upconv3(u4)
        u3 = F.interpolate(u3, size=x3.size()[2:], mode='bilinear', align_corners=True)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec_conv3(u3)
        
        u2 = self.upconv2(u3)
        u2 = F.interpolate(u2, size=x2.size()[2:], mode='bilinear', align_corners=True)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec_conv2(u2)
        
        u1 = self.upconv1(u2)
        u1 = F.interpolate(u1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec_conv1(u1)
        
        out = self.final_conv(u1)
        return out


def train_cnn(model, train_loader, val_loader,
              criterion, optimizer, device,
              save_path='/home/jovyan/SSH/B_data/updated_dm/test3/model.pth', 
              n_epochs=2000, patience=50): 


    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    def print_number_of_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable Parameters: {trainable_params}")

    print_number_of_parameters(model)
    model.to(device)

    total_start_time = time.time()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        logger.info(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.6f}")
    else:
        start_epoch = 0

    def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, patience_counter, save_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, save_path)
        logger.info(f'Model saved at epoch {epoch+1}')

    try:
        for epoch in range(start_epoch, n_epochs):

            # Training phase
            start_time = time.time()
            model.train()
            train_running_loss = 0.0

            for batch_x, batch_y in train_loader:

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_running_loss += loss.item() * batch_x.size(0)

            epoch_loss = train_running_loss / len(train_loader.dataset)

            # Validation phase
            model.eval()   
            val_running_loss = 0.0

            with torch.no_grad():  
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_running_loss += loss.item() * batch_x.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            scheduler.step(val_loss)

            end_time = time.time()
            epoch_duration = end_time - start_time
            if torch.cuda.is_available() and 'cuda' in device.type:
                peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
            else:
                peak_memory = 0  

            logger.info(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.6e}, Val Loss: {val_loss:.6e}, '
                        f'Epoch Time: {epoch_duration:.2f}s, Peak Memory: {peak_memory:.2f} MB')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, patience_counter, save_path)
            else:
                patience_counter += 1
                logger.info(f'Patience counter: {patience_counter}/{patience}')

            if patience_counter >= patience:
                logger.info('Early stopping triggered')
                break

    except KeyboardInterrupt:
        logger.info('Training interrupted by user.')
        total_time = time.time() - total_start_time
        return total_time

    total_time = time.time() - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    logger.info(f'Training complete. Total time: {hours}h {minutes}m {seconds}s')
    
    return total_time



class NaNMSELoss(nn.Module):
    def __init__(self):
        super(NaNMSELoss, self).__init__()

    def forward(self, output, target):
        mask = torch.isfinite(target)
        
        if not mask.any():
            
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        loss = torch.mean((output[mask] - target[mask]) ** 2)
        return loss


def evaluate_cnn(model, device, test_loader, checkpoint_path):

    model = model.to(device)

    model.eval()  

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model parameters from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    predictions = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:  
            batch_x = batch_x.to(device)
            y_pred = model(batch_x)
            predictions.append(y_pred.cpu())
            
    prediction = torch.cat(predictions, dim=0).squeeze(1).numpy() 

    return prediction



