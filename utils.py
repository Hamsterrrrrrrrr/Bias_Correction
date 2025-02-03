import random
import numpy as np
import torch

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def calc_spatial_correlation(x, y):
    corr = np.zeros((x.shape[1], x.shape[2]))
    corr.fill(np.nan) 

    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
   
            x_loc = x[:,i,j]
            y_loc = y[:,i,j]
            
            if not (np.all(np.isnan(x_loc)) or np.all(np.isnan(y_loc))):
    
                mask = ~(np.isnan(x_loc) | np.isnan(y_loc))
                if np.sum(mask) > 0:  
                    corr[i,j] = np.corrcoef(x_loc[mask], y_loc[mask])[0,1]
    
    return corr

