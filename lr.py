from scipy.stats import linregress   
import numpy as np   

def perform_linear_regression(X_train_val, y_train_val, X_test, y_test):

    X_mean, X_std = X_train_val.mean(axis=0), X_train_val.std(axis=0)
    y_mean, y_std = y_train_val.mean(axis=0), y_train_val.std(axis=0)
    
    X_train_val_norm = (X_train_val - X_mean) / X_std
    y_train_val_norm = (y_train_val - y_mean) / y_std
    
    def ols_regression(X, y):

        slope, intercept, r_value, p_value, std_err = linregress(X, y)
        return slope, intercept, r_value, p_value, std_err
    
    n_lat, n_lon = X_train_val.shape[1:]
    
    slopes = np.zeros((n_lat, n_lon))
    intercepts = np.zeros((n_lat, n_lon))
    r_values = np.zeros((n_lat, n_lon))
    p_values = np.zeros((n_lat, n_lon))
    std_errs = np.zeros((n_lat, n_lon))
    
    # Perform regression for each spatial point
    for i in range(n_lat):
        for j in range(n_lon):
            X_point = X_train_val_norm[:, i, j]
            y_point = y_train_val_norm[:, i, j]
            
            if np.all(np.isnan(X_point)) or np.all(np.isnan(y_point)) or \
               np.std(X_point) == 0 or np.std(y_point) == 0:
                slopes[i, j] = np.nan
                intercepts[i, j] = np.nan
                r_values[i, j] = np.nan
                p_values[i, j] = np.nan
                std_errs[i, j] = np.nan
                continue
                
            slope, intercept, r_value, p_value, std_err = ols_regression(X_point, y_point)
            
            slopes[i, j] = slope
            intercepts[i, j] = intercept
            r_values[i, j] = r_value
            p_values[i, j] = p_value
            std_errs[i, j] = std_err
    
    X_test_norm = (X_test - X_mean) / X_std
    
    y_pred = np.zeros_like(y_test)
    
    # Make predictions for each spatial point
    for i in range(n_lat):
        for j in range(n_lon):
            if not np.isnan(slopes[i, j]):

                y_pred[:, i, j] = slopes[i, j] * X_test_norm[:, i, j] + intercepts[i, j]
                y_pred[:, i, j] = y_pred[:, i, j] * y_std[i, j] + y_mean[i, j]
                
            else:
                y_pred[:, i, j] = y_mean[i, j]
    
    return y_pred


