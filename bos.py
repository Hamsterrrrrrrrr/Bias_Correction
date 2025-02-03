import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import time


class BayesianEnsembleWeighting:
    def __init__(self, n_models):
        self.n_models = n_models
        
    def _normalize_weights(self, weights):
        return weights / np.sum(weights)
    
    def _compute_masked_mse(self, y_true, y_pred):
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not mask.any():
            return np.inf
        return mean_squared_error(y_true[mask], y_pred[mask])
        
    def optimize_weights(self, y_true, preds_dict):
        # Start timing
        start_time = time.time()
        
        def objective(weights):
            weights = self._normalize_weights(weights)
            ensemble_pred = np.zeros_like(y_true)
            for i, (model_name, _) in enumerate(preds_dict.items()):
                ensemble_pred += weights[i] * preds_dict[model_name]
            return self._compute_masked_mse(y_true, ensemble_pred)
            
        kernel = ConstantKernel(1) * RBF(length_scale=1)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        n_initial = 10
        weights_samples = np.random.dirichlet(np.ones(self.n_models), n_initial)
        scores = [objective(w) for w in weights_samples]
        
        for i in range(40):
            gpr.fit(weights_samples, scores)
            candidates = np.random.dirichlet(np.ones(self.n_models), 100)
            mean, std = gpr.predict(candidates, return_std=True)
            ucb = mean - 2 * std
            next_weights = candidates[np.argmin(ucb)]
            score = objective(next_weights)
            weights_samples = np.vstack((weights_samples, next_weights))
            scores.append(score)
            
        best_weights = weights_samples[np.argmin(scores)]
        
        # Calculate optimization time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        return dict(zip(preds_dict.keys(), best_weights)), total_time

