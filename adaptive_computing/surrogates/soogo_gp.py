from soogo.model import GaussianProcess
from sklearn.gaussian_process.kernels import DotProduct, RBF
from sklearn.preprocessing import StandardScaler
from adaptive_computing.surrogates.base import SurrogateModelBase
import numpy as np
import warnings

class SOOGO_GP(SurrogateModelBase):
    """
    A wrapper class for using soogo Gaussian Process as surrogate model.
    
    Attributes:
        surrogate_model (list): List of soogo GaussianProcess models for each fidelity level.
        scaler (StandardScaler): Scaler for normalizing input data.
        
    Methods:
        __init__(dataset, soogo_kwargs=None): Initializes the SOOGO_GP.
        train(x_data, y_data): Trains the surrogate models.
        predict_values(x_data, fidelity_level=-1): Predicts values using the surrogate model.
        predict_variances(x_data, fidelity_level=-1): Predicts variances using the surrogate model.
    """
    
    def __init__(self, dataset, soogo_kwargs=None):
        """
        Initializes the SOOGO_GP with the dataset and optional soogo-specific keyword arguments.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            soogo_kwargs (dict, optional): Additional keyword arguments for configuring soogo models. Defaults to None.
        """
        super().__init__(dataset)

        if soogo_kwargs is None:
            soogo_kwargs = {}
        
        # Simplified approach: use fixed alpha to avoid expensive tuning
        self.best_alpha = soogo_kwargs.get('alpha', 1e-3)  # Default alpha
        self.tune_alpha = soogo_kwargs.get('tune_alpha', False)  # Disable alpha tuning by default
        self.alpha_candidates = soogo_kwargs.get('alpha_candidates', [1e-4, 1e-3, 1e-2]) if self.tune_alpha else [self.best_alpha]
        
        self.surrogate_model = []
        self.scaler = StandardScaler()
        
        # Note: soogo doesn't natively support multifidelity, so we treat each fidelity as separate model
        for i_fidelity in range(self.n_fidelity):
            kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed") + RBF(length_scale=1.0)
            model = GaussianProcess(kernel=kernel, alpha=self.best_alpha)
            self.surrogate_model.append(model)
        
        self.untrained = True
        self.is_scaled = False

    def _tune_alpha(self, x_scaled, y_data, n_splits=3):
        """
        Tune the alpha (noise) parameter using cross-validation.
        Simplified version with timeout protection.
        
        Args:
            x_scaled (np.ndarray): Scaled input data.
            y_data (np.ndarray): Output data.
            n_splits (int): Number of cross-validation folds.
            
        Returns:
            float: Best alpha value.
        """
        from sklearn.model_selection import KFold
        import signal
        
        # Timeout protection
        def timeout_handler(signum, frame):
            raise TimeoutError("Alpha tuning timeout")
        
        kf = KFold(n_splits=min(n_splits, max(2, len(x_scaled)//3)), shuffle=True, random_state=42)
        best_alpha, best_rmse = self.best_alpha, np.inf

        for alpha in self.alpha_candidates:
            try:
                # Set timeout for each alpha evaluation
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 10 second timeout
                
                fold_rmses = []
                for train_idx, val_idx in kf.split(x_scaled):
                    kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed") + RBF(length_scale=1.0)
                    model = GaussianProcess(kernel=kernel, alpha=alpha)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.update(x_scaled[train_idx], y_data[train_idx].ravel())
                        y_pred = model(x_scaled[val_idx])
                        fold_rmses.append(np.sqrt(np.mean((y_data[val_idx].ravel() - y_pred) ** 2)))
                    except Exception:
                        fold_rmses.append(np.inf)
                        break  # Skip this alpha if any fold fails

                if len(fold_rmses) == n_splits:  # Only consider if all folds succeeded
                    mean_rmse = float(np.mean(fold_rmses))
                    if mean_rmse < best_rmse:
                        best_rmse, best_alpha = mean_rmse, alpha
                
                signal.alarm(0)  # Cancel timeout
                
            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel timeout
                print(f"Skipping alpha {alpha}: {e}")
                continue

        return best_alpha

    def train(self, x_data, y_data):
        """
        Trains the surrogate models with the provided data.
        
        Args:
            x_data (list): List of input data arrays for each fidelity level.
                Each element of list has shape (N samples, N input dimension)
            y_data (list): List of output data arrays for each fidelity level.
                Each element of list has shape (N samples, N output dimension)
        """
        x_data, y_data = self._validate_data(x_data, y_data)

        for i_fidelity in range(self.n_fidelity):
            if x_data[i_fidelity].shape[0] == 0:
                continue
                
            # Scale the input data for the first fidelity level (assuming single fidelity)
            if i_fidelity == 0 and not self.is_scaled:
                x_scaled = self.scaler.fit_transform(x_data[i_fidelity])
                self.is_scaled = True
            else:
                x_scaled = self.scaler.transform(x_data[i_fidelity])
            
            # Optional alpha tuning - only if enabled and we have enough data
            if self.tune_alpha and x_scaled.shape[0] > 10:
                try:
                    self.best_alpha = self._tune_alpha(x_scaled, y_data[i_fidelity])
                except Exception as e:
                    print(f"Alpha tuning failed: {e}. Using default alpha = {self.best_alpha}")
            
            # Create fresh model with chosen alpha
            kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed") + RBF(length_scale=1.0)
            self.surrogate_model[i_fidelity] = GaussianProcess(kernel=kernel, alpha=self.best_alpha)
            
            # Train the model with error handling
            try:
                self.surrogate_model[i_fidelity].update(x_scaled, y_data[i_fidelity].ravel())
            except Exception as e:
                print(f"GP training failed: {e}. Trying with higher alpha.")
                # Retry with higher noise level
                kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed") + RBF(length_scale=1.0)
                self.surrogate_model[i_fidelity] = GaussianProcess(kernel=kernel, alpha=max(self.best_alpha * 10, 1e-2))
                self.surrogate_model[i_fidelity].update(x_scaled, y_data[i_fidelity].ravel())
        
        self.untrained = False

    def predict_values(self, x_data, fidelity_level=-1):
        """
        Predicts values using the surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted values.
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model values, but user never called train() since initializing the surrogate.")
        
        # Ensure x_data is 2D
        x_data = np.atleast_2d(x_data)
        
        # Scale the input data
        if not self.is_scaled:
            raise Exception("Model has not been trained yet, cannot scale input data.")
        x_scaled = self.scaler.transform(x_data)
        
        # Get predictions
        y_pred = self.surrogate_model[fidelity_level](x_scaled)
        
        # Reshape to match expected output format (N_samples, 1)
        return y_pred.reshape(-1, 1)

    def predict_variances(self, x_data, fidelity_level=-1):
        """
        Predicts variances using the surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted variances.
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model variances, but user never called train() since initializing the surrogate.")
        
        # Ensure x_data is 2D
        x_data = np.atleast_2d(x_data)
        
        # Scale the input data
        if not self.is_scaled:
            raise Exception("Model has not been trained yet, cannot scale input data.")
        x_scaled = self.scaler.transform(x_data)
        
        try:
            # Get predictions with variance
            _, std_pred = self.surrogate_model[fidelity_level].model.predict(x_scaled, return_std=True)
            
            # Ensure non-negative variances and avoid numerical issues
            variances = np.maximum(std_pred ** 2, 1e-10)
            
            # Return variance in expected format (N_samples, 1)
            return variances.reshape(-1, 1)
        except Exception as e:
            print(f"Warning: Variance prediction failed: {e}")
            # Return small positive variances as fallback  
            return np.full((x_data.shape[0], 1), 1e-6)
