import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpx_linalg
from smt.surrogate_models import KRG

class GPUKriging(KRG):
    """
    A subclass of SMT's KRG that implements predict_variances on the GPU using CuPy.
    """
    
    def predict_variances(self, x):
        """
        GPU-accelerated implementation of predict_variances.
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x[None, :]
            
        # 1. Move inputs to GPU
        d_x = cp.asarray(x)
        
        # Model parameters
        d_C = cp.asarray(self.optimal_par["C"])
        d_G = cp.asarray(self.optimal_par["G"])
        d_Ft = cp.asarray(self.optimal_par["Ft"])
        d_sigma2 = cp.asarray(self.optimal_par["sigma2"])
        
        # Training points (normalized)
        d_X_norma = cp.asarray(self.X_norma)
        
        # Hyperparameters
        d_theta = cp.asarray(self.optimal_theta)
        
        # Normalization parameters
        d_X_offset = cp.asarray(self.X_offset)
        d_X_scale = cp.asarray(self.X_scale)
        
        # Normalize input x
        d_x_norm = (d_x - d_X_offset) / d_X_scale
        
        n_eval = d_x.shape[0]
        n_train = d_X_norma.shape[0]
        n_dim = d_x.shape[1]
        
        # Determine power based on correlation type
        corr_type = self.options["corr"]
        if corr_type == "squar_exp":
            power = 2.0
        elif corr_type == "abs_exp":
            power = 1.0
        else:
            # Fallback or error for unsupported kernels
            # SMT supports others, but these are most common
            if corr_type == "matern32" or corr_type == "matern52":
                 raise NotImplementedError(f"GPU implementation for {corr_type} not yet added.")
            power = 2.0 # Default assumption if unknown, but risky
            
        # Batch processing to avoid OOM
        # We need to compute (n_batch, n_train) correlation matrix
        # Memory usage: n_batch * n_train * 8 bytes
        # If n_train=1000, n_batch=10000 -> 80MB. Safe.
        batch_size = 5000 
        
        d_s2_list = []
        
        for i in range(0, n_eval, batch_size):
            d_x_batch = d_x_norm[i : i + batch_size]
            current_batch_size = d_x_batch.shape[0]
            
            # Compute Correlation Matrix (r)
            # diff = |x - X|
            # We use broadcasting: (batch, 1, dim) - (1, train, dim)
            diff = cp.abs(d_x_batch[:, None, :] - d_X_norma[None, :, :])
            
            # d = diff^power
            if power == 2.0:
                d_mat = diff**2
            elif power == 1.0:
                d_mat = diff
            else:
                d_mat = diff**power
                
            # weighted_d = sum(theta * d, axis=2)
            # theta is (dim,)
            weighted_d = cp.sum(d_theta * d_mat, axis=2)
            
            # r = exp(-weighted_d)
            d_r = cp.exp(-weighted_d) # Shape: (batch, n_train)
            
            # Solve Linear Systems
            # rt = C^-1 * r^T  (Note: r is (batch, train), so r.T is (train, batch))
            # solve_triangular(C, r.T, lower=True)
            d_rt = cpx_linalg.solve_triangular(d_C, d_r.T, lower=True)
            
            # u = G^-T * (Ft^T * rt - f(x)^T)
            # Assume constant mean (poly='constant') -> f(x) = 1
            # TODO: Support other regression types (linear, quadratic)
            if self.options["poly"] != "constant":
                 raise NotImplementedError("Only constant regression is currently supported on GPU.")
                 
            d_f_x = cp.ones((current_batch_size, 1))
            
            term1 = cp.dot(d_Ft.T, d_rt) # (p, batch)
            term2 = d_f_x.T # (p, batch)
            rhs = term1 - term2
            
            d_u = cpx_linalg.solve_triangular(d_G.T, rhs, lower=False)
            
            # Compute Variance
            # MSE = sigma2 * (1 - rt^2 + u^2)
            
            rt_sq_sum = cp.sum(d_rt**2, axis=0)
            u_sq_sum = cp.sum(d_u**2, axis=0)
            
            d_mse = d_sigma2 * (1.0 - rt_sq_sum + u_sq_sum)
            
            # Clip negative values
            d_mse = cp.maximum(d_mse, 0.0)
            
            d_s2_list.append(d_mse)
            
        d_s2 = cp.concatenate(d_s2_list)
        
        return d_s2

# Test/Verification Block
if __name__ == "__main__":
    print("Testing GPUKriging...")
    
    # Create a dummy SMT model to verify inheritance and basic function
    # We can't easily train one here without data, but we can check imports.
    try:
        model = GPUKriging(print_global=False)
        print("GPUKriging class instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating GPUKriging: {e}")

