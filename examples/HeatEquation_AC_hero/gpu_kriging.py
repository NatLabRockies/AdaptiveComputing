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
        Wrapper to catch and print exceptions that C++ AMReX might swallow.
        """
        try:
            return self._predict_variances_internal(x)
        except Exception:
            import traceback
            import sys
            sys.stderr.write("Exception caught in GPUKriging.predict_variances:\n")
            traceback.print_exc()
            sys.stderr.flush()
            raise

    def _predict_variances_internal(self, x):
        """
        Internal implementation.
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
            
            if corr_type == "squar_exp":
                # r = exp( - sum( theta * (x-X)^2 ) )
                d_mat = diff**2
                weighted_d = cp.sum(d_theta * d_mat, axis=2)
                d_r = cp.exp(-weighted_d)
                
            elif corr_type == "abs_exp":
                # r = exp( - sum( theta * |x-X| ) )
                d_mat = diff
                weighted_d = cp.sum(d_theta * d_mat, axis=2)
                d_r = cp.exp(-weighted_d)
                
            elif corr_type == "matern32":
                # k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
                # where r = theta * |x-X|
                
                ll = d_theta * diff # (batch, train, dim)
                sqrt3 = cp.sqrt(3.0)
                
                # term1 = prod(1 + sqrt(3)*ll, axis=2)
                term1 = cp.prod(1.0 + sqrt3 * ll, axis=2)
                
                # term2 = exp(-sqrt(3) * sum(ll, axis=2))
                term2 = cp.exp(-sqrt3 * cp.sum(ll, axis=2))
                
                d_r = term1 * term2
                
            elif corr_type == "matern52":
                # k(r) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
                # Product of 1D kernels
                
                ll = d_theta * diff
                sqrt5 = cp.sqrt(5.0)
                
                # term1 = prod(1 + sqrt(5)*ll + 5/3*ll^2, axis=2)
                poly_term = 1.0 + sqrt5 * ll + (5.0 / 3.0) * (ll**2)
                term1 = cp.prod(poly_term, axis=2)
                
                # term2 = exp(-sqrt(5) * sum(ll, axis=2))
                term2 = cp.exp(-sqrt5 * cp.sum(ll, axis=2))
                
                d_r = term1 * term2
                
            elif corr_type == "act_exp":
                # r = exp( - 1/2 * sum( (theta * d)^2 ) )
                
                # Check if theta length is multiple of dim
                if d_theta.size % n_dim != 0:
                     raise ValueError("ActExp: theta length must be multiple of input dimension")
                
                n_small = d_theta.size // n_dim
                d_A_proj = d_theta.reshape(n_small, n_dim).T # (dim, n_small)
                
                # We need to compute d.dot(A) for every pair (x, X)
                # diff is (batch, train, dim)
                # We want result (batch, train, n_small)
                # result[b, t, s] = sum_k( diff[b, t, k] * A[k, s] )
                
                d_diff_proj = cp.tensordot(diff, d_A_proj, axes=(2, 0)) # (batch, train, n_small)
                
                # r = exp(-0.5 * sum(d_diff_proj^2, axis=2))
                weighted_sum = cp.sum(d_diff_proj**2, axis=2)
                d_r = cp.exp(-0.5 * weighted_sum)

            else:
                 raise NotImplementedError(f"GPU implementation for correlation '{corr_type}' not yet added.")
            
            # Solve Linear Systems
            # rt = C^-1 * r^T  (Note: r is (batch, train), so r.T is (train, batch))
            # solve_triangular(C, r.T, lower=True)
            d_rt = cpx_linalg.solve_triangular(d_C, d_r.T, lower=True)
            
            # u = G^-T * (Ft^T * rt - f(x)^T)
            # Regression matrix f(x)
            poly_type = self.options["poly"]
            
            if poly_type == "constant":
                d_f_x = cp.ones((current_batch_size, 1))
            elif poly_type == "linear":
                # f(x) = [1, x_1, ..., x_n]
                d_f_x = cp.hstack([cp.ones((current_batch_size, 1)), d_x_batch])
            elif poly_type == "quadratic":
                # f(x) = [1, x_1, ..., x_n, x_1*x_1, x_1*x_2, ...]
                d_f_x = cp.hstack([cp.ones((current_batch_size, 1)), d_x_batch])
                for k in range(n_dim):
                    # x[:, k, newaxis] * x[:, k:]
                    # (batch, 1) * (batch, dim-k) -> (batch, dim-k)
                    term = d_x_batch[:, k:k+1] * d_x_batch[:, k:]
                    d_f_x = cp.hstack([d_f_x, term])
            else:
                 raise NotImplementedError(f"GPU implementation for regression '{poly_type}' not yet added.")
            
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

