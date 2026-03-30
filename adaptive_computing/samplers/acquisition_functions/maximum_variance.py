import numpy as np
from scipy.stats import norm

def maximum_variance(x, surrogate, dataset, fidelity_level):
    """
    Maximum variance acquisition function for uncertainty-focused sampling.
    
    Args:
        x: Input point(s) to evaluate
        surrogate: Trained surrogate model
        dataset: Dataset object  
        fidelity_level: Fidelity level to use
        
    Returns:
        float or array: Negative variance (for minimization to achieve maximization)
    """
    try:
        # Ensure x is properly shaped
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Get variance predictions
        variances = surrogate.predict_variances(x, fidelity_level)
        
        # Handle edge cases
        if np.any(np.isnan(variances)):
            variances = np.nan_to_num(variances, nan=1e-10)
        if np.any(np.isinf(variances)):
            variances = np.nan_to_num(variances, posinf=1e10, neginf=1e-10)
        if np.any(variances <= 0):
            variances = np.maximum(variances, 1e-10)
        
        # Return negative variance for maximization (always as array for optimizer compatibility)
        result = -variances
        
        # Always return as array to ensure [0] indexing works in optimizer  
        if result.size == 1:
            return np.array([float(result.item())])
        else:
            return result.ravel()
            
    except Exception as e:
        print(f"Error in maximum_variance: {e}")
        # Safe fallback
        if hasattr(x, 'shape'):
            n_points = x.shape[0] if x.ndim > 1 else 1
        else:
            n_points = 1
        return np.array([-1e-6]) if n_points == 1 else np.full(n_points, -1e-6)
