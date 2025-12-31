import sys
import os
# Ensure adaptive_computing is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.surrogates.smt import SMTGP
from adaptive_computing.surrogates.base import SurrogateModelBase
from gpu_kriging import GPUKriging

class GPUSMTGP(SMTGP):
    """
    A wrapper class for using GPU-accelerated Kriging as surrogate model.
    Inherits from SMTGP but replaces KRG with GPUKriging.
    """
    
    def __init__(self, dataset, smt_kwargs=None):
        # Initialize SurrogateModelBase directly to avoid SMTGP.__init__ creating KRG models
        SurrogateModelBase.__init__(self, dataset)

        if smt_kwargs is None:
            smt_kwargs = {}
        
        self.surrogate_model = []
        for i_fidelity in range(self.n_fidelity):
            if self.multifidelity and i_fidelity > 0:
                raise NotImplementedError("Multi-fidelity GPU Kriging not yet implemented.")
            else:
                # Use GPUKriging instead of KRG
                self.surrogate_model.append(GPUKriging(
                    **smt_kwargs,
                    print_global=False)) 
            
            if self.mixed_type:
                raise NotImplementedError("Mixed integer GPU Kriging not yet implemented.")
        
        self.untrained = True 

    def predict_variances(self, x_data, fidelity_level=-1):
        """
        Predicts variances using the GPU surrogate model and converts result to NumPy.
        """
        import cupy as cp
        # Call parent implementation
        result = super().predict_variances(x_data, fidelity_level)
        
        # Convert to NumPy if it's a CuPy array
        if isinstance(result, cp.ndarray):
            return result.get()
        return result 
