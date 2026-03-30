from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
from smt.applications.mixed_integer import MixedIntegerSurrogateModel, MixedIntegerKrigingModel
from adaptive_computing.surrogates.base import SurrogateModelBase
import numpy as np

class SMT_GP(SurrogateModelBase):
    """
    A wrapper class for using Surrogate Modeling Toolbox (SMT)
     Kriging or Multi-fidelity Kriging as surrogate model.
    
    Attributes:
        surrogate_model (list): List of surrogate models for each fidelity level.
        
    Methods:
        __init__(dataset, smt_kwargs=None): Initializes the SMT_GP.
        train(x_data, y_data): Trains the surrogate models.
        predict_values(x_data, fidelity_level=-1): Predicts values using the surrogate model.
        predict_variances(x_data, fidelity_level=-1): Predicts variances using the surrogate model.
    """
    
    def __init__(self, dataset, smt_kwargs=None, design_space=None):
        """
        Initializes the SMT_GP with the dataset and optional SMT-specific keyword arguments.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            smt_kwargs (dict, optional): Additional keyword arguments for configuring SMT models. Defaults to None.
            design_space (DesignSpace, optional): SMT design space for mixed-type problems. Defaults to None.
        """
        super().__init__(dataset)

        if smt_kwargs is None:
            smt_kwargs = {}
            
        # Get design space from dataset if available
        if design_space is None and hasattr(dataset, 'design_space'):
            design_space = dataset.design_space
        
        self.surrogate_model = []
        for i_fidelity in range(self.n_fidelity):
            if self.mixed_type:
                # Use regular KRG - mixed types are handled in the optimization algorithm
                self.surrogate_model.append(KRG(**smt_kwargs, print_global=False))
                print("Note: Mixed-type optimization using KRG surrogate with discrete enumeration in acquisition optimization")
            elif self.multifidelity and i_fidelity > 0:
                self.surrogate_model.append(MFK(
                    **smt_kwargs,
                    print_global=False))
            else:
                self.surrogate_model.append(KRG(
                    **smt_kwargs,
                    print_global=False))
        
        self.untrained = True # used to track if the surrogate model has never been trained

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
            for ii_fidelity in range(i_fidelity):
                self.surrogate_model[i_fidelity].set_training_values(x_data[ii_fidelity], y_data[ii_fidelity], name=ii_fidelity)
            self.surrogate_model[i_fidelity].set_training_values(x_data[i_fidelity], y_data[i_fidelity])
            self.surrogate_model[i_fidelity].train()
        
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
        return self.surrogate_model[fidelity_level].predict_values(np.atleast_2d(x_data))

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
        return self.surrogate_model[fidelity_level].predict_variances(np.atleast_2d(x_data))


class ConstrainedSMT_GP(SMT_GP):
    """
    A wrapper class for using Constrained Surrogate Modeling Toolbox (SMT) models as surrogate models.
    
    Attributes:
        constraint_func (function): Function to compute constraints.
        
    Methods:
        __init__(dataset, constraint_func, smt_kwargs=None, i_out=0): Initializes the ConstrainedSMT_GP.
        predict_constraint(x_data, fidelity_level=-1): Predicts constraints using the constraint function.
    """
    
    def __init__(self, dataset, constraint_func, smt_kwargs=None, i_out=0):
        """
        Initializes the ConstrainedSMT_GP with the dataset, constraint function, optional SMT-specific keyword arguments, and output index.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            constraint_func (function): Function to compute constraints based on input data.
            smt_kwargs (dict, optional): Additional keyword arguments for configuring SMT models. Defaults to None.
            i_out (int, optional): Index of the output for constrained optimization. Defaults to 0.
        """
        super().__init__(dataset, smt_kwargs=smt_kwargs)
        self.constraint_func = constraint_func
        
    def predict_constraint(self, x_data, fidelity_level=-1):
        """
        Predicts constraints using the constraint function at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            float: Predicted constraint value.
        """
        return self.constraint_func(x_data)
