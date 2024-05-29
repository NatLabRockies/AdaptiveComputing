from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
from smt.applications.mixed_integer import MixedIntegerSurrogateModel
from adaptive_computing.surrogates.base import SurrogateModelBase
import numpy as np

# Implement surrogate modeling using the Surrogate Modeling Toolbox (SMT)
class SMTWrapper(SurrogateModelBase):
    def __init__(self, dataset, smt_kwargs=None):
        # Call the constructor of the base class
        super().__init__(dataset)

        if smt_kwargs is None:
            smt_kwargs = {}
        
        # Initialize SMT-specific surrogate model
        # set upt the GPs Gaussian Process models (AKA the Kriging model)

        self.surrogate_model = []
        for i_fl in range(self.n_fl): # create at hierarchy of GPs
            if self.multifidelity and i_fl > 0:
                self.surrogate_model.append(MFK(
                    **smt_kwargs,
                    print_global = False))
            else:
                self.surrogate_model.append(KRG(
                    **smt_kwargs,
                    print_global = False)) 
            if self.mixed_type:
                self.surrogate_model[i_fl] = MixedIntegerSurrogateModel(surrogate=self.surrogate_model[i_fl], xtypes=dataset.xtypes, xlimits=dataset.xlimits)
        #dataset.train_on_unmasked_data(self)

    def train(self, x_data, y_data):
        x_data, y_data = self._validate_data(x_data, y_data)

        for i_fl in range(self.n_fl):

            # Set the training values for all levels below the current fidelity level by indicating the name field
            # Note: other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
            for ii_fl in range(i_fl):
                self.surrogate_model[i_fl].set_training_values(x_data[ii_fl],y_data[ii_fl], name=ii_fl)
            # Set the training values for the current fidelity level
            # Note: the name field should not be specified for the current fidelity level    )
            self.surrogate_model[i_fl].set_training_values(x_data[i_fl],y_data[i_fl])
            
            # Update the surrogate model using this data
            self.surrogate_model[i_fl].train()

    def predict_values(self, x_data, fidelity_level=-1):
        return self.surrogate_model[fidelity_level].predict_values(np.atleast_2d(x_data))

    def predict_variances(self, x_data, fidelity_level=-1):
        return self.surrogate_model[fidelity_level].predict_variances(np.atleast_2d(x_data))    


# Implement surrogate modeling using the Surrogate Modeling Toolbox (SMT)
class ConstrainedSMTWrapper(SMTWrapper):
    def __init__(self, dataset, constraint_func, smt_kwargs=None,i_out=0):
        # Call the constructor of the base class
        super().__init__(dataset, i_out=i_out, smt_kwargs=smt_kwargs)
        self.constraint_func = constraint_func
        
    def predict_constraint(self, x_data, fidelity_level=-1):
        return self.constraint_func(x_data)