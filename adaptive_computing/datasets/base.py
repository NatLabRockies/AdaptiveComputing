import numpy as np

class DatasetBase():
    """
    A base class for handling datasets with optional multi-fidelity data.
    
    Attributes:
        params (list): List of parameter objects, each with 'limits' and 'type' attributes.
        n_fidelity (int): Number of fidelity levels.
        y_bounds (tuple): Bounds for the output data.
        nan_behavior (str): Behavior when encountering NaN values ('fail', 'mask_replace', 'mask_ignore').
        oob_behavior (str): Behavior when encountering out-of-bounds values (None, 'fail').
        
    Methods:
        x_data (property): Returns the input data.
        y_data (property): Returns the output data.
        add_samples(x_data, y_data, n_fidelity): Adds new samples to the dataset after validation.
        N_samples (property): Returns the number of samples at each fidelity level.
        _sampler_ranges (property): Returns the ranges for sampling.
    """
    
    def __init__(self, params, n_fidelity=1, y_bounds=None, nan_behavior='fail', oob_behavior=None):
        """
        Initializes the DatasetBase with the given parameters and settings.
        
        Args:
            params (list): List of parameter objects, each with 'limits' and 'type' attributes.
            n_fidelity (int): Number of fidelity levels. Defaults to 1.
            y_bounds (tuple, optional): Bounds for the output data. Defaults to None.
            nan_behavior (str): Behavior when encountering NaN values ('fail', 'mask_replace', 'mask_ignore'). Defaults to 'fail'.
            oob_behavior (str, optional): Behavior when encountering out-of-bounds values (None, 'fail'). Defaults to None.
        
        Raises:
            ValueError: If invalid values are provided for nan_behavior or oob_behavior.
        """
        self.n_fl = n_fidelity
        self.multifidelity = self.n_fl > 1
        self.params = params

        self.n_in = len(params)
        self.x_limits = np.array([p.limits for p in params])
        self.x_types = [p.type for p in params]

        self.n_continuous = sum([t == 'continuous' for t in self.x_types])
        self.mixed_type = np.any([t != 'continuous' for t in self.x_types])

        self.n_out = 1
        # x_data and y_data are lists of length n_fl
        # Each entry will be an n_samp[i_fl] x (n_in or n_out) np array
        self._x_data = [np.empty([0, self.n_in])] * self.n_fl
        self._y_data = [np.empty([0, self.n_out])] * self.n_fl

        self.y_bounds = y_bounds

        if nan_behavior not in ['fail', 'mask_replace', 'mask_ignore']:
            print("nan_behavior must be one of ('fail','mask_replace', 'mask_ignore')")
            raise ValueError
        if oob_behavior not in [None, 'fail']:
            print("oob_behavior must be one of (None, 'fail')")
            raise ValueError
        if oob_behavior is not None and self.y_bounds is None:
            print("If oob_behavior is not None, y_bounds must be provided")
            raise ValueError
        
        self.nan_behavior = nan_behavior
        self.oob_behavior = oob_behavior

    @property
    def x_data(self):
        """Returns the input data."""
        return self._x_data
    
    @property
    def y_data(self):
        """Returns the output data."""
        return self._y_data
    
    def _validate_data(self, x_data, y_data, n_fidelity):
        """
        Validates the provided data for NaNs and out-of-bounds values.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            y_data (N Samples, N output dimensions): The output data.
            n_fidelity (int): The fidelity level of the data.
        
        Returns:
            tuple: Validated input and output data arrays.
        
        Raises:
            ValueError: If NaNs or out-of-bounds values are found and behavior is set to 'fail'.
        """
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        if np.any(np.isnan(y_data)):
            if self.nan_behavior == 'mask_ignore':
                idx = ~np.isnan(y_data)
                x_data = x_data[idx].reshape(-1, self.n_in)
                y_data = y_data[idx].reshape(-1, 1)
                print(f"Ignoring NaN data point at {x_data}, n_fidelity {n_fidelity}")
                print("This may result in repeated sampling of the same value")
            else:
                print(f"Simulation at {x_data}, n_fidelity {n_fidelity} returned NaN value")
                print(f"{y_data}")
                raise ValueError
        
        if self.y_bounds is not None:
            if np.any(y_data < self.y_bounds[0]) or np.any(y_data > self.y_bounds[1]):
                print(f"Simulation at {x_data}, n_fidelity {n_fidelity} returned OOB value")
                print(f"{y_data}")
                raise ValueError

        return x_data, y_data
            
    def add_samples(self, x_data, y_data, n_fidelity):
        """
        Validates and adds new samples to the dataset.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            y_data (N Samples, N output dimensions): The output data.
            n_fidelity (int): The fidelity level of the data.
        """
        x_data, y_data = self._validate_data(x_data, y_data, n_fidelity)

        self._x_data[n_fidelity] = np.concatenate([self._x_data[n_fidelity], x_data])
        self._y_data[n_fidelity] = np.concatenate([self._y_data[n_fidelity], y_data])
        
    @property
    def N_samples(self):
        """Returns the number of samples at each fidelity level."""
        return [self.x_data[i_fl].shape[0] for i_fl in range(self.n_fl)]

    @property
    def _sampler_ranges(self):
        """Returns the ranges for sampling."""
        ranges = ()
        for i in range(self.n_in):
            ranges = ranges + (slice(self.x_limits[i][0], self.x_limits[i][-1] + 1, 1),)
        return ranges


"""# self.funcs =[]
        # for i in range(self.n_fl):
        #     self.funcs.append(ComposedFunction(self.simulations[i],self.params))

        # Define xlimits, the domain for the design parameters
        if self.mixed_type:
            from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
            self.xtypes = []
            self.xlimits = [] # this is the domain for the user defined simulations[] (which may include mixed types)
            self.xlimits_num = [] # this is the domain with the categoricals and integers converted to continuous types. Categoricals are a list of floats.
            for i in range(self.n_in):
                if self.params[i].type == 'continuous':
                    self.xtypes.append(FLOAT)
                    self.xlimits.append([self.params[i].min_val, self.params[i].max_val])
                    self.xlimits_num.append([self.params[i].min_val, self.params[i].max_val])
                elif self.params[i].type == 'ordered':
                    self.xtypes.append(ORD)
                    self.xlimits.append([self.params[i].min_val, self.params[i].max_val])
                    self.xlimits_num.append([self.params[i].min_val, self.params[i].max_val])
                elif self.params[i].type == 'categorical':
                    self.xtypes.append((ENUM, len(self.params[i].categories)))
                    self.xlimits.append(self.params[i].categories)
                    self.xlimits_num.append(list(range(len(self.params[i].categories))))
                else:
                    raise Exception('Unrecognized type for parameter '+str(i)) 
        else:"""