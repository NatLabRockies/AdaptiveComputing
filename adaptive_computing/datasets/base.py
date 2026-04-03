import numpy as np

class DatasetBase():
    """
    A base class for handling datasets with optional multi-fidelity data.
    
    Attributes:
        params (list): List of parameter objects, each with 'limits' and 'type' attributes.
        n_fidelity (int): Number of fidelity levels.
        y_bounds (tuple): Bounds for the output data.
        nan_behavior (str): Behavior when encountering NaN values ('fail', 'mask_ignore').
        oob_behavior (str): Behavior when encountering out-of-bounds values (None, 'fail').
        
    Methods:
        x_data (property): Returns the input data.
        y_data (property): Returns the output data.
        add_samples(x_data, y_data, i_fidelity): Adds new samples to the dataset after validation.
        N_samples (property): Returns the number of samples at each fidelity level.
        _sampler_ranges (property): Returns the ranges for sampling.
    """
    
    def __init__(self, params, n_fidelity=1, y_bounds=None, nan_behavior='fail', oob_behavior=None, n_out=1):
        """
        Initializes the DatasetBase with the given parameters and settings.
        
        Args:
            params (list): List of parameter objects, each with 'limits' and 'type' attributes.
            n_fidelity (int): Number of fidelity levels. Defaults to 1.
            y_bounds (tuple, optional): Bounds for the output data. Defaults to None.
            nan_behavior (str): Behavior when encountering NaN values ('fail', 'mask_ignore'). Defaults to 'fail'.
            oob_behavior (str, optional): Behavior when encountering out-of-bounds values (None, 'fail'). Defaults to None.
        
        Raises:
            ValueError: If invalid values are provided for nan_behavior or oob_behavior.
        """
        self.n_fidelity = n_fidelity
        self.multifidelity = self.n_fidelity > 1
        self.params = params

        self.n_in = len(params)
        # For mixed types, x_limits may have different structures, so keep as list
        if np.any([p.type != 'continuous' for p in params]):
            self.x_limits = [p.limits for p in params]
        else:
            # For continuous only, can use numpy array
            self.x_limits = np.array([p.limits for p in params])
        self.x_types = [p.type for p in params]

        self.n_continuous = sum([t == 'continuous' for t in self.x_types])
        self.mixed_type = np.any([t != 'continuous' for t in self.x_types])

        # Initialize SMT-specific mixed-type properties
        if self.mixed_type:
            self._setup_smt_mixed_type_properties()

        self.n_out = n_out
        # x_data and y_data are lists of length n_fidelity
        # Each entry will be an n_samp[i_fidelity] x (n_in or n_out) np array
        self._x_data = [np.empty([0, self.n_in])] * self.n_fidelity
        self._y_data = [np.empty([0, self.n_out])] * self.n_fidelity
        
        # Unified masking: track validity for each output dimension (True=valid, False=masked)
        # Shape: [n_samples, n_out] per fidelity level
        self._unmasked_data = [np.empty([0, self.n_out], dtype=bool)] * self.n_fidelity

        self.y_bounds = y_bounds

        if nan_behavior not in ['fail', 'mask_ignore']:
            print("nan_behavior must be one of ('fail', 'mask_ignore')")
            raise ValueError
        if oob_behavior not in [None, 'fail', 'mask_ignore']:
            print("oob_behavior must be one of (None, 'fail', 'mask_ignore')")
            raise ValueError
        if oob_behavior is not None and self.y_bounds is None:
            print("If oob_behavior is not None, y_bounds must be provided")
            raise ValueError
        
        self.nan_behavior = nan_behavior
        self.oob_behavior = oob_behavior

    def _setup_smt_mixed_type_properties(self):
        """
        Sets up SMT 2.x design space for mixed-type optimization.
        """
        try:
            from smt.design_space import DesignSpace, FloatVariable, IntegerVariable, CategoricalVariable as SMTCategoricalVariable
            
            # Create design variables for SMT 2.x
            design_vars = []
            for i in range(self.n_in):
                param = self.params[i]
                if param.type == 'continuous':
                    design_vars.append(FloatVariable(param.min, param.max))
                elif param.type == 'ordered':
                    design_vars.append(IntegerVariable(param.min_val, param.max_val))
                elif param.type == 'categorical':
                    design_vars.append(SMTCategoricalVariable(param.categories))
                else:
                    raise ValueError(f'Unrecognized parameter type: {param.type}')
            
            # Create the design space
            self.design_space = DesignSpace(design_vars)
            self.smt_mixed_support = True
            self.smt_version = '2.x'
            
            print(f"SMT 2.x design space created with {len(design_vars)} variables")
            
        except ImportError as e:
            raise ImportError(f"SMT 2.x is required for mixed-type optimization. Please install/upgrade SMT: pip install --upgrade smt. Error: {e}")

    @property
    def x_data(self):
        """Returns the input data."""
        return self._x_data
    
    @property
    def y_data(self):
        """Returns the output data."""
        return self._y_data
        
    def get_unmasked_data(self, i_fidelity=None, i_output=None):
        """
        Get only the unmasked (valid) data for training.
        
        Args:
            i_fidelity (int, optional): Specific fidelity level. If None, returns all fidelities.
            i_output (int, optional): Specific output dimension. If None, requires all outputs to be valid.
                For single-output surrogates (SMT_GP, SOOGO_GP): pass i_output to filter by that specific output.
                For multi-output surrogates (TFMELT_*): don't pass i_output to require all outputs valid.
            
        Returns:
            tuple: (x_data, y_data) containing only unmasked data points
        """
        if i_fidelity is not None:
            if i_output is not None:
                # Filter by specific output dimension
                mask = self._unmasked_data[i_fidelity][:, i_output]
            else:
                # All outputs must be valid
                mask = np.all(self._unmasked_data[i_fidelity], axis=1)
            return self._x_data[i_fidelity][mask], self._y_data[i_fidelity][mask]
        else:
            x_unmasked = []
            y_unmasked = []
            for i_fid in range(self.n_fidelity):
                if i_output is not None:
                    # Filter by specific output dimension
                    mask = self._unmasked_data[i_fid][:, i_output]
                else:
                    # All outputs must be valid
                    mask = np.all(self._unmasked_data[i_fid], axis=1) if self._unmasked_data[i_fid].shape[0] > 0 else np.empty([0], dtype=bool)
                x_unmasked.append(self._x_data[i_fid][mask])
                y_unmasked.append(self._y_data[i_fid][mask])
            return x_unmasked, y_unmasked
    
    def _validate_input(self, x_data, i_fidelity):
        """
        Validates the provided x_data for NaNs and out-of-bounds values.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            i_fidelity (int): The fidelity level of the data.
        
        Raises:
            ValueError: If NaNs or out-of-bounds values are found.
        """
        x_data = np.asarray(x_data)
        
        # Check for nans (only for continuous and ordered variables)
        if np.any(np.isnan(x_data)):
            print(f"One or more of the entries in x_data={x_data} for i_fidelity={i_fidelity} is a nan value.")
            raise ValueError("x_data contains nan values.")
        
        # Check for out of bounds values
        if x_data.shape[1] != len(self.params):
            raise ValueError(f"x_data has {x_data.shape[1]} columns, but expected {len(self.params)} based on self.params.")
        
        for i in range(len(self.params)):
            param = self.params[i]
            if param.type == 'continuous':
                param_min = param.min
                param_max = param.max
                if np.any(x_data[:,i] < param_min) or np.any(x_data[:,i] > param_max):
                    print(f"One or more of the entries in x_data={x_data} for i_fidelity={i_fidelity} is an out of bounds value based on the user specified params min and max.")
                    raise ValueError(f"x_data[:, {i}] contains values outside the range [{param_min}, {param_max}].")
            elif param.type == 'ordered':
                param_min = param.min_val
                param_max = param.max_val
                if np.any(x_data[:,i] < param_min) or np.any(x_data[:,i] > param_max):
                    print(f"One or more of the entries in x_data={x_data} for i_fidelity={i_fidelity} is an out of bounds value based on the user specified params min_val and max_val.")
                    raise ValueError(f"x_data[:, {i}] contains values outside the range [{param_min}, {param_max}].")
                # Check that values are integers
                if not np.all(np.equal(np.mod(x_data[:,i], 1), 0)):
                    print(f"One or more of the entries in x_data={x_data} for i_fidelity={i_fidelity} is not an integer for ordered variable {i}.")
                    raise ValueError(f"x_data[:, {i}] contains non-integer values for ordered variable.")
            elif param.type == 'categorical':
                # Check that indices are within the valid range
                n_categories = len(param.categories)
                if np.any(x_data[:,i] < 0) or np.any(x_data[:,i] >= n_categories):
                    print(f"One or more of the entries in x_data={x_data} for i_fidelity={i_fidelity} is an out of bounds categorical index.")
                    raise ValueError(f"x_data[:, {i}] contains categorical indices outside the range [0, {n_categories-1}].")
                # Check that indices are integers
                if not np.all(np.equal(np.mod(x_data[:,i], 1), 0)):
                    print(f"One or more of the entries in x_data={x_data} for i_fidelity={i_fidelity} is not an integer for categorical variable {i}.")
                    raise ValueError(f"x_data[:, {i}] contains non-integer indices for categorical variable.")
    
    def _validate_data(self, x_data, y_data, i_fidelity):
        """
        Validates the provided y_data for NaNs and out-of-bounds values.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            y_data (N Samples, N output dimensions): The output data.
            i_fidelity (int): The fidelity level of the data.
        
        Returns:
            tuple: (validated x_data, validated y_data, unmasked_data) where unmasked_data 
                   indicates valid data points per output dimension [N samples, N outputs].
                   For 'mask_ignore' behavior: returns original data with mask indicating valid outputs.
                   For 'fail' behavior: raises exception if invalid data found.
        
        Raises:
            ValueError: If NaNs or out-of-bounds values are found and behavior is set to 'fail'.
        """
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        
        # Ensure y_data is 2D for consistent handling
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
        
        # Initialize mask as all valid for each output dimension
        unmasked_data = np.ones((len(x_data), self.n_out), dtype=bool)

        # Check for NaN values per output dimension
        idx_nan = np.isnan(y_data)
        if np.any(idx_nan):
            nan_samples = np.any(idx_nan, axis=1)  # Which samples have any NaN
            print(f"NaN data point detected at {x_data[nan_samples]}, i_fidelity {i_fidelity}")
            
            if self.nan_behavior == 'mask_ignore':
                print("Marking NaN output dimensions as masked. This may result in repeated sampling of the same value.")
                # Mark specific output dimensions with NaN as masked
                unmasked_data[idx_nan] = False
            else:
                print("NaN data triggers a ValueError. Consider setting nan_behavior='mask_ignore' to mask NaNs.")
                raise ValueError("NaN values detected in y_data. Use 'mask_ignore' to ignore them.")
        
        # Check for out-of-bounds values per output dimension
        if self.y_bounds is not None:
            idx_oob = (y_data < self.y_bounds[0]) | (y_data > self.y_bounds[1])
            if np.any(idx_oob):
                oob_samples = np.any(idx_oob, axis=1)  # Which samples have any OOB
                print(f"Simulation at {x_data[oob_samples]}, i_fidelity {i_fidelity} returned OOB value")
                print(f"{y_data[oob_samples]}")
                
                if self.oob_behavior == 'mask_ignore':
                    print("Marking out-of-bounds output dimensions as masked. This may result in repeated sampling of the same value.")
                    # Mark specific output dimensions that are OOB as masked
                    unmasked_data[idx_oob] = False
                else:
                    print("Out-of-bounds data triggers a ValueError. Consider setting oob_behavior='mask_ignore' to mask OOB values.")
                    raise ValueError("Out-of-bounds values detected in y_data. Use 'mask_ignore' to ignore them.")

        return x_data, y_data, unmasked_data
            
    def add_samples(self, x_data, y_data, i_fidelity=0):
        """
        Validates and adds new samples to the dataset.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            y_data (N Samples, N output dimensions): The output data.
            i_fidelity (int): The fidelity level of the data.
        """
        self._validate_input(x_data, i_fidelity)
        x_data, y_data, unmasked_data = self._validate_data(x_data, y_data, i_fidelity)

        # Keep all data and track which output dimensions are valid
        self._x_data[i_fidelity] = np.concatenate([self._x_data[i_fidelity], x_data])
        self._y_data[i_fidelity] = np.concatenate([self._y_data[i_fidelity], y_data])
        self._unmasked_data[i_fidelity] = np.concatenate([self._unmasked_data[i_fidelity], unmasked_data])
        
    @property
    def N_samples(self):
        """Returns the number of samples at each fidelity level."""
        return [self.x_data[i_fidelity].shape[0] for i_fidelity in range(self.n_fidelity)]

    @property
    def _sampler_ranges(self):
        """Returns the ranges for sampling based on variable types."""
        ranges = ()
        for i in range(self.n_in):
            param = self.params[i]
            if param.type == 'continuous':
                ranges = ranges + (slice(param.min, param.max + 1, 1),)
            elif param.type == 'ordered':
                ranges = ranges + (slice(param.min_val, param.max_val + 1, 1),)
            elif param.type == 'categorical':
                ranges = ranges + (slice(0, len(param.categories), 1),)
            else:
                raise ValueError(f"Unknown parameter type: {param.type}")
        return ranges

