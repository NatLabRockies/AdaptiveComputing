from adaptive_computing.datasets import DatasetBase
import numpy as np

class StochasticDataset(DatasetBase):
    """
    A base class for handling datasets with optional multi-fidelity data.
    Differs from the DatasetBase class since it can have multiple y values (stochastic samples)
    for a single x value (sample).
    
    Attributes:
        params (list): List of parameter objects, each with 'limits' and 'type' attributes.
        n_fidelity (int): Number of fidelity levels.
        y_bounds (tuple): Bounds for the output data.
        nan_behavior (str): Behavior when encountering NaN values ('fail', 'mask_replace', 'mask_ignore').
        oob_behavior (str): Behavior when encountering out-of-bounds values (None, 'fail').
        
    Methods:
        x_data (property): Returns the input data.
        y_data (property): Returns the output data.
        add_samples(x_data, y_data, i_fidelity): Adds new samples to the dataset after validation.
        add_more_sample(self, ix_data, y_data, i_fidelity): Adds more stochastic samples to the ix_data existing sample.
        N_samples (property): Returns the number of samples at each fidelity level.
        N_stoch_samples[N_samples] (property): Returns the number of stochastic samples for a particular sample (unique input location) at each fidelity level.
        _sampler_ranges (property): Returns the ranges for sampling.
    """
    
    def __init__(self, params, n_fidelity=1, y_bounds=None, nan_behavior='fail', oob_behavior=None, n_out=1):
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
        self.n_fidelity = n_fidelity
        self.multifidelity = self.n_fidelity > 1
        self.params = params

        self.n_in = len(params)
        self.x_limits = np.array([p.limits for p in params])
        self.x_types = [p.type for p in params]

        self.n_continuous = sum([t == 'continuous' for t in self.x_types])
        self.mixed_type = np.any([t != 'continuous' for t in self.x_types])

        self.n_out = n_out
        # x_data and y_data are lists of length n_fidelity
        # Each entry will be an n_samp[i_fidelity] x (n_in or n_out) np array
        self._x_data = [np.empty([0, self.n_in])] * self.n_fidelity
        self._n_stoch_samples = [np.empty([0, self.n_out])] * self.n_fidelity
        #self._y_data = [np.empty([0, self.n_out]) * self._n_stoch_samples[]] * self.n_fidelity
        self._y_data = [np.empty([0, self.n_out]) * 0] * self.n_fidelity

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
    
    def _validate_data(self, x_data, y_data, i_fidelity):
        """
        Validates the provided data for NaNs and out-of-bounds values.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            y_data (N Samples, N output dimensions): The output data.
            i_fidelity (int): The fidelity level of the data.
        
        Returns:
            tuple: Validated input and output data arrays.
        
        Raises:
            TypeError: If arguments are of the wrong type.
            ValueError: If NaNs or out-of-bounds values are found and behavior is set to 'fail'.
        """
        # Validate x_data type and dimensions
        if not isinstance(x_data, np.ndarray):
            raise TypeError(f"Expected a NumPy array, but got {type(x_data)}.")
        if x_data.ndim != 2:
            raise ValueError(f"Expected a 2D array for x_data, but got an array with {x_data.ndim} dimensions.")
        if x_data.shape[1] != self.n_in:
            raise ValueError(f"x_data must have {self.n_in} features, but got {x_data.shape[1]}.")

        # Validate y_data type
        if not isinstance(y_data, np.ndarray):
            raise TypeError(f"Expected a NumPy array, but got {type(y_data)}.")
        # Enforce float dtype for the inner arrays, even if y_data.ndim is not 3
        if y_data.ndim > 1:
            # Iterate over the elements and cast inner arrays to float
            for i in range(y_data.shape[0]):
                for j in range(y_data.shape[1]):
                    # Ensure the inner array is of type float (float64 specifically)
                    y_data[i, j] = np.asarray(y_data[i, j], dtype=float)
        if y_data.ndim == 3:  # Likely 3D due to casting lists to arrays
            result = np.empty((y_data.shape[0], y_data.shape[1]), dtype=object)
            for i in range(y_data.shape[0]):
                for j in range(y_data.shape[1]):
                    result[i, j] = np.asarray(y_data[i, j], dtype=float)
            y_data = result

        if y_data.ndim != 2:
            raise ValueError(f"Expected a 2D array for y_data, but got an array with {y_data.ndim} dimensions.")
        if y_data.shape[1] != self.n_out:
            raise ValueError(f"y_data must have {self.n_out} output dimensions, but got {y_data.shape[1]}.")
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError("x_data and y_data must have the same number of samples (rows).")

        # Ensure y_data elements are 1D numpy arrays
        for i, row in enumerate(y_data):
            for j, element in enumerate(row):
                if not isinstance(element, np.ndarray) or element.ndim != 1:
                    raise ValueError(
                        f"Expected element at position ({i}, {j}) to be a 1D numpy array, "
                        f"but got {type(element)} with ndim={getattr(element, 'ndim', None)}."
                    )

        valid_x_data = []
        valid_y_data = []

        # Clean up and validate rows
        for x_row, y_row in zip(x_data, y_data):
            if np.any(np.isnan(x_row)):
                continue  # Skip rows with NaNs in x_data

            # Check y_row for any output dimension with all NaNs
            skip_sample = False
            for out_dim in y_row:
                if np.all(np.isnan(out_dim)):  # Check if all elements are NaN in the current array
                    skip_sample = True
                    break  # Skip the entire sample if any output dimension has all NaNs
            if skip_sample:
                continue  # Skip the entire sample

            # Ensure all 1D arrays in y_row have the same length
            lengths = [len(arr) for arr in y_row]
            if len(set(lengths)) != 1:
                raise ValueError("All stochastic sample arrays in y_row must have the same length.")
            
            # Identify invalid stochastic samples (NaNs in any output dimension)
            drop_stoch = np.any([np.isnan(arr) for arr in y_row], axis=0)
            if not np.all(drop_stoch):  # Retain rows with valid stochastic samples
                valid_x_data.append(x_row)
                cleaned_y_row = [
                    arr[~drop_stoch] for arr in y_row  # Remove invalid stochastic samples
                ]
                valid_y_data.append(cleaned_y_row)

        # Convert valid data back to numpy arrays
        x_data = np.array(valid_x_data)
        y_data = np.array(valid_y_data, dtype=object)
        if y_data.ndim == 3:  # Likely 3D due to casting lists to arrays
            result = np.empty((y_data.shape[0], y_data.shape[1]), dtype=object)
            for i in range(y_data.shape[0]):
                for j in range(y_data.shape[1]):
                    result[i, j] = np.asarray(y_data[i, j], dtype=float)
            y_data = result
        return x_data, y_data
        
    def add_sample(self, x_data, y_data, i_fidelity):
        """
        Validates and adds samples at new input location.
        
        Args:
            x_data (N Samples, N input dimensions) array: The input data.
            y_data np_array of (N Samples, N output dimensions) of lists of length N_stoch_samples
            i_fidelity (int): The fidelity level of the data.
        """
        x_data, y_data = self._validate_data(x_data, y_data, i_fidelity)

        if x_data.size > 0:
            self._x_data[i_fidelity] = np.concatenate([self._x_data[i_fidelity], x_data])
        if y_data.size > 0:
            self._y_data[i_fidelity] = np.concatenate([self._y_data[i_fidelity], y_data])
        
    def add_more_sample(self, ix_data, y_data, i_fidelity):
        """
        Validates and appends more samples at a single existing input location.
        
        Args:
            ix_data (int): The index of which sample to add more stochastics samples to.
            y_data np_array of (N Stochastic Samples, N output dimensions) of new data points to append to the list of stochastic samples.
            i_fidelity (int): The fidelity level of the data.
        """
        x_data = np.atleast_2d(self._x_data[i_fidelity][ix_data])
        x_data, y_data = self._validate_data(x_data, y_data, i_fidelity)

        # if y_data.size > 0:
        #     self._y_data[i_fidelity][ix_data] = self._y_data[i_fidelity][ix_data].append([y_data])
        # Check if y_data size is valid
        if y_data.size > 0:
            # Flatten y_data if it has an extra dimension
            if y_data.ndim == 2 and len(y_data) == 1:
                y_data = y_data[0]  # Unpack the outer dimension
            
            # Ensure y_data structure matches self._y_data[i_fidelity][ix_data]
            if len(self._y_data[i_fidelity][ix_data]) != len(y_data):
                raise ValueError("Mismatch in output dimensions between existing data and y_data.")

            # Append to each output dimension
            for dim_idx in range(len(self._y_data[i_fidelity][ix_data])):
                self._y_data[i_fidelity][ix_data][dim_idx] = np.concatenate(
                    [self._y_data[i_fidelity][ix_data][dim_idx], y_data[dim_idx]]
                )

    @property
    def N_stoch_samples(self,ix_data):
        """Returns the number of stochastic samples at a particular input location."""
        return [self.y_data[i_fidelity][ix_data].shape[0] for i_fidelity in range(self.n_fidelity)]

