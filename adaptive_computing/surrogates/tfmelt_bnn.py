from tfmelt.models import BayesianNeuralNetwork
from tfmelt.utils.evaluation import ensemble_predictions
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from adaptive_computing.surrogates.base import SurrogateModelBase
import numpy as np
import tensorflow as tf

class TFMELT_BNN(SurrogateModelBase):
    """
    A wrapper class for using TF_MELT Bayesian Neural Network as surrogate model.
    
    Attributes:
        surrogate_model (list): List of TF_MELT BNN models for each fidelity level.
        x_scaler (StandardScaler): Scaler for normalizing input data.
        y_scaler (StandardScaler): Scaler for normalizing output data.
        
    Methods:
        __init__(dataset, tfmelt_kwargs=None): Initializes the TFMELT_BNN.
        train(x_data, y_data): Trains the surrogate models.
        predict_values(x_data, fidelity_level=-1): Predicts values using the surrogate model.
        predict_variances(x_data, fidelity_level=-1): Predicts variances using the surrogate model.
    """
    
    def __init__(self, dataset, tfmelt_kwargs=None):
        """
        Initializes the TFMELT_BNN with the dataset and optional TF_MELT-specific keyword arguments.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            tfmelt_kwargs (dict, optional): Additional keyword arguments for configuring TF_MELT BNN. Defaults to None.
        """
        super().__init__(dataset)

        if tfmelt_kwargs is None:
            tfmelt_kwargs = {}
        
        # Optimized BNN: proven architecture + minimal Bayesian regularization
        self.n_epochs = tfmelt_kwargs.get('n_epochs', 200)  # More epochs for BNN convergence
        self.batch_size = tfmelt_kwargs.get('batch_size', 10)
        self.learning_rate = tfmelt_kwargs.get('learning_rate', 1e-2)
        self.width = tfmelt_kwargs.get('width', 64)
        self.depth = tfmelt_kwargs.get('depth', 3)
        self.node_list = tfmelt_kwargs.get('node_list', None)  # Let TF_MELT handle architecture 
        self.bayesian_mask = tfmelt_kwargs.get('bayesian_mask', [False, False, False])  # Regular hidden layers (not counting the Bayesian output layer)
        self.act_fun = tfmelt_kwargs.get('act_fun', 'relu') 
        self.l1_reg = tfmelt_kwargs.get('l1_reg', 0)  
        self.l2_reg = tfmelt_kwargs.get('l2_reg', 0)  
        self.input_dropout = tfmelt_kwargs.get('input_dropout', 0.0)  
        self.dropout = tfmelt_kwargs.get('dropout', 0.1)
        self.batch_norm = tfmelt_kwargs.get('batch_norm', False)  
        self.use_batch_renorm = tfmelt_kwargs.get('use_batch_renorm', False)  
        self.output_activation = tfmelt_kwargs.get('output_activation', 'linear')
        self.do_aleatoric = tfmelt_kwargs.get('do_aleatoric', False)
        self.aleatoric_scale_factor = tfmelt_kwargs.get('aleatoric_scale_factor', 5e-2)
        self.scale_epsilon = tfmelt_kwargs.get('scale_epsilon', 1e-3)
        self.do_bayesian_output = tfmelt_kwargs.get('do_bayesian_output', True)  # Only Bayesian output for uncertainty
        self.initializer = tfmelt_kwargs.get('initializer', 'glorot_uniform')
        self.n_iter = tfmelt_kwargs.get('n_iter', 5)  
        self.n_iter_full = tfmelt_kwargs.get('n_iter_full', 100)  
        self.verbose = tfmelt_kwargs.get('verbose', 1)
        self.fast_mode = tfmelt_kwargs.get('fast_mode', True)
        
        # For multi-fidelity: TF_MELT doesn't natively support it, so we use separate models
        self.surrogate_model = []
        self.x_scaler = []
        self.y_scaler = []
        self.input_dim = None
        self.output_dim = None
        
        for i_fidelity in range(self.n_fidelity):
            self.x_scaler.append(StandardScaler())
            self.y_scaler.append(StandardScaler())
            self.surrogate_model.append(None)  # Will be created during training
        
        self.untrained = True

    def _train_impl(self, x_data, y_data):
        """
        Internal TFMELT_BNN training implementation.
        This method receives only validated, unmasked training data.
        
        Args:
            x_data (list): List of input data arrays for each fidelity level (unmasked only).
            y_data (list): List of output data arrays for each fidelity level (unmasked only).
        """
        for i_fidelity in range(self.n_fidelity):
            if len(x_data[i_fidelity]) == 0:
                continue
                
            # Get input/output dimensions from data
            self.input_dim = x_data[i_fidelity].shape[1]
            self.output_dim = y_data[i_fidelity].shape[1] if len(y_data[i_fidelity].shape) > 1 else 1
            
            # Scale the data
            x_scaled = self.x_scaler[i_fidelity].fit_transform(x_data[i_fidelity])
            y_scaled = self.y_scaler[i_fidelity].fit_transform(
                y_data[i_fidelity].reshape(-1, 1) if len(y_data[i_fidelity].shape) == 1 
                else y_data[i_fidelity]
            )
            
            # Create the BNN model - temporarily using regular NN to debug
            self.surrogate_model[i_fidelity] = BayesianNeuralNetwork(
                num_outputs=self.output_dim,
                width=self.width,  # Use width/depth instead of node_list
                depth=self.depth,
                act_fun=self.act_fun,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                input_dropout=self.input_dropout,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                use_batch_renorm=self.use_batch_renorm,
                output_activation=self.output_activation,
                num_points=x_scaled.shape[0],  # Critical: number of training points for BNN
                do_aleatoric=self.do_aleatoric,
                do_bayesian_output=self.do_bayesian_output,  # False for regular NN first
                aleatoric_scale_factor=self.aleatoric_scale_factor, 
                scale_epsilon=self.scale_epsilon,
                initializer=self.initializer,
                node_list=self.node_list,  # None - use width/depth
                bayesian_mask=self.bayesian_mask,  # [False, False, False] = regular NN
            )
            
            # Compile the model
            self.surrogate_model[i_fidelity].compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
            )
            
            # Build the model
            self.surrogate_model[i_fidelity].build(input_shape=(None, self.input_dim))
            
            # Train the model
            history = self.surrogate_model[i_fidelity].fit(
                x_scaled, y_scaled,
                epochs=self.n_epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                shuffle=True
            )
        
        self.untrained = False

    def predict_values(self, x_data, fidelity_level=-1):
        """
        Predicts values using the surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted values (mean).
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model values, but user never called train() since initializing the surrogate.")
        
        x_data = np.atleast_2d(x_data)
        x_scaled = self.x_scaler[fidelity_level].transform(x_data)
        
        # Use fast mode for optimization, full mode for final predictions
        n_iterations = self.n_iter if self.fast_mode or x_data.shape[0] > 5 else self.n_iter_full
        
        # Get predictions using Monte Carlo sampling
        pred_mean, _ = ensemble_predictions(
            self.surrogate_model[fidelity_level], 
            x_scaled, 
            self.y_scaler[fidelity_level], 
            unnormalize=True, 
            n_iter=n_iterations, 
            training=True
        )
        
        return pred_mean

    def predict_variances(self, x_data, fidelity_level=-1):
        """
        Predicts variances using the surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted variances (uncertainty squared).
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model variances, but user never called train() since initializing the surrogate.")
        
        x_data = np.atleast_2d(x_data)
        x_scaled = self.x_scaler[fidelity_level].transform(x_data)
        
        # Use fast mode for optimization, full mode for final predictions
        n_iterations = self.n_iter if self.fast_mode or x_data.shape[0] > 5 else self.n_iter_full
        
        # Get predictions using Monte Carlo sampling
        _, pred_std = ensemble_predictions(
            self.surrogate_model[fidelity_level], 
            x_scaled, 
            self.y_scaler[fidelity_level], 
            unnormalize=True, 
            n_iter=n_iterations, 
            training=True
        )
        
        # Return variance (std squared)
        return pred_std ** 2
    
    def enable_fast_mode(self):
        """Enable fast mode for acquisition optimization."""
        self.fast_mode = True
    
    def enable_full_mode(self):
        """Enable full mode for final predictions."""
        self.fast_mode = False
