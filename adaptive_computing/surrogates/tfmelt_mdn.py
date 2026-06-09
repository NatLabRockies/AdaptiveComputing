from tfmelt.models import ArtificialNeuralNetwork
from tfmelt.utils.evaluation import make_predictions
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from adaptive_computing.surrogates.base import SurrogateModelBase
import numpy as np

class TFMELT_MDN(SurrogateModelBase):
    """
    A wrapper class for using TF_MELT Mixture Density Network as surrogate model.
    
    MDN provides aleatoric uncertainty through multiple Gaussian mixture components,
    making it excellent for multi-modal functions and complex uncertainty patterns.
    
    Attributes:
        surrogate_model (list): List of TF_MELT MDN models for each fidelity level.
        x_scaler (StandardScaler): Scaler for normalizing input data.
        y_scaler (StandardScaler): Scaler for normalizing output data.
        
    Methods:
        __init__(dataset, tfmelt_kwargs=None): Initializes the TFMELT_MDN.
        train(x_data, y_data): Trains the surrogate models.
        predict_values(x_data, fidelity_level=-1): Predicts values using the surrogate model.
        predict_variances(x_data, fidelity_level=-1): Predicts variances using the surrogate model.
    """
    
    def __init__(self, dataset, tfmelt_kwargs=None):
        """
        Initializes the TFMELT_MDN with the dataset and optional TF_MELT-specific keyword arguments.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            tfmelt_kwargs (dict, optional): Additional keyword arguments for configuring TF_MELT MDN. Defaults to None.
        """
        super().__init__(dataset)

        if tfmelt_kwargs is None:
            tfmelt_kwargs = {}
        
        # Default MDN hyperparameters optimized for mixture density modeling
        self.n_epochs = tfmelt_kwargs.get('n_epochs', 300)  
        self.batch_size = tfmelt_kwargs.get('batch_size', 16)  
        self.learning_rate = tfmelt_kwargs.get('learning_rate', 1e-3)  # Standard learning rate for MDN
        # Architecture optimized for mixture density modeling
        self.width = tfmelt_kwargs.get('width', 64)  
        self.depth = tfmelt_kwargs.get('depth', 3)   
        self.act_fun = tfmelt_kwargs.get('act_fun', 'relu')
        self.l1_reg = tfmelt_kwargs.get('l1_reg', 0)  
        self.l2_reg = tfmelt_kwargs.get('l2_reg', 1e-5)  # Small regularization for MDN stability
        self.input_dropout = tfmelt_kwargs.get('input_dropout', 0.0)  
        self.dropout = tfmelt_kwargs.get('dropout', 0.1)  # Light dropout for regularization
        self.batch_norm = tfmelt_kwargs.get('batch_norm', False)  
        self.output_activation = tfmelt_kwargs.get('output_activation', 'linear')
        # MDN-specific parameters
        self.num_mixtures = tfmelt_kwargs.get('num_mixtures', 3)  # Number of Gaussian mixture components
        self.initializer = tfmelt_kwargs.get('initializer', 'glorot_uniform')
        self.verbose = tfmelt_kwargs.get('verbose', 1)
        
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
        Internal TFMELT_MDN training implementation.
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
            
            # Create the MDN model
            self.surrogate_model[i_fidelity] = ArtificialNeuralNetwork(
                num_outputs=self.output_dim,
                width=self.width,
                depth=self.depth,
                act_fun=self.act_fun,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                input_dropout=self.input_dropout,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                output_activation=self.output_activation,
                initializer=self.initializer,
                num_mixtures=self.num_mixtures,  # Critical: enables mixture density network
            )
            
            # Compile the model with custom loss for MDN
            self.surrogate_model[i_fidelity].compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',  # TF_MELT handles MDN loss internally when num_mixtures > 0
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
        Predicts values using the MDN surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted values (mean of mixture).
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model values, but user never called train() since initializing the surrogate.")
        
        x_data = np.atleast_2d(x_data)
        x_scaled = self.x_scaler[fidelity_level].transform(x_data)
        
        # Get predictions using TF_MELT's make_predictions for MDN
        pred_mean, _ = make_predictions(
            self.surrogate_model[fidelity_level], 
            x_scaled, 
            y_normalizer=self.y_scaler[fidelity_level], 
            unnormalize=True
        )
        
        return pred_mean

    def predict_variances(self, x_data, fidelity_level=-1):
        """
        Predicts variances using the MDN surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted variances (uncertainty from mixture components).
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model variances, but user never called train() since initializing the surrogate.")
        
        x_data = np.atleast_2d(x_data)
        x_scaled = self.x_scaler[fidelity_level].transform(x_data)
        
        # Get predictions with uncertainty using TF_MELT's make_predictions for MDN
        pred_mean, pred_std = make_predictions(
            self.surrogate_model[fidelity_level], 
            x_scaled, 
            y_normalizer=self.y_scaler[fidelity_level], 
            unnormalize=True
        )
        
        # Return variance (std squared) from mixture density components
        return pred_std ** 2