"""
Test suite for output selection functionality in single-output surrogate models.

Tests the i_output parameter functionality in SMT_GP and SOOGO_GP models
to ensure users can explicitly select which output dimension to model
when working with multi-output datasets.
"""

import pytest
import numpy as np
from adaptive_computing.datasets.base import DatasetBase
from adaptive_computing.datasets.variables import ContinuousVariable
from adaptive_computing.surrogates.smt_gp import SMT_GP, ConstrainedSMT_GP
from adaptive_computing.surrogates.soogo_gp import SOOGO_GP


class TestSurrogateOutputSelection:
    """Test output selection functionality for single-output surrogate models."""
    
    def setup_method(self):
        """Set up test data with multiple outputs."""
        np.random.seed(42)
        
        # Create parameter objects
        self.params = [
            ContinuousVariable(min=0.0, max=1.0),  # x1
            ContinuousVariable(min=0.0, max=1.0)   # x2  
        ]
        
        # Create multi-output test data with clearly distinguishable patterns
        self.x_train = np.array([
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]
        ])
        
        # Create 3 outputs with very different scales/patterns to test selection
        self.y_train_3out = np.array([
            [10.0, 1000.0, 0.1],        # Output 0: ~10s, Output 1: ~1000s, Output 2: ~0.1s
            [30.0, 3000.0, 0.3],       
            [50.0, 5000.0, 0.5],       
            [70.0, 7000.0, 0.7],       
            [90.0, 9000.0, 0.9]        
        ])
        
        # Create 2-output version for some tests
        self.y_train_2out = self.y_train_3out[:, :2]
        
        # Single output version
        self.y_train_1out = self.y_train_3out[:, :1]
        
    def create_dataset(self, n_out):
        """Create a dataset with specified number of outputs."""
        dataset = DatasetBase(params=self.params, n_fidelity=1, n_out=n_out)
        
        if n_out == 1:
            y_data = self.y_train_1out
        elif n_out == 2:
            y_data = self.y_train_2out
        elif n_out == 3:
            y_data = self.y_train_3out
        else:
            raise ValueError(f"Unsupported n_out: {n_out}")
            
        dataset.add_known_samples(self.x_train, y_data)
        return dataset


class TestSMT_GP_OutputSelection(TestSurrogateOutputSelection):
    """Test SMT_GP output selection functionality."""
    
    def test_constructor_validation_valid_indices(self):
        """Test that valid i_output values are accepted."""
        dataset_3out = self.create_dataset(3)
        
        # Valid indices should work
        gp0 = SMT_GP(dataset_3out, i_output=0)
        assert gp0.i_output == 0
        
        gp1 = SMT_GP(dataset_3out, i_output=1)
        assert gp1.i_output == 1
        
        gp2 = SMT_GP(dataset_3out, i_output=2)
        assert gp2.i_output == 2
        
    def test_constructor_validation_invalid_indices(self):
        """Test that invalid i_output values raise appropriate errors."""
        dataset_3out = self.create_dataset(3)
        
        # Test out-of-range indices
        with pytest.raises(ValueError, match="i_output=3 is invalid.*valid range: 0 to 2"):
            SMT_GP(dataset_3out, i_output=3)
            
        with pytest.raises(ValueError, match="i_output=-1 is invalid.*valid range: 0 to 2"):
            SMT_GP(dataset_3out, i_output=-1)
            
        with pytest.raises(ValueError, match="i_output=10 is invalid.*valid range: 0 to 2"):
            SMT_GP(dataset_3out, i_output=10)
    
    def test_single_output_dataset_default(self):
        """Test that single output datasets work with default i_output=0."""
        dataset_1out = self.create_dataset(1)
        
        gp = SMT_GP(dataset_1out)
        assert gp.i_output == 0
        
        # Should be able to train and predict
        gp.train(dataset_1out)
        assert not gp.untrained
        
    def test_multi_output_training_different_outputs(self):
        """Test that different i_output values produce different models."""
        dataset_3out = self.create_dataset(3)
        
        # Create GPs for different outputs
        gp0 = SMT_GP(dataset_3out, i_output=0)
        gp1 = SMT_GP(dataset_3out, i_output=1) 
        gp2 = SMT_GP(dataset_3out, i_output=2)
        
        # Train all models
        gp0.train(dataset_3out)
        gp1.train(dataset_3out)
        gp2.train(dataset_3out)
        
        # All should be trained
        assert not gp0.untrained
        assert not gp1.untrained
        assert not gp2.untrained
        
        # Test predictions on new point
        x_test = np.array([[0.4, 0.5]])
        
        pred0 = gp0.predict_values(x_test)
        pred1 = gp1.predict_values(x_test)
        pred2 = gp2.predict_values(x_test)
        
        # Predictions should be very different due to different output scales
        # Output 0: ~10s, Output 1: ~1000s, Output 2: ~0.1s
        print(f"Predictions: pred0={pred0[0,0]:.3f}, pred1={pred1[0,0]:.3f}, pred2={pred2[0,0]:.3f}")
        
        # Check that output 1 >> output 0 (should be ~100x larger)
        assert abs(pred1[0,0] / pred0[0,0]) > 10, f"GP output 1 should be much larger than output 0: pred0={pred0[0,0]}, pred1={pred1[0,0]}"
        
        # Check that output 0 >> output 2 (should be ~100x larger)  
        assert abs(pred0[0,0] / pred2[0,0]) > 10, f"GP output 0 should be much larger than output 2: pred0={pred0[0,0]}, pred2={pred2[0,0]}"
        
        # Check that predictions are in reasonable ranges based on training data
        assert 5 < pred0[0,0] < 100, f"Output 0 prediction should be in range [5,100]: {pred0[0,0]}"
        assert 500 < pred1[0,0] < 10000, f"Output 1 prediction should be in range [500,10000]: {pred1[0,0]}"
        assert 0.05 < pred2[0,0] < 1.0, f"Output 2 prediction should be in range [0.05,1.0]: {pred2[0,0]}"
    
    def test_constrained_smt_gp_output_selection(self):
        """Test that ConstrainedSMT_GP properly passes i_out to parent."""
        dataset_2out = self.create_dataset(2)
        
        # Simple constraint function
        def constraint_func(x):
            return 0.0  # Always feasible
            
        # Test valid i_out parameter (note: ConstrainedSMT_GP uses i_out, not i_output)
        constrained_gp = ConstrainedSMT_GP(dataset_2out, constraint_func, i_out=1)
        assert constrained_gp.i_output == 1
        
        # Should be able to train
        constrained_gp.train(dataset_2out)
        assert not constrained_gp.untrained


class TestSOOGO_GP_OutputSelection(TestSurrogateOutputSelection):
    """Test SOOGO_GP output selection functionality."""
    
    def test_constructor_validation_valid_indices(self):
        """Test that valid i_output values are accepted."""
        dataset_3out = self.create_dataset(3)
        
        # Valid indices should work
        gp0 = SOOGO_GP(dataset_3out, i_output=0)
        assert gp0.i_output == 0
        
        gp1 = SOOGO_GP(dataset_3out, i_output=1)
        assert gp1.i_output == 1
        
        gp2 = SOOGO_GP(dataset_3out, i_output=2)
        assert gp2.i_output == 2
        
    def test_constructor_validation_invalid_indices(self):
        """Test that invalid i_output values raise appropriate errors."""
        dataset_3out = self.create_dataset(3)
        
        # Test out-of-range indices
        with pytest.raises(ValueError, match="i_output=3 is invalid.*valid range: 0 to 2"):
            SOOGO_GP(dataset_3out, i_output=3)
            
        with pytest.raises(ValueError, match="i_output=-1 is invalid.*valid range: 0 to 2"):
            SOOGO_GP(dataset_3out, i_output=-1)
    
    def test_single_output_dataset_default(self):
        """Test that single output datasets work with default i_output=0."""
        dataset_1out = self.create_dataset(1)
        
        gp = SOOGO_GP(dataset_1out, soogo_kwargs={'tune_alpha': False})
        assert gp.i_output == 0
        
        # Should be able to train and predict
        gp.train(dataset_1out)
        assert not gp.untrained
        
    def test_multi_output_training_different_outputs(self):
        """Test that different i_output values produce different models."""
        dataset_3out = self.create_dataset(3)
        
        # Create GPs for different outputs (disable alpha tuning for faster tests)
        soogo_kwargs = {'tune_alpha': False}
        gp0 = SOOGO_GP(dataset_3out, soogo_kwargs=soogo_kwargs, i_output=0)
        gp1 = SOOGO_GP(dataset_3out, soogo_kwargs=soogo_kwargs, i_output=1) 
        gp2 = SOOGO_GP(dataset_3out, soogo_kwargs=soogo_kwargs, i_output=2)
        
        # Train all models
        gp0.train(dataset_3out)
        gp1.train(dataset_3out)
        gp2.train(dataset_3out)
        
        # All should be trained
        assert not gp0.untrained
        assert not gp1.untrained
        assert not gp2.untrained
        
        # Test predictions on new point
        x_test = np.array([[0.4, 0.5]])
        
        pred0 = gp0.predict_values(x_test)
        pred1 = gp1.predict_values(x_test)
        pred2 = gp2.predict_values(x_test)
        
        # Predictions should be very different due to different output scales
        # Output 0: ~10s, Output 1: ~1000s, Output 2: ~0.1s
        print(f"Predictions: pred0={pred0[0,0]:.3f}, pred1={pred1[0,0]:.3f}, pred2={pred2[0,0]:.3f}")
        
        # Check that output 1 >> output 0 (should be ~100x larger)
        assert abs(pred1[0,0] / pred0[0,0]) > 10, f"GP output 1 should be much larger than output 0: pred0={pred0[0,0]}, pred1={pred1[0,0]}"
        
        # Check that output 0 >> output 2 (should be ~100x larger)  
        assert abs(pred0[0,0] / pred2[0,0]) > 10, f"GP output 0 should be much larger than output 2: pred0={pred0[0,0]}, pred2={pred2[0,0]}"
        
        # Check that predictions are in reasonable ranges based on training data
        assert 5 < pred0[0,0] < 100, f"Output 0 prediction should be in range [5,100]: {pred0[0,0]}"
        assert 500 < pred1[0,0] < 10000, f"Output 1 prediction should be in range [500,10000]: {pred1[0,0]}"
        assert 0.05 < pred2[0,0] < 1.0, f"Output 2 prediction should be in range [0.05,1.0]: {pred2[0,0]}"


class TestOutputSelectionIntegration:
    """Integration tests for output selection across different scenarios."""
    
    def setup_method(self):
        """Set up test data."""
        self.params = [ContinuousVariable(min=0.0, max=1.0)]
        
    def test_output_selection_consistency(self):
        """Test that output selection is consistent between SMT_GP and SOOGO_GP."""
        # Create dataset with 2 outputs that have a clear mathematical relationship
        x_data = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_data = np.array([[x[0], x[0]**2] for x in x_data])  # y0 = x, y1 = x^2
        
        dataset = DatasetBase(params=self.params, n_fidelity=1, n_out=2)
        dataset.add_known_samples(x_data, y_data)
        
        # Train SMT and SOOGO on same outputs
        smt_gp0 = SMT_GP(dataset, i_output=0)
        smt_gp1 = SMT_GP(dataset, i_output=1)
        
        soogo_gp0 = SOOGO_GP(dataset, soogo_kwargs={'tune_alpha': False}, i_output=0)
        soogo_gp1 = SOOGO_GP(dataset, soogo_kwargs={'tune_alpha': False}, i_output=1)
        
        # Train all models
        smt_gp0.train(dataset)
        smt_gp1.train(dataset)
        soogo_gp0.train(dataset)
        soogo_gp1.train(dataset)
        
        # Test predictions
        x_test = np.array([[0.4]])
        
        smt_pred0 = smt_gp0.predict_values(x_test)[0,0]
        smt_pred1 = smt_gp1.predict_values(x_test)[0,0]
        soogo_pred0 = soogo_gp0.predict_values(x_test)[0,0]
        soogo_pred1 = soogo_gp1.predict_values(x_test)[0,0]
        
        # Both should predict that output 1 > output 0 (since x^2 > x for x > 1, but x^2 < x for 0 < x < 1)
        # At x=0.4, we expect y0≈0.4 and y1≈0.16, so y1 < y0
        assert smt_pred1 < smt_pred0, f"SMT: Expected y1 < y0 at x=0.4, got y0={smt_pred0}, y1={smt_pred1}"
        assert soogo_pred1 < soogo_pred0, f"SOOGO: Expected y1 < y0 at x=0.4, got y0={soogo_pred0}, y1={soogo_pred1}"