"""
Test suite for dataset masking functionality.

Tests the capability of DatasetBase and HeroDataset to handle NaN and 
out-of-bounds values with mask_ignore behavior.
"""

import pytest
import numpy as np
from adaptive_computing.datasets.base import DatasetBase
from adaptive_computing.datasets.variables import ContinuousVariable
from adaptive_computing.surrogates.smt_gp import SMT_GP


class TestDatasetMasking:
    """Test cases for dataset masking functionality."""
    
    def test_nan_masking_basic(self):
        """Test that DatasetBase correctly masks NaN values."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add data with NaN values
        x_test = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_test = np.array([[10.0], [np.nan], [30.0], [40.0]])  # Second point has NaN
        
        dataset.add_samples(x_test, y_test)
        
        # Verify mask correctly identifies valid points (sample-level)
        expected_mask = np.array([True, False, True, True])  # Second point should be masked
        actual_mask = np.all(dataset._unmasked_data[0], axis=1)  # Sample valid if all outputs valid
        assert np.array_equal(actual_mask, expected_mask)
        
        # Verify unmasked data contains only valid points 
        x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
        expected_valid_x = x_test[expected_mask]
        expected_valid_y = y_test[expected_mask]
        assert np.array_equal(x_unmasked, expected_valid_x)
        assert np.allclose(y_unmasked, expected_valid_y, equal_nan=False)
    
    def test_oob_masking_basic(self):
        """Test that DatasetBase correctly masks out-of-bounds values."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, y_bounds=(0.0, 50.0), oob_behavior='mask_ignore')
        
        # Add data with out-of-bounds values
        x_test = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_test = np.array([[10.0], [100.0], [30.0], [-5.0]])  # Second and fourth points are OOB
        
        dataset.add_samples(x_test, y_test)
        
        # Verify that OOB points are masked (sample-level)
        expected_mask = np.array([True, False, True, False])  # Second and fourth points should be masked
        sample_mask = np.all(dataset._unmasked_data[0], axis=1)
        assert np.array_equal(sample_mask, expected_mask)
    
    def test_multidimensional_nan_masking(self):
        """Test NaN masking with multi-dimensional y_data."""
        params = [ContinuousVariable(min=0.0, max=10.0), ContinuousVariable(min=0.0, max=10.0)]  
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add data with NaN in multi-dimensional output
        x_test = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        y_test = np.array([[10.0], [np.nan], [30.0], [40.0]])  # Second point has NaN
        
        dataset.add_samples(x_test, y_test)
        
        # Verify mask correctly identifies valid points (sample-level)
        expected_mask = np.array([True, False, True, True])
        sample_mask = np.all(dataset._unmasked_data[0], axis=1)
        assert np.array_equal(sample_mask, expected_mask)
    
    def test_mask_copy_functionality(self):
        """Test the unified masking functionality."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add data with some masked points
        x_test = np.array([[1.0], [2.0], [3.0]])
        y_test = np.array([[10.0], [np.nan], [30.0]]) 
        dataset.add_samples(x_test, y_test)
        
        # Test that unmasked data excludes NaN points
        x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
        assert x_unmasked.shape[0] == 2  # Should have 2 valid points
        assert not np.any(np.isnan(y_unmasked))  # No Nان values in unmasked data
    
    def test_no_nans_in_unmasked_data(self):
        """Test that unmasked data contains no NaN values."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add training data with some NaN points
        x_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_train = np.array([[10.0], [np.nan], [30.0], [np.nan], [50.0]])  # 2nd and 4th points are NaN
        dataset.add_samples(x_train, y_train)
        
        # Get unmasked data
        x_valid, y_valid = dataset.get_unmasked_data(0)
        
        # Verify no NaN values in unmasked data
        assert not np.any(np.isnan(y_valid))
        assert len(x_valid) == 3  # Should have 3 valid points
        assert len(y_valid) == 3
    
    def test_fail_behavior_compatibility(self):
        """Test that 'fail' behavior still works as expected."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='fail')  # Default behavior
        
        # Try to add data with NaN - should raise ValueError
        x_test = np.array([[1.0], [2.0]])
        y_test = np.array([[10.0], [np.nan]])
        
        with pytest.raises(ValueError, match="NaN values detected"):
            dataset.add_samples(x_test, y_test)


class TestSurrogateMasking:
    """Test cases for surrogate training with masked datasets."""
    
    def test_surrogate_training_with_masked_data(self):
        """Test that surrogate models can train with masked dataset."""
        # Create dataset with some masked points
        params = [ContinuousVariable(min=0.0, max=10.0), ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add training data with NaN points
        x_train = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
        y_train = np.array([[1.0], [4.0], [9.0], [16.0], [25.0], [np.nan]])  # Last point has NaN
        dataset.add_samples(x_train, y_train)
        
        # Verify masking worked
        # Verify that 5 out of 6 points are valid (1 masked)
        sample_mask = np.all(dataset._unmasked_data[0], axis=1)
        assert np.sum(sample_mask) == 5  # 5 valid points
        assert np.sum(~sample_mask) == 1  # 1 masked point
        
        # Test surrogate training with masking
        surrogate = SMT_GP(dataset)
        surrogate.train(dataset)
        
        # Test prediction works
        x_test = np.array([[2.5, 2.5]])
        y_pred = surrogate.predict_values(x_test)
        
        assert y_pred is not None
        assert not np.any(np.isnan(y_pred))
        assert y_pred.shape == (1, 1)
    
    def test_surrogate_unmasked_data_extraction(self):
        """Test that surrogate gets only unmasked data for training."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add data with masked points
        x_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([[1.0], [np.nan], [9.0], [16.0]])
        dataset.add_samples(x_train, y_train)
        
        # Get unmasked data directly
        x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
        
        # Verify only valid data is returned
        assert len(x_unmasked) == 3
        assert len(y_unmasked) == 3
        assert not np.any(np.isnan(y_unmasked))
        
        # Expected unmasked data (points 0, 2, 3)
        expected_x = np.array([[1.0], [3.0], [4.0]])
        expected_y = np.array([[1.0], [9.0], [16.0]])
        
        assert np.array_equal(x_unmasked, expected_x)
        assert np.array_equal(y_unmasked, expected_y)


class TestMaskingEdgeCases:
    """Test edge cases and error conditions for masking."""
    
    def test_all_data_masked(self):
        """Test behavior when all data points are masked."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add data where all points are NaN
        x_test = np.array([[1.0], [2.0], [3.0]])
        y_test = np.array([[np.nan], [np.nan], [np.nan]])
        dataset.add_samples(x_test, y_test)
        
        # Verify all points are masked
        # All data should be masked since all have NaN
        sample_mask = np.all(dataset._unmasked_data[0], axis=1)
        assert not np.any(sample_mask)
        
        # Unmasked data should be empty arrays
        x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
        assert len(x_unmasked) == 0
        assert len(y_unmasked) == 0
    
    def test_no_masked_data(self):
        """Test behavior when no data is masked."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, nan_behavior='mask_ignore')
        
        # Add data with no NaN values
        x_test = np.array([[1.0], [2.0], [3.0]])
        y_test = np.array([[10.0], [20.0], [30.0]])
        dataset.add_samples(x_test, y_test)
        
        # Verify no points are masked
        # All should be valid (no masking)
        sample_mask = np.all(dataset._unmasked_data[0], axis=1)
        assert np.all(sample_mask)
        
        # Unmasked data should be identical to original data
        x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
        assert np.array_equal(x_unmasked, x_test)
        assert np.array_equal(y_unmasked, y_test)
    
    def test_combined_nan_and_oob_masking(self):
        """Test masking when both NaN and OOB values are present."""
        params = [ContinuousVariable(min=0.0, max=10.0)]
        dataset = DatasetBase(params, y_bounds=(0.0, 50.0), nan_behavior='mask_ignore', oob_behavior='mask_ignore')
        
        # Add data with both NaN and OOB values
        x_test = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_test = np.array([[10.0], [np.nan], [100.0], [40.0]])  # NaN at index 1, OOB at index 2
        dataset.add_samples(x_test, y_test)
        
        # Verify both points are masked
        expected_unmasked = np.array([[True], [False], [False], [True]])  # Only indices 0 and 3 are valid
        assert np.array_equal(dataset._unmasked_data[0], expected_unmasked)


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])