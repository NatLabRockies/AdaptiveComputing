import numpy as np

class ContinuousVariable():
    """
    A class representing a continuous variable with a specified range.
    
    Attributes:
        min (float): The minimum value of the variable.
        max (float): The maximum value of the variable.
        type (str): The type of the variable, set to 'continuous'.
        limits (list): A list containing the minimum and maximum values.
    """
    
    def __init__(self, min, max):
        """
        Initializes the ContinuousVariable with specified minimum and maximum values.
        
        Args:
            min (float): The minimum value of the variable.
            max (float): The maximum value of the variable.
        """
        self.min = min
        self.max = max
        self.type = 'continuous'
        self.limits = [self.min, self.max]


class OrderedVariable():
    """
    A class representing an ordered discrete variable (e.g., integer) with a specified range.
    
    Attributes:
        min_val (int): The minimum value of the variable.
        max_val (int): The maximum value of the variable.
        type (str): The type of the variable, set to 'ordered'.
        limits (list): A list containing the minimum and maximum values.
    """
    
    def __init__(self, min_val, max_val):
        """
        Initializes the OrderedVariable with specified minimum and maximum values.
        
        Args:
            min_val (int): The minimum value of the variable.
            max_val (int): The maximum value of the variable.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.type = 'ordered'
        self.limits = [self.min_val, self.max_val]


class CategoricalVariable():
    """
    A class representing a categorical variable with a specified set of categories.
    
    Attributes:
        categories (list): The list of possible categorical values.
        type (str): The type of the variable, set to 'categorical'.
        limits (list): A list of indices corresponding to the categories.
    """
    
    def __init__(self, categories):
        """
        Initializes the CategoricalVariable with specified categories.
        
        Args:
            categories (list): The list of possible categorical values.
        """
        self.categories = categories
        self.type = 'categorical'
        self.limits = list(range(len(categories)))  # Encoded as indices
