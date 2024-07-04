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
