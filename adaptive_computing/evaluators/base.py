import numpy as np

class BaseEvaluator():
    """
    A base class for evaluating points using a specified evaluation function.
    
    Attributes:
        _eval_function (callable): The function used to evaluate points.
        n_in (int): The number of input dimensions.
        
    Methods:
        evaluate_points(points): Evaluates a set of points.
        evaluate_point(point): Evaluates a single point.
    """
    
    def __init__(self, evaluate_function, n_in=1):
        """
        Initializes the BaseEvaluator with the specified evaluation function and input dimensions.
        
        Args:
            evaluate_function (callable): The function used to evaluate points.
            n_in (int): The number of input dimensions. Defaults to 1.
        """
        self._eval_function = evaluate_function
        self.n_in = n_in

    def evaluate_points(self, points):
        """
        Evaluates a set of points using the evaluation function.
        
        Args:
            points (N samples, N input dimension): A 2D array of points to evaluate.
        
        Returns:
            (N samples, N output dimension): A 2D array of evaluated values.
        
        Raises:
            ValueError: If the points array is not 2D or does not have the correct number of input dimensions.
        """
        points = np.asarray(points)
        self._validate_points(points)

        y = np.zeros((points.shape[0], 1))
        for i, point in enumerate(points):
            y[i] = self.evaluate_point(point)

        return y
    
    def evaluate_point(self, point):
        """
        Evaluates a single point using the evaluation function.
        
        Args:
            point (N input dimension): A 1D array representing a point to evaluate.
        
        Returns:
            y (N output dimension): A 1D array of evaluated values.
        
        Raises:
            ValueError: If the point does not have the correct number of input dimensions.
        """
        point = np.atleast_1d(np.asarray(point))
        self._validate_point(point)
        y = np.atleast_1d(self._eval_function(point))

        return y
    
    def _validate_point(self, point):
        """
        Validates a single point to ensure it has the correct number of input dimensions.
        
        Args:
            point (N input dimension): A 1D array representing a point to validate.
        
        Raises:
            ValueError: If the point does not have the correct number of input dimensions.
        """
        if point.shape[0] != self.n_in:
            print("Point has incorrect number of input dimensions")
            print(f"Point shape: {point.shape}")
            print(f"Target input dimension: [{self.n_in}]")
            raise ValueError

    def _validate_points(self, points):
        """
        Validates a set of points to ensure it is a 2D array with the correct number of input dimensions.
        
        Args:
            points (N samples, N input dimension): A 2D array of points to validate.
        
        Raises:
            ValueError: If the points array is not 2D or does not have the correct number of input dimensions.
        """
        if points.ndim != 2:
            print("Points array must be 2 dimensional")
            raise ValueError
        if points.shape[1] != self.n_in:
            print("Point array has incorrect number of input dimensions")
            print(f"Point array shape: {points.shape}")
            print(f"Target shape: [N_samples, {self.n_in}]")
            raise ValueError
