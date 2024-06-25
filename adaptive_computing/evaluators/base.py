import numpy as np

class BaseEvaluator():
    def __init__(self, evaluate_function, n_in=1):
        self._eval_function = evaluate_function
        self.n_in = n_in

    def evaluate_points(self, points):
        points = np.asarray(points)
        self._validate_points(points)

        y = np.zeros((points.shape[0],1))
        points = np.asarray(points)
        for i, point in enumerate(points):
            y[i] = self.evaluate_point(point)

        return y
    
    def evaluate_point(self, point):
        point = np.atleast_1d(np.asarray(point))
        self._validate_point(point)
        y = np.atleast_1d(self._eval_function(point))

        return y
    
    def _validate_point(self, point):
        if point.shape[0] != self.n_in:
            print("Point has incorrect number of input dimensions")
            print(f"Point shape: {point.shape}")
            print(f"Target input dimension: [{self.n_in}]")
            raise(ValueError)

    def _validate_points(self, points):
            if points.ndim != 2:
                print("Points array must be 2 dimensional ")
                raise(ValueError)
            if points.shape[1] != self.n_in:
                print("Point array has incorrect number of input dimensions")
                print(f"Point array shape: {points.shape}")
                print(f"Target shape: [ N_samples, {self.n_in}]")
                raise(ValueError)
            

