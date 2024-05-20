class ContinuousVariable():
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.type = 'continuous'
        self.limits = [self.min, self.max]