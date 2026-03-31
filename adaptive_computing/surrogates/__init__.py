from adaptive_computing.surrogates.base import SurrogateModelBase

def surrogate_initializer(s,dataset):
    if s == 'SMT' or s == 'SMT_GP':
        # Lazy import to avoid loading TensorFlow unless needed
        from adaptive_computing.surrogates.smt import SMT_GP
        return SMT_GP(dataset=dataset)
    elif s == 'SOOGO' or s == 'SOOGO_GP':
        from adaptive_computing.surrogates.soogo import SOOGO_GP
        return SOOGO_GP(dataset=dataset)
    elif s == 'TFMELT' or s == 'TFMELT_BNN':
        from adaptive_computing.surrogates.tfmelt import TFMELT_BNN
        return TFMELT_BNN(dataset=dataset)
    else:
        raise ValueError(f"Unknown surrogate type: {s}")

def __getattr__(name):
    """Lazy loading of surrogate classes to avoid importing heavy dependencies unless needed."""
    if name == 'SMT_GP':
        from adaptive_computing.surrogates.smt import SMT_GP
        return SMT_GP
    elif name == 'SOOGO_GP':
        from adaptive_computing.surrogates.soogo import SOOGO_GP
        return SOOGO_GP
    elif name == 'TFMELT_BNN':
        from adaptive_computing.surrogates.tfmelt import TFMELT_BNN
        return TFMELT_BNN
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
