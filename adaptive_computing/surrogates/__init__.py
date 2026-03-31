from adaptive_computing.surrogates.base import SurrogateModelBase

def surrogate_initializer(s,dataset):
    if s == 'SMT' or s == 'SMT_GP':
        # Lazy import to avoid loading TensorFlow unless needed
        from adaptive_computing.surrogates.smt_gp import SMT_GP
        return SMT_GP(dataset=dataset)
    elif s == 'SOOGO' or s == 'SOOGO_GP':
        from adaptive_computing.surrogates.soogo_gp import SOOGO_GP
        return SOOGO_GP(dataset=dataset)
    elif s == 'TFMELT' or s == 'TFMELT_BNN':
        from adaptive_computing.surrogates.tfmelt_bnn import TFMELT_BNN
        return TFMELT_BNN(dataset=dataset)
    elif s == 'TFMELT_MDN':
        from adaptive_computing.surrogates.tfmelt_mdn import TFMELT_MDN
        return TFMELT_MDN(dataset=dataset)
    else:
        raise ValueError(f"Unknown surrogate type: {s}")

def __getattr__(name):
    """Lazy loading of surrogate classes to avoid importing heavy dependencies unless needed."""
    if name == 'SMT_GP':
        from adaptive_computing.surrogates.smt_gp import SMT_GP
        return SMT_GP
    elif name == 'SOOGO_GP':
        from adaptive_computing.surrogates.soogo_gp import SOOGO_GP
        return SOOGO_GP
    elif name == 'TFMELT_BNN':
        from adaptive_computing.surrogates.tfmelt_bnn import TFMELT_BNN
        return TFMELT_BNN
    elif name == 'TFMELT_MDN':
        from adaptive_computing.surrogates.tfmelt_mdn import TFMELT_MDN
        return TFMELT_MDN
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
