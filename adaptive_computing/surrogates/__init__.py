from adaptive_computing.surrogates.base import SurrogateModelBase
from adaptive_computing.surrogates.smt import SMTGP, ConstrainedSMTGP

def surrogate_initializer(s,dataset):
    if s == 'SMT':
        return SMTGP(dataset=dataset)
