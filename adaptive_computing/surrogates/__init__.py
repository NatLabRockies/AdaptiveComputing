from adaptive_computing.surrogates.base import SurrogateModelBase
from adaptive_computing.surrogates.smt import SMTGP, ConstrainedSMTGP
from adaptive_computing.surrogates.soogo import SOOGOGP

def surrogate_initializer(s,dataset):
    if s == 'SMT':
        return SMTGP(dataset=dataset)
    elif s == 'SOOGO':
        return SOOGOGP(dataset=dataset)
