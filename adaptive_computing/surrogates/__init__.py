from adaptive_computing.surrogates.base import SurrogateModelBase
from adaptive_computing.surrogates.smt import SMTWrapper, ConstrainedSMTWrapper

def surrogate_initializer(s,dataset):
    if s == 'SMT':
        return SMTWrapper(dataset=dataset)