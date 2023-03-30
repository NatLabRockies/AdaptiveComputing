#########################################################
# acquisitionFunctions.py
# Available acquisition functions:
# - SBO (surrogate based optimization): directly using the prediction of the surrogate model ($\mu$)
# - LCB (Lower Confidence bound): using the confidence interval : $\mu -3 \times \sigma$
# - EI for expected Improvement (EGO)
# Note: be sure to add any user defined aquisition function to getAcqFunc
import numpy as np 
from scipy.stats import norm
#########################################################
# expected improvement: 
def EI(GP,points,f_min):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)
    if var.size == 1 and var == 0.0:  # can be use only if one point is computed
        return 0.0
    ei = args1 + args2
    return ei
#########################################################
#surrogate Based optimization: min the Surrogate model by using the mean mu
def SBO(GP,points):
    res = GP.predict_values(points)
    return res
#########################################################
#lower confidence bound optimization: minimize by using mu - 3*sigma
def LCB(GP,points):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    res = pred-3.*np.sqrt(var)
    return res
#########################################################
#maximal standard deviation: tries to minimize the max standard deviation
def MSD(GP,points):
    var = GP.predict_variances(points)
    res = -np.sqrt(var)
    return res
#########################################################
def getAcqFunc(IC,gpr,f_min_k):
    if IC == 'EI':
        obj_k = lambda x: -EI(gpr,np.atleast_2d(x),f_min_k)[:,0]
    elif IC =='SBO':
        obj_k = lambda x: SBO(gpr,np.atleast_2d(x))
    elif IC == 'LCB':
        obj_k = lambda x: LCB(gpr,np.atleast_2d(x))
    elif IC == 'MSD':
        obj_k = lambda x: MSD(gpr,np.atleast_2d(x))
    return obj_k
#########################################################