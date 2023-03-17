#from acquisitionFunctions import *
# optimalParams.py
def optimalParams(func, params, options):
    import numpy as np
    xlimits = np.array([[params.minVals[0],params.maxVals[0]]]) # assumes 1D 
    
    # Set up plot
    import matplotlib.pyplot as plt
    if options.animation:
        from pathlib import Path
        Path(options.animation_dir).mkdir(parents=True, exist_ok=True)
        plt.ioff()

        X_plot = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 10000)).T
        Y_plot = func(X_plot)

    from scipy.optimize import minimize
    from acquisitionFunctions import EI, SBO, LCB, getAcqFunc

    # Sample the objective function at three points. This is the training data.
    x_data = np.atleast_2d([xlimits[0][0],params.initVals[0],xlimits[0][1]]).T # assumes 1D
    y_data = func(x_data)

    # Define the Gaussian Process model (AKA Kriging model)
    from smt.surrogate_models import KPLS, KRG, KPLSK
    ndim = 1 # dimension of the problem 
    # The variable 'theta0' is a list of length ndim.
    gpr = KRG(theta0=[1e-2]*ndim,print_global = False) #, corr='squar_exp'

    # Iteratively select new sample points and update the GP according to the acquisition function
    for k in range(options.n_iter):
        x_start = np.atleast_2d(np.random.rand(20)*25).T
        f_min_k = np.min(y_data)
        gpr.set_training_values(x_data,y_data)
        gpr.train()
        obj_k = getAcqFunc(options.acqFunc,gpr,f_min_k)
        opt_all = np.array([minimize(lambda x: float(obj_k(x)), x_st, method='SLSQP', bounds=[(0,25)]) for x_st in x_start])
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]]
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success])
        ind_min = np.argmin(obj_success)
        opt = opt_success[ind_min]
        x_et_k = opt['x']
        y_et_k = func(x_et_k)
        y_data = np.atleast_2d(np.append(y_data,y_et_k)).T
        x_data = np.atleast_2d(np.append(x_data,x_et_k)).T

        # Plotting
        if options.animation == True:
            Y_GP_plot = gpr.predict_values(X_plot)
            Y_GP_plot_var  =  gpr.predict_variances(X_plot)
            Y_EI_plot = -EI(gpr,X_plot,f_min_k)
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            if options.acqFunc == 'LCB' or options.acqFunc == 'SBO':
                ei, = ax.plot(X_plot,Y_EI_plot,color='red')
            else:    
                ax1 = ax.twinx()
                ei, = ax1.plot(X_plot,Y_EI_plot,color='red')
            true_fun, = ax.plot(X_plot,Y_plot)
            data, = ax.plot(x_data[0:k+3],y_data[0:k+3],linestyle='',marker='o',color='orange')
            opt, = ax.plot(x_data[k+3],y_data[k+3],linestyle='',marker='*',color='r')
            gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
            sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
            sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
            un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
            lines = [true_fun,data,gp,un_gp,opt,ei]
            ax.set_title('$x \sin{x}$ function')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(lines,['True function','Data','GPR prediction','99 % confidence','Next point to Evaluate','Infill Criteria'])
            plt.savefig(options.animation_dir + ('/frame_%d' %k))
            plt.close(fig)

    ind_best = np.argmin(y_data)
    x_opt = x_data[ind_best]
    y_opt = y_data[ind_best]
    params.optVals = x_opt
    #print('Results : X = %s, Y = %s' %(x_opt,y_opt))
    return gpr
