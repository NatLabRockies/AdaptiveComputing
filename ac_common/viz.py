#########################################################
# viz.py
# There are two types of visualizations:
# plot- 
# animation-
# For each type, there are 3 subtypes:
# 1D- for ndim=1 parameters: plot of obj vs the 1 design param
# 2D- for ndim=2 parameters: 1 param vs the other param with color indicating the obj func value
# ND- for arbitrary ndim parameters: radial plot of the distance from the optimal param vector vs 
# the angle (dot product) from the optimal parameter vector with color indicating the obj func value
import sys
import numpy as np 
import matplotlib.pyplot as plt
from .acqFunc import EI
import matplotlib.image as mpimg
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import cm
from . import utils
#########################################################
# Validate input plot types and set up paths for animations
def init(options,ndim):
    print('Initializing plots...') 
    # validate the selected visualizations are compatible with the number of design parameters
    if options.animation_1D or options.plot_1D:
        if ndim != 1:
            raise Exception('options.animation_1D and options.plot_1D should be False unless ndim=1')
    if options.animation_2D or options.plot_2D:
        if ndim != 2:
            raise Exception('options.animation_2D and options.plot_2D should be False unless ndim=2')
        
    # create output directory
    if options.plot_1D or options.plot_2D or options.plot_ND or options.animation_1D or options.animation_2D or options.animation_ND:
        from pathlib import Path
        Path(options.output_dir).mkdir(parents=True, exist_ok=True)
        plt.ioff()

    return
#########################################################
# After each iteration, one frame of the animation is written 
def animate(options,xlimits,func,gpr,x_data,y_data,f_min_k,ndoe,k):
    if options.animation_1D:
        X_plot = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 10000)).T
        Y_plot = np.zeros_like(X_plot)
        for i in range(len(X_plot)):
            Y_plot[i] = func(X_plot[i])
        Y_GP_plot = gpr.predict_values(X_plot)
        Y_GP_plot_var  =  gpr.predict_variances(X_plot)
        Y_EI_plot = -EI(gpr,X_plot,f_min_k)
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        # if options.acqFunc == 'LCB' or options.acqFunc == 'SBO':
        #     ei, = ax.plot(X_plot,Y_EI_plot,color='red')
        # else:    
        #     ax1 = ax.twinx()
        #     ei, = ax1.plot(X_plot,Y_EI_plot,color='red')
        true_fun, = ax.plot(X_plot,Y_plot)
        data, = ax.plot(x_data[0:k+ndoe],y_data[0:k+ndoe],linestyle='',marker='o',color='orange')
        opt, = ax.plot(x_data[k+ndoe],y_data[k+ndoe],linestyle='',marker='*',color='r')
        gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
        sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
        sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
        un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
        ind_best = np.argmin(y_data[:ndoe+k])
        est = ax.scatter(x_data[ind_best],y_data[ind_best],s=100,marker='s',color='b')
        lines = [true_fun,data,gp,un_gp,opt,est]
        ax.set_title('$x \sin{x}$ function')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(lines,['True function','Data','GPR prediction','99 % confidence','Next point to evaluate','Current estimate of optimum'])
        plt.savefig(options.output_dir + ('/frame_1D_%d' %k))
        plt.close(fig)
    
    if options.animation_2D:
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        n_data = len(y_data)
        plt.scatter(x_data[:ndoe,0],x_data[:ndoe,1],s=20,marker='x',c=y_data[:ndoe],cmap=cm.coolwarm,label='Initial DOE')
        plt.scatter(x_data[ndoe:,0],x_data[ndoe:,1],s=20,marker='o',c=y_data[ndoe:],cmap=cm.coolwarm,label='Added points')
        plt.scatter(x_data[n_data-1,0],x_data[n_data-1,1],s=60,marker='s',facecolors='none',edgecolors='g',label='Next point to evaluate')
        ind_best = np.argmin(y_data)
        plt.scatter(x_data[ind_best,0],x_data[ind_best,1],s=100,marker='*',color='m',label='Current estimate of optimum')
        plt.title('Samples during EGO algorithm')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.legend()
        plt.savefig(options.output_dir + ('/frame_2D_%d' %k))
        plt.close(fig)

    if options.animation_ND:
        raise Exception('options.animation_ND is not supported. Use options.plot_ND instead.')

    return
#########################################################
# After all iterations are complete make final plots and make any finishing touches
def finalize(options,xlimits,func,gpr,x_data,y_data,f_min_k,ndoe,ind_best):
    print('Finalize ...') 
    if options.plot_1D:
        X_plot = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 10000)).T
        Y_plot = np.zeros_like(X_plot)
        for i in range(len(X_plot)):
            Y_plot[i] = func(X_plot[i])
        Y_GP_plot = gpr.predict_values(X_plot)
        Y_GP_plot_var  =  gpr.predict_variances(X_plot)
        Y_EI_plot = -EI(gpr,X_plot,f_min_k)
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        true_fun, = ax.plot(X_plot,Y_plot)
        n_data = len(x_data)
        data_init, = ax.plot(x_data[0:ndoe],y_data[0:ndoe],linestyle='',marker='o',color='orange')
        data, = ax.plot(x_data[ndoe:n_data],y_data[ndoe:n_data],linestyle='',marker='o',color='k')
        opt, = ax.plot(x_data[ind_best],y_data[ind_best],linestyle='',marker='*',color='r')
        gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
        sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
        sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
        un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
        lines = [true_fun,data_init,data,gp,un_gp,opt]
        ax.set_title('$x \sin{x}$ function')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(lines,['True function','Initial samples','Additional samples','GPR prediction','99 % confidence','Optimum found'])
        plt.savefig(options.output_dir + ('/final_1D'))
        plt.close(fig)
        
    if options.plot_2D:
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        plt.scatter(x_data[:ndoe,0],x_data[:ndoe,1],s=20,marker='x',c=y_data[:ndoe],cmap=cm.coolwarm,label='Initial DOE')
        sm = plt.scatter(x_data[ndoe:,0],x_data[ndoe:,1],s=20,marker='o',c=y_data[ndoe:],cmap=cm.coolwarm,label='Added points')
        plt.scatter(x_data[ind_best,0],x_data[ind_best,1],s=100,marker='*',color='m',label='Optimum found')
        plt.title('Samples during EGO algorithm')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.colorbar(sm,label='Objective function')
        plt.legend()
        plt.savefig(options.output_dir + ('/final_2D'))
        plt.close(fig)
    
    if options.plot_ND:
        fig = plt.figure(figsize=[10,10])
        radius = np.zeros_like(y_data)
        color = np.zeros_like(y_data) # the iteration in which this data point was collected
        for i in range(len(x_data)):
            x1 = x_data[i,:]
            x2 = x_data[ind_best,:]
            radius[i] = np.linalg.norm(x1-x2)
            if i < ndoe:
                color[i] = 0
            else:
                color[i] = i-ndoe+1
        plt.scatter(radius[:ndoe,0],y_data[:ndoe],s=20,marker='x',c=color[:ndoe],cmap=cm.coolwarm,label='Initial DOE')
        sm = plt.scatter(radius[ndoe:,0],y_data[ndoe:],s=20,marker='o',c=color[ndoe:],cmap=cm.coolwarm,label='Added points')
        plt.scatter(radius[ind_best,0],y_data[ind_best],s=60,marker='*',facecolors='none',edgecolors='g',label='Optimum found')
        plt.title('Objective function vs distance from optimum in parameter vector space')
        plt.colorbar(sm,label='Iteration')
        plt.xlabel('$||\\vec{x}-\\vec{x}_{opt}||_2$')
        plt.ylabel('Objective function')
        plt.legend()
        plt.savefig(options.output_dir + ('/final_ND'))
        plt.close(fig)

    return
#########################################################
# Show the plots and play the animations
def show_plots(options):
    print('Displaying plots and animations in', options.output_dir)
    if options.plot_1D:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(options.output_dir + ('/final_1D') + '.png')
        im = plt.imshow(image_pt)
        plt.show()

    if options.plot_2D:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(options.output_dir + ('/final_2D') + '.png')
        im = plt.imshow(image_pt)
        plt.show()

    if options.plot_ND:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(options.output_dir + ('/final_ND') + '.png')
        im = plt.imshow(image_pt)
        plt.show()
    
    if options.animation_1D:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ims = []
        for k in range(options.n_iter):
            image_pt = mpimg.imread(options.output_dir + ('/frame_1D_%d' %k) + '.png')
            im = plt.imshow(image_pt)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims,interval=1000)
        # display a javascript animation if this is running in a jupyter notebook
        if utils.is_notebook():
            display(HTML(ani.to_jshtml()))
        else:
            plt.show() # display a movie
        writergif = animation.PillowWriter(fps=1) 
        ani.save(options.output_dir + '/movie_1D' + '.gif', writer=writergif, dpi=500)

    if options.animation_2D:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ims = []
        for k in range(options.n_iter):
            image_pt = mpimg.imread(options.output_dir + ('/frame_2D_%d' %k) + '.png')
            im = plt.imshow(image_pt)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims,interval=1000)
        # display a javascript animation if this is running in a jupyter notebook
        if utils.is_notebook():
            display(HTML(ani.to_jshtml()))
        else:
            plt.show() # display a movie
        writergif = animation.PillowWriter(fps=1) 
        ani.save(options.output_dir + '/movie_2D' + '.gif', writer=writergif, dpi=500)
        
    if options.animation_ND:
        pass # XXX implement this

    return
#########################################################
