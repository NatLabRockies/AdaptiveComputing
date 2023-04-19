#########################################################
# viz.py
# There are two types of visualizations:
# plot- 
# animation-
# For each type, there are 3 subtypes:
# 1D- for n_dim=1 parameters: plot of obj vs the 1 design param
# 2D- for n_dim=2 parameters: 1 param vs the other param with color indicating the obj func value
# ND- for arbitrary n_dim parameters: radial plot of the distance from the optimal param vector vs 
# the angle (dot product) from the optimal parameter vector with color indicating the obj func value
import sys
import numpy as np 
import matplotlib.pyplot as plt
from .acq_func import EI
import matplotlib.image as mpimg
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import cm
from . import utils
#########################################################
# Validate input plot types and set up paths for animations
def viz_init(options,n_dim):
    # validate the selected visualizations are compatible with the number of design parameters
    if options.animation_1d or options.plot_1d:
        if n_dim != 1:
            raise Exception('options.animation_1d and options.plot_1d should be False unless n_dim=1')
    if options.animation_2d or options.plot_2d:
        if n_dim != 2:
            raise Exception('options.animation_2d and options.plot_2d should be False unless n_dim=2')
        
    # create output directory
    if options.plot_1d or options.plot_2d or options.plot_nd or options.animation_1d or options.animation_2d or options.animation_nd:
        from pathlib import Path
        Path(options.output_dir).mkdir(parents=True, exist_ok=True)
        plt.ioff()

    return
#########################################################
# After each iteration, one frame of the animation is written 
def viz_animate(options,xlimits,funcs,gpr,x_data,y_data,n_init,k):
    # just plot the highest fidelity level
    ndoe = n_init[-1]
    if options.animation_1d:
        X_plot = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 10000)).T
        Y_plot = np.zeros_like(X_plot)
        for i in range(len(X_plot)):
            Y_plot[i] = funcs[-1](X_plot[i])
        Y_GP_plot = gpr.predict_values(X_plot)
        Y_GP_plot_var  =  gpr.predict_variances(X_plot)
        Y_EI_plot = -EI(gpr,X_plot,np.min(y_data[-1]))
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        # if options.acq_func == 'LCB' or options.acq_func == 'SBO':
        #     ei, = ax.plot(X_plot,Y_EI_plot,color='red')
        # else:    
        #     ax1 = ax.twinx()
        #     ei, = ax1.plot(X_plot,Y_EI_plot,color='red')
        true_fun, = ax.plot(X_plot,Y_plot)
        data, = ax.plot(x_data[-1][0:k+ndoe],y_data[-1][0:k+ndoe],linestyle='',marker='o',color='orange')
        opt, = ax.plot(x_data[-1][k+ndoe],y_data[-1][k+ndoe],linestyle='',marker='*',color='r')
        gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
        sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
        sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
        un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
        ind_best = np.argmin(y_data[-1][:ndoe+k])
        est = ax.scatter(x_data[-1][ind_best],y_data[-1][ind_best],s=100,marker='s',color='b')
        lines = [true_fun,data,gp,un_gp,opt,est]
        ax.set_title('$x \sin{x}$ function')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(lines,['True function','Data','GPR prediction','99 % confidence','Next point to evaluate','Current estimate of optimum'])
        plt.savefig(options.output_dir + ('/frame_1D_%d' %k))
        plt.close(fig)
    
    if options.animation_2d:
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        n_data = len(y_data[-1])
        plt.scatter(x_data[-1][:ndoe,0],x_data[-1][:ndoe,1],s=20,marker='x',c=y_data[-1][:ndoe],cmap=cm.coolwarm,label='Initial DOE')
        plt.scatter(x_data[-1][ndoe:,0],x_data[-1][ndoe:,1],s=20,marker='o',c=y_data[-1][ndoe:],cmap=cm.coolwarm,label='Added points')
        plt.scatter(x_data[-1][n_data-1,0],x_data[-1][n_data-1,1],s=60,marker='s',facecolors='none',edgecolors='g',label='Next point to evaluate')
        ind_best = np.argmin(y_data[-1])
        plt.scatter(x_data[-1][ind_best,0],x_data[-1][ind_best,1],s=100,marker='*',color='m',label='Current estimate of optimum')
        plt.title('Samples during EGO algorithm')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.legend()
        plt.savefig(options.output_dir + ('/frame_2D_%d' %k))
        plt.close(fig)

    if options.animation_nd:
        raise Exception('options.animation_nd is not supported. Use options.plot_nd instead.')

    return
#########################################################
# After all iterations are complete make final plots and make any finishing touches
def viz_finalize(options,xlimits,funcs,gpr,x_data,y_data,n_init,ind_best):
    # just plot the highest fidelity level
    ndoe = n_init[-1]
    if options.plot_1d:
        x_plot = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 10000)).T
        y_plot = np.zeros_like(x_plot)
        for i in range(len(x_plot)):
            y_plot[i] = funcs[-1](x_plot[i])
        y_gp_plot = gpr.predict_values(x_plot)
        y_gp_plot_var  =  gpr.predict_variances(x_plot)
        y_ei_plot = -EI(gpr,x_plot,np.min(y_data[-1]))
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        true_fun, = ax.plot(x_plot,y_plot)
        n_data = len(x_data[-1])
        data_init, = ax.plot(x_data[-1][0:ndoe],y_data[-1][0:ndoe],linestyle='',marker='o',color='orange')
        data, = ax.plot(x_data[-1][ndoe:n_data],y_data[-1][ndoe:n_data],linestyle='',marker='o',color='k')
        opt, = ax.plot(x_data[-1][ind_best],y_data[-1][ind_best],linestyle='',marker='*',color='r')
        gp, = ax.plot(x_plot,y_gp_plot,linestyle='--',color='g')
        sig_plus = y_gp_plot+3*np.sqrt(y_gp_plot_var)
        sig_moins = y_gp_plot-3*np.sqrt(y_gp_plot_var)
        un_gp = ax.fill_between(x_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
        lines = [true_fun,data_init,data,gp,un_gp,opt]
        ax.set_title('$x \sin{x}$ function')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(lines,['True function','Initial samples','Additional samples','GPR prediction','99 % confidence','Optimum found'])
        plt.savefig(options.output_dir + ('/final_1D'))
        plt.close(fig)
        
    if options.plot_2d:
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        plt.scatter(x_data[-1][:ndoe,0],x_data[-1][:ndoe,1],s=20,marker='x',c=y_data[-1][:ndoe],cmap=cm.coolwarm,label='Initial DOE')
        sm = plt.scatter(x_data[-1][ndoe:,0],x_data[-1][ndoe:,1],s=20,marker='o',c=y_data[-1][ndoe:],cmap=cm.coolwarm,label='Added points')
        plt.scatter(x_data[-1][ind_best,0],x_data[-1][ind_best,1],s=100,marker='*',color='m',label='Optimum found')
        plt.title('Samples during EGO algorithm')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.colorbar(sm,label='Objective function')
        plt.legend()
        plt.savefig(options.output_dir + ('/final_2D'))
        plt.close(fig)
    
    if options.plot_nd:
        fig = plt.figure(figsize=[10,10])
        radius = np.zeros_like(y_data[-1])
        color = np.zeros_like(y_data[-1]) # the iteration in which this data point was collected
        n_dim = len(xlimits)
        max_dist = np.zeros([1,n_dim])
        for i in range(n_dim):
            max_dist[0][i] = xlimits[i][-1]-xlimits[i][0]
        for i in range(len(x_data[-1])):
            x1 = x_data[-1][i,:]
            x2 = x_data[-1][ind_best,:]
            radius[i] = np.linalg.norm((x1-x2)/(max_dist))
            if i < ndoe:
                color[i] = 0
            else:
                color[i] = i-ndoe+1
        plt.scatter(radius[:ndoe,0],y_data[-1][:ndoe],s=20,marker='x',c=color[:ndoe],cmap=cm.coolwarm,label='Initial DOE')
        sm = plt.scatter(radius[ndoe:,0],y_data[-1][ndoe:],s=20,marker='o',c=color[ndoe:],cmap=cm.coolwarm,label='Added points')
        plt.scatter(radius[ind_best,0],y_data[-1][ind_best],s=60,marker='*',facecolors='none',edgecolors='g',label='Optimum found')
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
def viz_show_plots(options):
    print('Displaying plots and animations in', options.output_dir)
    if options.plot_1d:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(options.output_dir + ('/final_1D') + '.png')
        im = plt.imshow(image_pt)
        plt.show()

    if options.plot_2d:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(options.output_dir + ('/final_2D') + '.png')
        im = plt.imshow(image_pt)
        plt.show()

    if options.plot_nd:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(options.output_dir + ('/final_ND') + '.png')
        im = plt.imshow(image_pt)
        plt.show()
    
    if options.animation_1d:
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

    if options.animation_2d:
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
        
    if options.animation_nd:
        pass # XXX implement this

    return
#########################################################
