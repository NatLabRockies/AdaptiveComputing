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
def viz_init(viz_ops,n_dim):
    # validate the selected visualizations are compatible with the number of design parameters
    if viz_ops.animation_1d or viz_ops.plot_1d:
        if n_dim != 1:
            raise Exception('viz_ops.animation_1d and viz_ops.plot_1d should be False unless n_dim=1')
    if viz_ops.animation_2d or viz_ops.plot_2d:
        if n_dim != 2:
            raise Exception('viz_ops.animation_2d and viz_ops.plot_2d should be False unless n_dim=2')

    # create output directory
    if viz_ops.plot_1d or viz_ops.plot_2d or viz_ops.plot_nd or viz_ops.animation_1d or viz_ops.animation_2d or viz_ops.animation_nd:
        from pathlib import Path
        Path(viz_ops.output_dir).mkdir(parents=True, exist_ok=True)
        plt.ioff()

    return
#########################################################
# After each iteration, one frame of the animation is written 
def viz_animate(model,viz_ops,frame_id):
    [x_opt, y_opt] = model.find_min()
    n_fl = len(model.y_data)
    if n_fl == 1:
        ndoe = len(model.y_data[0])-(frame_id+1)
        if viz_ops.animation_1d:
            x_plot = np.atleast_2d(np.linspace(model.xlimits_num[0][0], model.xlimits_num[0][1], 10000)).T
            y_plot = np.zeros_like(x_plot)
            for i in range(len(x_plot)):
                y_plot[i] = model.funcs[0](x_plot[i])
            y_gp_plot = model.gprs[-1].predict_values(x_plot)
            y_gp_plot_var  =  model.gprs[-1].predict_variances(x_plot)
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            #y_ei_plot = -EI(model.gprs[-1],x_plot,np.min(model.y_data[0]))
            # if options.acq_func == 'LCB' or options.acq_func == 'SBO':
            #     ei, = ax.plot(x_plot,y_ei_plot,color='red')
            # else:    
            #     ax1 = ax.twinx()
            #     ei, = ax1.plot(x_plot,y_ei_plot,color='red')
            plt.plot(x_plot,y_plot,label='True function')
            plt.scatter(x_opt,y_opt,70,marker='s',color='blue',label='Optimum found')
            plt.scatter(model.x_data[0][0:ndoe],model.y_data[0][0:ndoe],marker='^',color='black',label='Initial samples')
            plt.scatter(model.x_data[0][ndoe:-1],model.y_data[0][ndoe:-1],marker='o',color='orange',label='Additional samples')
            plt.scatter(model.x_data[0][-1],model.gprs[-1].predict_values(model.x_data[0][-1]),80,marker='>',color='magenta',label='Next point to evaluate')
            plt.plot(x_plot,y_gp_plot,linestyle='--',color='g',label='model.gprs[-1] prediction')
            sig_plus = y_gp_plot+3*np.sqrt(y_gp_plot_var)
            sig_moins = y_gp_plot-3*np.sqrt(y_gp_plot_var)
            plt.fill_between(x_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.25,color='g',label='99 % confidence')
            plt.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.savefig(viz_ops.output_dir + ('/frame_1D_%d' %frame_id))
            plt.close(fig)
        
        if viz_ops.animation_2d:
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            plt.scatter(model.x_data[0][:ndoe,0],model.x_data[0][:ndoe,1],s=20,marker='x',c='black',label='Initial DOE')
            sm = plt.scatter(model.x_data[0][ndoe:,0],model.x_data[0][ndoe:,1],s=20,marker='o',c=model.y_data[0][ndoe:],cmap=cm.coolwarm,label='Added points')
            plt.scatter(model.x_data[0][-1,0],model.x_data[0][-1,1],s=60,marker='s',facecolors='none',edgecolors='g',label='Next point to evaluate')
            plt.scatter(x_opt[0],x_opt[1],s=100,marker='*',color='m',label='Current estimate of optimum')
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.colorbar(sm,label='Objective function')
            plt.legend()
            plt.savefig(viz_ops.output_dir + ('/frame_2D_%d' %frame_id))
            plt.close(fig)

        if viz_ops.animation_nd:
            raise Exception('viz_ops.animation_nd is not supported. Use viz_ops.plot_nd instead.')
    elif n_fl == 2:
        if viz_ops.animation_1d:
            x_plot = np.atleast_2d(np.linspace(model.xlimits_num[0][0], model.xlimits_num[0][1], 10000)).T
            y_plot = np.zeros_like(x_plot)
            for i in range(len(x_plot)):
                y_plot[i] = model.funcs[-1](x_plot[i])
            y_gp_plot = model.gprs[-1].predict_values(x_plot)
            y_gp_plot_var  =  model.gprs[-1].predict_variances(x_plot)
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            #y_ei_plot = -EI(model.gprs[-1],x_plot,np.min(model.y_data[-1]))
            # if options.acq_func == 'LCB' or options.acq_func == 'SBO':
            #     ei, = ax.plot(x_plot,y_ei_plot,color='red')
            # else:    
            #     ax1 = ax.twinx()
            #     ei, = ax1.plot(x_plot,y_ei_plot,color='red')
            plt.plot(x_plot,y_plot,label='True function')
            plt.scatter(x_opt,y_opt,70,marker='s',color='blue',label='Optimum found')
            plt.scatter(model.x_data[0],model.y_data[0],marker='^',color='black',label='LF samples')
            plt.scatter(model.x_data[1],model.y_data[1],marker='o',color='orange',label='HF samples')
            plt.plot(x_plot,y_gp_plot,linestyle='--',color='g',label='model.gprs[-1] prediction')
            sig_plus = y_gp_plot+3*np.sqrt(y_gp_plot_var)
            sig_moins = y_gp_plot-3*np.sqrt(y_gp_plot_var)
            plt.fill_between(x_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.25,color='g',label='99 % confidence')
            plt.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.savefig(viz_ops.output_dir + ('/frame_1D_%d' %frame_id))
            plt.close(fig)
        
        if viz_ops.animation_2d:
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            plt.scatter(model.x_data[0][:,0],model.x_data[0][:,1],s=20,marker='^',color='k',label='LF samples')
            sm = plt.scatter(model.x_data[1][:,0],model.x_data[1][:,1],s=20,marker='o',c=model.y_data[1],cmap=cm.coolwarm,label='HF samples')
            plt.scatter(x_opt[0],x_opt[1],s=100,marker='*',color='m',label='Current estimate of optimum')
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.colorbar(sm,label='Objective function')
            plt.legend()
            plt.savefig(viz_ops.output_dir + ('/frame_2D_%d' %frame_id))
            plt.close(fig)

        if viz_ops.animation_nd:
            raise Exception('viz_ops.animation_nd is not supported. Use viz_ops.plot_nd instead.')
    else:
        raise Exception('Animation for 3+ fidelity levels is not supported but the user can access model.model.x_data and model.model.y_data for plotting.')

    return
#########################################################
# After all iterations are complete make final plots and make any finishing touches
def viz_finalize(model,viz_ops,frame_id):
    [x_opt, y_opt] = model.find_min()
    n_fl = len(model.y_data)
    if n_fl == 1:
        ndoe = len(model.y_data[0])-(frame_id+1)
        if viz_ops.plot_1d:
            x_plot = np.atleast_2d(np.linspace(model.xlimits_num[0][0], model.xlimits_num[0][1], 10000)).T
            y_plot = np.zeros_like(x_plot)
            for i in range(len(x_plot)):
                y_plot[i] = model.funcs[0](x_plot[i])
            y_gp_plot = model.gprs[-1].predict_values(x_plot)
            y_gp_plot_var  =  model.gprs[-1].predict_variances(x_plot)
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            plt.plot(x_plot,y_plot,label='True function')
            plt.scatter(x_opt,y_opt,70,marker='s',color='blue',label='Optimum found')
            plt.scatter(model.x_data[0][0:ndoe],model.y_data[0][0:ndoe],marker='^',color='black',label='Initial samples')
            plt.scatter(model.x_data[0][ndoe:],model.y_data[0][ndoe:],marker='o',color='orange',label='Additional samples')
            plt.plot(x_plot,y_gp_plot,linestyle='--',color='g',label='model.gprs[-1] prediction')
            sig_plus = y_gp_plot+3*np.sqrt(y_gp_plot_var)
            sig_moins = y_gp_plot-3*np.sqrt(y_gp_plot_var)
            plt.fill_between(x_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.25,color='g',label='99 % confidence')
            plt.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.savefig(viz_ops.output_dir + ('/final_1D'))
            plt.close(fig)
            
        if viz_ops.plot_2d:
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            plt.scatter(model.x_data[0][:ndoe,0],model.x_data[0][:ndoe,1],s=20,marker='x',c='black',label='Initial DOE')
            sm = plt.scatter(model.x_data[0][ndoe:,0],model.x_data[0][ndoe:,1],s=20,marker='o',c=model.y_data[0][ndoe:],cmap=cm.coolwarm,label='Added points')
            plt.scatter(x_opt[0],x_opt[1],s=100,marker='*',color='m',label='Optimum found')
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.colorbar(sm,label='Objective function')
            plt.legend()
            plt.savefig(viz_ops.output_dir + ('/final_2D'))
            plt.close(fig)
        
        if viz_ops.plot_nd:
            fig = plt.figure(figsize=[10,10])
            radius = np.zeros_like(model.y_data[0])
            color = np.zeros_like(model.y_data[0]) # the iteration in which this data point was collected
            n_dim = len(model.xlimits_num)
            max_dist = np.zeros([1,n_dim])
            for i in range(n_dim):
                max_dist[0][i] = model.xlimits_num[i][-1]-model.xlimits_num[i][0]
            for i in range(len(model.x_data[0])):
                x1 = model.x_data[0][i,:]
                radius[i] = np.linalg.norm((x1-x_opt)/(max_dist))
                if i < ndoe:
                    color[i] = 0
                else:
                    color[i] = i-ndoe+1
            plt.scatter(radius[:ndoe,0],model.y_data[0][:ndoe],s=20,marker='x',color='black',label='Initial DOE')
            sm = plt.scatter(radius[ndoe:,0],model.y_data[0][ndoe:],s=20,marker='o',c=color[ndoe:],cmap=cm.coolwarm,label='Added points')
            plt.scatter(0,y_opt,s=100,marker='*',facecolors='none',edgecolors='m',label='Optimum found')
            plt.title('Objective function vs distance from optimum in parameter vector space')
            plt.colorbar(sm,label='Iteration')
            plt.xlabel('$||\\vec{x}-\\vec{x}_{opt}||_2$')
            plt.ylabel('Objective function')
            plt.legend()
            plt.savefig(viz_ops.output_dir + ('/final_ND'))
            plt.close(fig)
    elif n_fl == 2:
        if viz_ops.plot_1d:
            x_plot = np.atleast_2d(np.linspace(model.xlimits_num[0][0], model.xlimits_num[0][1], 10000)).T
            y_plot = np.zeros_like(x_plot)
            for i in range(len(x_plot)):
                y_plot[i] = model.funcs[-1](x_plot[i])
            y_gp_plot = model.gprs[-1].predict_values(x_plot)
            y_gp_plot_var  =  model.gprs[-1].predict_variances(x_plot)
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            plt.plot(x_plot,y_plot,label='True function')
            plt.scatter(x_opt,y_opt,70,marker='s',color='blue',label='Optimum found')
            plt.scatter(model.x_data[0],model.y_data[0],marker='^',color='k',label='LF samples')
            plt.scatter(model.x_data[1],model.y_data[1],marker='o',color='orange',label='HF samples')
            plt.plot(x_plot,y_gp_plot,linestyle='--',color='g',label='model.gprs[-1] prediction')
            sig_plus = y_gp_plot+3*np.sqrt(y_gp_plot_var)
            sig_moins = y_gp_plot-3*np.sqrt(y_gp_plot_var)
            plt.fill_between(x_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.25,color='g',label='99 % confidence')
            plt.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.savefig(viz_ops.output_dir + ('/final_1D'))
            plt.close(fig)
            
        if viz_ops.plot_2d:
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            plt.scatter(model.x_data[0][:,0],model.x_data[0][:,1],s=20,marker='^',color='k',label='LF samples')
            sm = plt.scatter(model.x_data[1][:,0],model.x_data[1][:,1],s=20,marker='o',c=model.y_data[1],cmap=cm.coolwarm,label='HF samples')
            plt.scatter(x_opt[0],x_opt[1],s=100,marker='*',color='m',label='Optimum found')
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.colorbar(sm,label='Objective function')
            plt.legend()
            plt.savefig(viz_ops.output_dir + ('/final_2D'))
            plt.close(fig)
        
        if viz_ops.plot_nd:
            fig = plt.figure(figsize=[10,10])
            radius_fl0 = np.zeros_like(model.y_data[0])
            radius_fl1 = np.zeros_like(model.y_data[1])
            color = np.zeros_like(model.y_data[1]) # the iteration in which this data point was collected
            n_dim = len(model.xlimits_num)
            max_dist = np.zeros([1,n_dim])
            for i in range(n_dim):
                max_dist[0][i] = model.xlimits_num[i][-1]-model.xlimits_num[i][0]
            for i in range(len(model.x_data[0])):
                x1 = model.x_data[0][i,:]
                radius_fl0[i] = np.linalg.norm((x1-x_opt)/(max_dist))
            plt.scatter(radius_fl0[:,0],model.y_data[0],s=20,marker='^',color='k',label='LF samples')
            for i in range(len(model.x_data[1])):
                x1 = model.x_data[1][i,:]
                radius_fl1[i] = np.linalg.norm((x1-x_opt)/(max_dist))
                color[i] = i+1
            sm = plt.scatter(radius_fl1[:,0],model.y_data[1],s=20,marker='o',c=color,cmap=cm.coolwarm,label='HF samples')
            plt.scatter(0,y_opt,s=100,marker='*',facecolors='none',edgecolors='m',label='Optimum found')
            plt.title('Objective function vs distance from optimum in parameter vector space')
            plt.colorbar(sm,label='Iteration')
            plt.xlabel('$||\\vec{x}-\\vec{x}_{opt}||_2$')
            plt.ylabel('Objective function')
            plt.legend()
            plt.savefig(viz_ops.output_dir + ('/final_ND'))
            plt.close(fig)
    else:
        raise Exception('Plotting for 3+ fidelity levels is not supported but the user can access model.model.x_data and model.model.y_data for plotting.')

    return
#########################################################
# Show the plots and play the animations
def viz_show_plots(viz_ops,n_frames=None):
    print('Displaying plots and animations in', viz_ops.output_dir)
    if viz_ops.plot_1d:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(viz_ops.output_dir + ('/final_1D') + '.png')
        im = plt.imshow(image_pt)
        plt.show()

    if viz_ops.plot_2d:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(viz_ops.output_dir + ('/final_2D') + '.png')
        im = plt.imshow(image_pt)
        plt.show()

    if viz_ops.plot_nd:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        image_pt = mpimg.imread(viz_ops.output_dir + ('/final_ND') + '.png')
        im = plt.imshow(image_pt)
        plt.show()
    
    if viz_ops.animation_1d:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ims = []
        for k in range(n_frames):
            image_pt = mpimg.imread(viz_ops.output_dir + ('/frame_1D_%d' %k) + '.png')
            im = plt.imshow(image_pt)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims,interval=3000)
        # display a javascript animation if this is running in a jupyter notebook
        if utils.is_notebook():
            display(HTML(ani.to_jshtml())) # noqa: F821
        else:
            plt.show() # display a movie
        writergif = animation.PillowWriter(fps=1) 
        ani.save(viz_ops.output_dir + '/movie_1D' + '.gif', writer=writergif, dpi=500)

    if viz_ops.animation_2d:
        fig = plt.figure(figsize=[10,10])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ims = []
        for k in range(n_frames):
            image_pt = mpimg.imread(viz_ops.output_dir + ('/frame_2D_%d' %k) + '.png')
            im = plt.imshow(image_pt)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims,interval=1000)
        # display a javascript animation if this is running in a jupyter notebook
        if utils.is_notebook():
            display(HTML(ani.to_jshtml())) # noqa: F821
        else:
            plt.show() # display a movie
        writergif = animation.PillowWriter(fps=1) 
        ani.save(viz_ops.output_dir + '/movie_2D' + '.gif', writer=writergif, dpi=500)
        
    if viz_ops.animation_nd:
        pass # XXX implement this

    return
#########################################################
