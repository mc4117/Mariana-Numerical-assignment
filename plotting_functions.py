#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_multiple_iterations(initialconditions, nx, number_iterations, numerical_method, xmin = 0, xmax = 1):
    """This function plots the solution of the numerical method at various time iterations

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    number_iterations:  number of time steps for the last iteration that is plotted by this function
    numerical_method:   function which specifies the numerical method used
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    """
    
    # initialize plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    # set the colours to be used in the plot
    mymap = plt.get_cmap("YlOrRd")
    colorrange = mymap(np.r_[np.array([0.3, 0.55, 1]), np.array([0.3, 0.55, 1])])

    # set the linestyles that are used for each line in the plot
    linestylerange = np.array(['-', '--', '-'])

    # set the timesteps that will be plotted
    timerange = np.linspace(0, number_iterations, 4)

    # iterate through and plot chosen timesteps - note we do not plot the timestep 0 as this is just the initial condition
    for i in range(len(timerange[1:])):
        u, h, x1 = numerical_method(initialconditions, nx, nt = timerange[1+i])
        ax1.plot(x1, u, c = colorrange[i], ls = linestylerange[i], label = 'u after ' + str(int(timerange[1+i])) + ' timesteps')  
        ax2.plot(x1, h, c = colorrange[i], ls = linestylerange[i], label = 'h after ' + str(int(timerange[1+i])) + ' timesteps')

    # for reference plot the x-meshgrid
    ax1.scatter(x1,np.zeros_like(x1), c = 'black', s = 10)
    ax2.scatter(x1,np.zeros_like(x1), c = 'black', s = 10)
    
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")
    ax1.legend(loc = 'best')

    ax2.set_xlim([0,1])
    ax2.set_xlabel("x")
    ax2.legend(loc = 'best') 

    # add space between the title and the plot
    plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
        
    return ax1, ax2
