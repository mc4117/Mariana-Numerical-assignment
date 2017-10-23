#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt




def plot_multiple_iterations(initialconditions, nx, number_iterations, number_plotted, numerical_method, plotparameterrange, xmin = 0, xmax = 1):
    """This function plots the solution of the numerical method at various time iterations

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    number_iterations:  number of time steps for the last iteration that is plotted by this function
    number_plotted:     number of different iterations that are plotted (ie. number of curves on graph)
    numerical_method:   function which specifies the numerical method used
    plotparameterrange: used to specify the colour and linestyle of lines plotted
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    """
    
    # initialize plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    # set the timesteps that will be plotted
    timerange = np.linspace(0, number_iterations, (number_plotted + 1))
     
    colorrange = plotparameterrange[0]
    linestylerange = plotparameterrange[1]

    # iterate through and plot chosen timesteps - note we do not plot the timestep 0 as this is just the initial condition
    for i in range(len(timerange[1:])):
        u, h, x = numerical_method(initialconditions, nx, nt = timerange[1+i])
        ax1.plot(x, u, c = colorrange[i], ls = linestylerange[i], label = 'u after ' + str(int(timerange[1+i])) + ' timesteps')  
        ax2.plot(x, h, c = colorrange[i], ls = linestylerange[i], label = 'h after ' + str(int(timerange[1+i])) + ' timesteps')

    # for reference plot the x-meshgrid
    ax1.scatter(x,np.zeros_like(x), c = 'black', s = 10)
    ax2.scatter(x,np.zeros_like(x), c = 'black', s = 10)
    
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")
    ax1.legend(loc = 'best')

    ax2.set_xlim([xmin, xmax])
    ax2.set_xlabel("x")
    ax2.legend(loc = 'best') 

    # add space between the title and the plot
    plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
        
    return ax1, ax2

def plot_multiple_c(initialconditions,  numerical_method, crange, colorrange, nx = 60, nt = 100, xmin = 0, xmax = 1):
    """This function plots the solution of the numerical method for various different courant numbers

    initial conditions: function which specifies the initial conditions for the system 
    numerical_method:   function which specifies the numerical method used
    crange:             range of different courant numbers used
    colorrange:         used to specify the colour lines plotted
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    """
    # initialize plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
        # iterate through different courant numbers and plot results
    for i in range(len(crange[1:])):
        u, h, x = numerical_method(initialconditions, nx, nt, c = crange[i + 1])

        ax1.plot(x, u, c = colorrange[i], label = 'c=' + str(crange[i+1]))
        ax1.set_xlim([xmin,xmax])
        ax1.set_xlabel("x")
        ax1.legend(loc = 'best')
    
        ax2.plot(x, h, c = colorrange[i], label = 'c=' + str(crange[i + 1]))
        ax2.set_xlim([xmin,xmax])
        ax2.set_xlabel("x")
        ax2.legend(loc = 'best')

    # add space between the title and the plot
    plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
        
    return ax1, ax2
