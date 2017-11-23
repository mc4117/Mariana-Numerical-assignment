#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt
import numerical_methods as nm
import time


def plot_multiple_iterations(initialconditions, nx, number_iterations, number_plotted, numerical_method, plotparameterrange, xmin = 0, xmax = 1, staggered = False):
    """This function plots the solution of the numerical method at various time iterations

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    number_iterations:  number of time steps for the last iteration that is plotted by this function
    number_plotted:     number of different iterations that are plotted (ie. number of curves on graph)
    numerical_method:   function which specifies the numerical method used
    plotparameterrange: used to specify the colour and linestyle of lines plotted
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    staggered:          if method is staggered then must plot u on a staggered grid
    """
    
    # initialize plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    # set the timesteps that will be plotted
    timerange = np.linspace(0, number_iterations, (number_plotted + 1)).astype('int')
     
    colorrange = plotparameterrange[0]
    linestylerange = plotparameterrange[1]

    # iterate through and plot chosen timesteps - note we do not plot the timestep 0 as this is just the initial condition
    for i in range(len(timerange[1:])):
        u, h, x = numerical_method(initialconditions, nx, nt = timerange[1+i])
        ax2.plot(x, h, c = colorrange[i], ls = linestylerange[i], label = 'h after ' + str(int(timerange[1+i])) + ' timesteps')
        if staggered == True:
            ax1.plot((x + (xmax-xmin)/(2*nx))[:-1], u[:-1], c = colorrange[i], ls = linestylerange[i], label = 'u after ' + str(int(timerange[1+i])) + ' timesteps')  
        else:
            ax1.plot(x, u, c = colorrange[i], ls = linestylerange[i], label = 'u after ' + str(int(timerange[1+i])) + ' timesteps')  
    
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
    #plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
        
    return fig1, fig2, ax1, ax2

def plot_multiple_c(initialconditions,  numerical_method, crange, colorrange, nx = 60, nt = 100, xmin = 0, xmax = 1, staggered = False):
    """This function plots the solution of the numerical method for various different courant numbers

    initial conditions: function which specifies the initial conditions for the system 
    numerical_method:   function which specifies the numerical method used
    crange:             range of different courant numbers used
    colorrange:         used to specify the colour lines plotted
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    staggered:          if this value is true then u is plotted on a staggered grid
    """
    # initialize plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    dx = (xmax - xmin)/nx
    
        # iterate through different courant numbers and plot results
    for i in range(len(crange[1:])):
        u, h, x = numerical_method(initialconditions, nx, nt, c = crange[i + 1])
        
        if staggered == True:
            ax1.plot(x + dx/2, u, c = colorrange[i], label = 'c=' + str(crange[i+1]))
            ax1.set_xlim([xmin,xmax])
            ax1.set_xlabel("x")
            ax1.legend(loc = 'best')
        else:
            ax1.plot(x, u, c = colorrange[i], label = 'c=' + str(crange[i+1]))
            ax1.set_xlim([xmin,xmax])
            ax1.set_xlabel("x")
            ax1.legend(loc = 'best')
    
        ax2.plot(x, h, c = colorrange[i], label = 'c=' + str(crange[i + 1]))
        ax2.set_xlim([xmin,xmax])
        ax2.set_xlabel("x")
        ax2.legend(loc = 'best')

    # add space between the title and the plot
    #plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
        
    return fig1, fig2, ax1, ax2

def compare_results(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1, timing = False):
    """This function compares the solutions of the 4 numerical methods studied for a given initial condition
    Note this function can be used with any initial condition
    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    timing:             if this is true then instead of returning the graph 
                        the function will return the running time for each scheme
    """
    
    
    # find u and h for each numerical method and time how long it takes to solve the scheme
    t0 = time.time()
    u_A_grid, h_A_grid, x1 = nm.A_grid_explicit(initialconditions, nx, nt, xmin, xmax, H, g, c)
    t1 = time.time()
    u_C_grid, h_C_grid, x2 = nm.C_grid_explicit(initialconditions, nx, nt, xmin, xmax, H, g, c)
    t2 = time.time()
    u_A_grid_implicit, h_A_grid_implicit, x3 = nm.A_grid_implicit_method(initialconditions, nx, nt, xmin, xmax, H, g, c)
    t3 = time.time()
    u_C_grid_implicit, h_C_grid_implicit, x4 = nm.C_grid_implicit_method(initialconditions, nx, nt, xmin, xmax, H, g, c)
    t4 = time.time()
    
    # plot u found by 4 different methods
    fig1, ax1 = plt.subplots()
    ax1.plot(x1, u_A_grid, c = 'blue', label = "A-grid explicit")
    ax1.plot(x2, u_C_grid, c = 'green', label = "C-grid explicit")
    ax1.plot((x3 + (xmax - xmin)/(2*nx))[:-1], u_A_grid_implicit[:-1], c = 'red', label = "A-grid implicit")
    ax1.plot((x4 + (xmax - xmin)/(2*nx))[:-1], u_C_grid_implicit[:-1], c ='orange', label = "C-grid implicit")
    
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")

    # plot h found by 4 different methods
    fig2, ax2 = plt.subplots()
    ax2.plot(x1, h_A_grid, c = 'blue', label = "A-grid explicit")
    ax2.plot(x2, h_C_grid, c = 'green', label = "C-grid explicit")
    ax2.plot(x3, h_A_grid_implicit, c = 'red', label = "A-grid implicit")
    ax2.plot(x4, h_C_grid_implicit, c = 'orange', label = "C-grid implicit")
    
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel("x")

    if timing == True:
        return t0, t1, t2, t3, t4
    else:
        return fig1, fig2, ax1, ax2, x1
