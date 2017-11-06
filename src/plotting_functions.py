#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt
import numerical_methods as nm
import initial_conditions as ic
import math



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
    timerange = np.linspace(0, number_iterations, (number_plotted + 1))
     
    colorrange = plotparameterrange[0]
    linestylerange = plotparameterrange[1]

    # iterate through and plot chosen timesteps - note we do not plot the timestep 0 as this is just the initial condition
    for i in range(len(timerange[1:])):
        u, h, x = numerical_method(initialconditions, nx, nt = timerange[1+i])
        ax2.plot(x, h, c = colorrange[i], ls = linestylerange[i], label = 'h after ' + str(int(timerange[1+i])) + ' timesteps')
        if staggered == True:
            ax1.plot((x + 1/(2*nx))[:-1], u[:-1], c = colorrange[i], ls = linestylerange[i], label = 'u after ' + str(int(timerange[1+i])) + ' timesteps')  
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
    #plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
        
    return fig1, fig2, ax1, ax2

def compare_results(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
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
    """
    
    
    # find u and h for each numerical method
    u_A_grid, h_A_grid, x1 = nm.A_grid_explicit(initialconditions, nx, nt, xmin, xmax, H, g, c)
    u_C_grid, h_C_grid, x2 = nm.C_grid_explicit(initialconditions, nx, nt, xmin, xmax, H, g, c)
    u_implicit, h_implicit, x3 = nm.implicit_method(initialconditions, nx, nt, xmin, xmax, H, g, c)
    u_semi_implicit, h_semi_implicit, x4 = nm.semi_implicit_method(initialconditions, nx, nt, xmin, xmax, H, g, c)

    # plot u found by 4 different methods
    fig1, ax1 = plt.subplots()
    ax1.plot(x1, u_A_grid, c = 'blue', label = "A-grid explicit")
    ax1.plot(x2, u_C_grid, c = 'green', label = "C-grid explicit")
    ax1.plot((x3 + 1/(2*nx))[:-1], u_implicit[:-1], c = 'red', label = "A-grid implicit")
    ax1.plot((x4 + 1/(2*nx))[:-1], u_semi_implicit[:-1], c ='orange', label = "C-grid semi-implicit")
    
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")

    # plot h found by 4 different methods
    fig2, ax2 = plt.subplots()
    ax2.plot(x1, h_A_grid, c = 'blue', label = "A-grid explicit")
    ax2.plot(x2, h_C_grid, c = 'green', label = "C-grid explicit")
    ax2.plot(x3, h_implicit, c = 'red', label = "A-grid implicit")
    ax2.plot(x4, h_semi_implicit, c = 'orange', label = "C-grid semi-implicit")
    
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel("x")

    return fig1, fig2, ax1, ax2, x1

def error_fn(nx, nt, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function compares the solutions of the 4 numerical methods studied for the initial condition that u = 0 everywhere 
        and h is cos(x) and finds the error between these solutions and the exact solution
        Note this function can only be used with this initial condition as otherwise the exact solution is incorrect.
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """
    # derive the width of the spacestep and timestep
    dx = (xmax - xmin)/nx
    dt = (c*dx)/math.sqrt(g*H)
    
    # find u and h for each numerical method
    u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_implicit, h_implicit, x3 = nm.implicit_method(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_semi_implicit, h_semi_implicit, x4 = nm.semi_implicit_method(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)

    # Note x1, x2, x3 and x4 are all the same variables as nx, nt, xmin and xmax are the same for all methods
    
    # construct exact solution on both colocated and staggered grid
    u_A_grid = np.zeros_like(x1)
    u_C_grid = np.zeros_like(x1[1:])
    h = np.zeros_like(x1)
    
    for i in range(len(x1)):
        u_A_grid[i] = math.sin(x1[i])*math.sin(dt*nt)
        h[i] = math.cos(x1[i])*math.cos(dt*nt)
        
    for i in range(len(x1) - 1):
        u_C_grid[i] = math.sin(x1[i]+ dx/2)*math.sin(dt*nt)
    
        
    # find error between exact solution and solution found by numerical method
    error_A_grid_u = (u_A_grid - u_A_grid_explicit)**2
    error_C_grid_u = (u_C_grid - u_C_grid_explicit[:-1])**2
    error_implicit_u = (u_A_grid - u_implicit)**2
    error_semi_implicit_u = (u_C_grid - u_semi_implicit[:-1])**2
    
    # plot error in u from 4 different methods
    fig1, ax1 = plt.subplots()
    ax1.plot(x1, error_A_grid_u, c = 'blue', label = "A-grid explicit")
    ax1.plot(x2[:-1], error_C_grid_u, c = 'green', label = "C-grid explicit")
    ax1.plot(x3, error_implicit_u, c = 'red', label = "A-grid implicit")
    ax1.plot(x4[:-1], error_semi_implicit_u, c ='orange', label = "C-grid semi-implicit")
    
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")
    
    error_A_grid_h = (h - h_A_grid_explicit)**2
    error_C_grid_h = (h - h_C_grid_explicit)**2
    error_implicit_h = (h - h_implicit)**2
    error_semi_implicit_h = (h - h_semi_implicit)**2

    # plot error in h from 4 different methods
    fig2, ax2 = plt.subplots()
    ax2.plot(x1, error_A_grid_h, c = 'blue', label = "A-grid explicit")
    ax2.plot(x2, error_C_grid_h, c = 'green', label = "C-grid explicit")
    ax2.plot(x3, error_implicit_h, c = 'red', label = "A-grid implicit")
    ax2.plot(x4, error_semi_implicit_h, c = 'orange', label = "C-grid semi-implicit")
    
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel("x")
    
    return fig1, fig2, ax1, ax2