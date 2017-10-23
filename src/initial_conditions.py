#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def flat_u(j):
    """produces a flat line at zero"""
    y = 0
    return y

def h_spike(j, midpoint):
    """produces a curve which is zero everywhere but h = 1 at one point in the centre"""
    if j == midpoint:
        y = 1
    else:
        y = 0
    return y

def h_cosbell(j):
    """produces a curve which has a bump in the centre and is surrounded by zero either side"""
    if 0.25 <= j <= 0.75:
        y = math.cos(2*(math.pi)*(j - 0.5))
    else:
        y = 0
    return y


def initialconditions_spike(nx, nt, xmin = 0, xmax = 1, plot = True):
    """
    xmin: minimum value of x on grid
    xmax: maximum value of x on grid
    nx: number of space steps
    nt: number of time steps
    plot: if this variable is True then the initial conditions will be plotted, but it False then no plot will be produced
    """
    x = np.linspace(xmin,xmax,nx+1) # want the extra point at the boundary for plot but in reality h[0] and h[nx] are equal
    midpoint = math.floor(math.floor(len(x)/2)/2)*2 
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is zero everywhere apart from one point at the centre where it is one
    for i in range(len(x)):
        initialu[i] = flat_u(i)
        initialh[i] = h_spike(i, midpoint)
    
    # plot these initial conditions
    if plot == True:
        fig1, ax = plt.subplots()
        ax.plot(x, initialh, 'g-', label = 'Initial h conditions')
        ax.plot(x, initialu, 'r--', label = 'Initial u conditions')
        ax.legend(loc = 'best')
        ax.set_xlabel("x")
        ax.set_title("Initial Condition where u is 0 everywhere and \n and h is zero everywhere apart from one point at the centre where it is one")
        
        # add space between the title and the plot
        plt.rcParams['axes.titlepad'] = 20 
        fig1.show()
        
    return initialu, initialh, midpoint, x



def initialconditions_cosbell(nx, nt, xmin = 0, xmax = 1, plot = True):
    """
    xmin: minimum value of x on grid
    xmax: maximum value of x on grid
    nx: number of space steps
    nt: number of time steps
    plot: if this variable is True then the initial conditions will be plotted, but it False then no plot will be produced
    """
    x = np.linspace(xmin,xmax,nx+1) # want the extra point at the boundary but in reality h[0] and h[nx] are equal
    
    midpoint = math.floor(math.floor(len(x)/2)/2)*2 # calculate midpoint to be used for the forcing term
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h has a bump in the centre and is surrounded by zero either side
    for i in range(len(x)):
        initialu[i] = flat_u(i)
        initialh[i] = h_cosbell(x[i])
        
    # plot these initial conditions
    if plot == True:
        fig1, ax = plt.subplots()
        
        ax.plot(x, initialh, 'g-', label = 'Initial h conditions')
        ax.plot(x, initialu, 'r--', label = 'Initial u conditions')
        ax.legend(loc = 'best')
        ax.set_xlabel("x")
        ax.set_title("Initial Condition where u is 0 everywhere and \n h has a bump in the centre and is surrounded by 0 either side")
    
        # add space between the title and the plot
        plt.rcParams['axes.titlepad'] = 20 
        fig1.show()
    return initialu, initialh, midpoint, x


