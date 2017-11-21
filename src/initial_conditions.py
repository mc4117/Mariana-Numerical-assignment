#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare

This file contains the functions which define the four initial conditions that
are considered by this code.

"""

import numpy as np
import matplotlib.pyplot as plt
import math


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


def initialconditions_spike(nx, xmin = 0, xmax = 1, plot = True):
    """
    xmin: minimum value of x on grid
    xmax: maximum value of x on grid
    nx: number of space steps
    plot: if this variable is True then the initial conditions will be plotted, but it False then no plot will be produced
    """
    x = np.linspace(xmin,xmax,nx+1) # want the extra point at the boundary for plot but in reality h[0] and h[nx] are equal
    midpoint = math.floor(math.floor(len(x)/2)/2)*2 
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is zero everywhere apart from one point at the centre where it is one
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = h_spike(i, midpoint)
    
    # plot these initial conditions
    if plot == True:
        fig1, ax = plt.subplots()
        ax.plot(x, initialh, 'g-', label = 'Initial h conditions')
        ax.plot(x, initialu, 'r--', label = 'Initial u conditions')
        ax.legend(loc = 'best')
        ax.set_xlabel("x")
        ax.set_title("Initial Condition with spike in h")
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([-0.1, 1.1])
        
        # add space between the title and the plot
        #plt.rcParams['axes.titlepad'] = 20 
        fig1.savefig("initial_condition_spike.png")
        fig1.show()
        
    return initialu, initialh, x



def initialconditions_cosbell(nx, xmin = 0, xmax = 1, plot = True):
    """
    xmin: minimum value of x on grid
    xmax: maximum value of x on grid
    nx: number of space steps
    plot: if this variable is True then the initial conditions will be plotted, but it False then no plot will be produced
    """
    x = np.linspace(xmin,xmax,nx+1) # want the extra point at the boundary but in reality h[0] and h[nx] are equal
    
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h has a bump in the centre and is surrounded by zero either side
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = h_cosbell(x[i])
        
    # plot these initial conditions
    if plot == True:
        fig1, ax = plt.subplots()
        
        ax.plot(x, initialh, 'g-', label = 'Initial h conditions')
        ax.plot(x, initialu, 'r--', label = 'Initial u conditions')
        ax.legend(loc = 'best')
        ax.set_xlabel("x")
        ax.set_title("Initial Condition where h has a bump in the centre")
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([-0.1, 1.1])
        
        # add space between the title and the plot
        #plt.rcParams['axes.titlepad'] = 20 
        fig1.savefig("initial_condition_cosbell.png")
        fig1.show()
    return initialu, initialh, x

def initialconditions_cos(nx, xmin = 0, xmax = 1, plot = True):
    """
    xmin: minimum value of x on grid
    xmax: maximum value of x on grid
    nx: number of space steps
    plot: if this variable is True then the initial conditions will be plotted, but it False then no plot will be produced
    """
    x = np.linspace(xmin,xmax,nx+1) # want the extra point at the boundary but in reality h[0] and h[nx] are equal
    
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is cos(x)
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = math.cos(x[i])
        
    # plot these initial conditions
    if plot == True:
        fig1, ax1 = plt.subplots()
        ax1.plot(x, initialh, 'g-', label = 'Initial h conditions')
        ax1.plot(x, initialu, 'r--', label = 'Initial u conditions')
        ax1.legend(loc = 'best')
        ax1.set_xlabel("x")
        ax1.set_title("Initital Condition where h is cos(x)")
        ax1.set_xlim = ([xmin, xmax])
        ax1.set_ylim([-1.1, 1.1])
        
        # add space between the title and the plot
        #plt.rcParams['axes.titlepad'] = 20 
        fig1.savefig("initial_condition_cos.png")
        fig1.show()
    return initialu, initialh, x

def initialconditions_cossin(nx, xmin = -math.pi, xmax = math.pi, plot = True):
    """
    xmin: minimum value of x on grid
    xmax: maximum value of x on grid
    nx: number of space steps
    plot: if this variable is True then the initial conditions will be plotted, but it False then no plot will be produced
    """
    x = np.linspace(xmin,xmax,nx+1) # want the extra point at the boundary but in reality h[0] and h[nx] are equal
    
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions
    for i in range(len(x)):
        initialu[i] = math.cos(x[i]) - math.sin(x[i])
        initialh[i] = math.cos(x[i]) + math.sin(x[i])
        
    # plot these initial conditions
    if plot == True:
        fig1, ax1 = plt.subplots()
        ax1.plot(x, initialh, 'g-', label = 'Initial h conditions')
        ax1.plot(x, initialu, 'r--', label = 'Initial u conditions')
        ax1.legend(loc = 'best')
        ax1.set_xlabel("x")
        ax1.set_title("Initital Condition where h is cos(x) + sin(x) and u is cos(x) - sin(x)")
        ax1.set_xlim = ([xmin, xmax])
        ax1.set_ylim([-1.5, 1.5])
        
        # add space between the title and the plot
        #plt.rcParams['axes.titlepad'] = 20 
        fig1.savefig("initial_condition_cossin.png")
        fig1.show()
    return initialu, initialh, x


