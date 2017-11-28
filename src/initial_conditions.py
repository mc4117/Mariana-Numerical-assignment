#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare

Functions which calculate various different initial conditions for shallow water 
equations

"""

import numpy as np
import math

def initialconditions_spike(x):
    """calculate the initial velocity and height such that u is zero everywhere
       and h is zero everywhere apart from one point at the centre where it is one
       
       Inputs:
           x:        array of x-meshgrid on which inital condition is calculated
    
       Outputs:
           initialu: array containing initial values of u on x
           initialh: array containing initial values of h on x
    """
    
    # calculate the midpoint where spike is situated
    midpoint = math.floor(len(x)/2)
    
    # initialize arrays
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is zero 
    # everywhere apart from one point at the centre where it is one
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = h_spike(i, midpoint)
    
    return initialu, initialh



def initialconditions_cosbell(x):
    """calculate the initial velocity and height such that u is zero everywhere
       and h has a bump in the centre and is surrounded by zero either side
       
       Inputs:
           x:        array of x-meshgrid on which inital condition is calculated
    
       Outputs:
           initialu: array containing initial values of u on x
           initialh: array containing initial values of h on x
    """

    # initialize arrays
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h has a 
    # bump in the centre and is surrounded by zero either side
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = h_cosbell(x[i])
        
    return initialu, initialh

def initialconditions_cos(x):
    """calculate the initial velocity and height such that u is zero everywhere
       and h is cos(x)
       
       Inputs:
           x:        array of x-meshgrid on which inital condition is calculated
    
       Outputs:
           initialu: array containing initial values of u on x
           initialh: array containing initial values of h on x
    """

    # initialize arrays
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is cos(x)
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = math.cos(x[i])
        
    return initialu, initialh

def initialconditions_cossin(x):
    """calculate the initial velocity and height such that u is zero everywhere
       and h is cos(x) + sin(x)
       
       Inputs:
           x:        array of x-meshgrid on which inital condition is calculated
    
       Outputs:
           initialu: array containing initial values of u on x
           initialh: array containing initial values of h on x
    """
    
    # initialize arrays
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial condition where u is cos(x) - sin(x) and h is cos(x) + sin(x)
    for i in range(len(x)):
        initialu[i] = math.cos(x[i]) - math.sin(x[i])
        initialh[i] = math.cos(x[i]) + math.sin(x[i])
    
    return initialu, initialh


def h_spike(j, midpoint):
    """produces a curve which is zero everywhere except at midpoint where it is 
       one
       
       Inputs:
           j:         point at which function is evaluated
           midpoint:  middle point of meshgrid
               
        Outputs:
            y:        result of evaluating function at j"""
    if j == midpoint:
        y = 1
    else:
        y = 0
    return y

def h_cosbell(j):
    """produces a curve which has a bump in the centre and is surrounded by zero 
       either side       
       
       Inputs:
           j:         point at which function is evaluated
               
       Outputs:
            y:        result of evaluating function at j"""
    if 0.25 <= j <= 0.75:
        y = math.cos(2*(math.pi)*(j - 0.5))
    else:
        y = 0
    return y
