#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare

Various different initial conditions for shallow water equations

"""

import numpy as np
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


def initialconditions_spike(x):
    """
    x:    array of x meshgrid where we would like to calculate the inital condition
    """
    
    # calculate the midpoint where spike is situated
    midpoint = math.floor(len(x)/2)
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is zero everywhere apart from one point at the centre where it is one
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = h_spike(i, midpoint)
    
    return initialu, initialh



def initialconditions_cosbell(x):
    """
    x:    array of x meshgrid where we would like to calculate the inital condition
    """

    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h has a bump in the centre and is surrounded by zero either side
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = h_cosbell(x[i])
        
    return initialu, initialh

def initialconditions_cos(x):
    """
    x:    array of x meshgrid where we would like to calculate the inital condition
    """

    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions such that u is zero everywhere and h is cos(x)
    for i in range(len(x)):
        initialu[i] = 0
        initialh[i] = math.cos(x[i])
        
    return initialu, initialh

def initialconditions_cossin(x):
    """
    x:    array of x meshgrid where we would like to calculate the inital condition
    """
    
    # initialize initial u and initial h
    initialu = np.zeros(len(x)).astype(float)
    initialh = np.zeros(len(x)).astype(float)
    
    # set the initial conditions
    for i in range(len(x)):
        initialu[i] = math.cos(x[i]) - math.sin(x[i])
        initialh[i] = math.cos(x[i]) + math.sin(x[i])
    
    return initialu, initialh


