#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:07:42 2017

@author: mc4117
"""

# -*- coding: utf-8 -*-
import numerical_methods as nm
import initial_conditions as ic
import math
import numpy as np
import matplotlib.pyplot as plt

    
# results of all 4 methods


def error_fn_norms(nx_range, nt_range, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 1):
    """This function compares the solutions of the 4 numerical methods studied for the initial condition that u = 0 everywhere 
        and h is cos(x) and finds the Frobenius norm of the error between these solutions and the exact solution
        Note this function can only be used with this initial condition as otherwise the exact solution is incorrect.
    nx_range:           range of total number of space steps (in order to vary mesh size)
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """
    norm_A_grid_listu = np.zeros_like(nx_range).astype('float')
    norm_C_grid_listu = np.zeros_like(nx_range).astype('float')
    norm_implicit_listu = np.zeros_like(nx_range).astype('float')
    norm_semi_implicit_listu = np.zeros_like(nx_range).astype('float')

    norm_A_grid_listh = np.zeros_like(nx_range).astype('float')
    norm_C_grid_listh = np.zeros_like(nx_range).astype('float')
    norm_implicit_listh = np.zeros_like(nx_range).astype('float')
    norm_semi_implicit_listh = np.zeros_like(nx_range).astype('float')

    dx_list = np.zeros_like(nx_range).astype('float')
    

    
    # find u and h for each numerical method for a range of space mesh sizes 
    for j in range(len(nx_range)):
        nx = nx_range[j]
        nt = nt_range[j]
        # derive the width of the spacestep and timestep
        dx_list[j] = (xmax - xmin)/nx
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
            u_C_grid[i] = math.sin(x1[i]+ dx_list[j]/2)*math.sin(dt*nt)
    
        
        # find the Frobenius norm of the error between exact solution and solution found by numerical method
        norm_A_grid_listu[j] = np.linalg.norm((u_A_grid - u_A_grid_explicit))
        norm_C_grid_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_explicit[:-1]))
        norm_implicit_listu[j] = np.linalg.norm((u_A_grid - u_implicit))
        norm_semi_implicit_listu[j] = np.linalg.norm((u_C_grid - u_semi_implicit[:-1]))
    
        norm_A_grid_listh[j] = np.linalg.norm((h - h_A_grid_explicit))
        norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
        norm_implicit_listh[j] = np.linalg.norm((h - h_implicit))
        norm_semi_implicit_listh[j] = np.linalg.norm((h - h_semi_implicit))
        
    plt.plot(np.log(dx_list), np.log(norm_A_grid_listu), label = 'u')
    plt.plot(np.log(dx_list), np.log(norm_A_grid_listh), label = 'h')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.plot(np.log(dx_list), np.log(norm_C_grid_listu), label = 'u')
    plt.plot(np.log(dx_list), np.log(norm_C_grid_listh), label = 'h')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.plot(np.log(dx_list), np.log(norm_implicit_listu), label = 'u')
    plt.plot(np.log(dx_list), np.log(norm_implicit_listh), label = 'h')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.plot(np.log(dx_list), np.log(norm_semi_implicit_listu), label = 'u')
    plt.plot(np.log(dx_list), np.log(norm_semi_implicit_listh), label = 'h')
    plt.legend(loc = 'best')
    plt.show()
    
    gradient_A_grid_u = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listu),2)
    gradient_C_grid_u = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listu),2)
    gradient_implicit_u = np.polyfit(np.log(dx_list), np.log(norm_implicit_listu),2)
    gradient_semi_implicit_u = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listu),2)
    
    gradient_A_grid_h = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listh),2)
    gradient_C_grid_h = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listh),2)
    gradient_implicit_h = np.polyfit(np.log(dx_list), np.log(norm_implicit_listh),2)
    gradient_semi_implicit_h = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listh),2)
    
    print(gradient_A_grid_u, gradient_C_grid_u, gradient_implicit_u, gradient_semi_implicit_u, gradient_A_grid_h, gradient_C_grid_h, gradient_implicit_h, gradient_semi_implicit_h)
    
    return gradient_A_grid_u, gradient_C_grid_u, gradient_implicit_u, gradient_semi_implicit_u, gradient_A_grid_h, gradient_C_grid_h, gradient_implicit_h, gradient_semi_implicit_h

nx_range = range(20, 200, 20)
nt_range = np.zeros_like(nx_range).astype('int')
dx_list = np.zeros_like(nx_range).astype('float')
total_time = math.pi
xmin = -math.pi
xmax = math.pi
c = 1
g = 1
H = 1


for j in range(len(nx_range)):
    nx = nx_range[j]
    # derive the width of the spacestep and timestep
    dx_list[j] = (xmax - xmin)/nx
    dt = (c*dx_list[j])/math.sqrt(g*H)
    nt_range[j] = total_time/dt

gradient_A_grid_u, gradient_C_grid_u, gradient_implicit_u, gradient_semi_implicit_u, gradient_A_grid_h, gradient_C_grid_h, gradient_implicit_h, gradient_semi_implicit_h = error_fn_norms(nx_range, nt_range, xmin, xmax, H, g, c)

