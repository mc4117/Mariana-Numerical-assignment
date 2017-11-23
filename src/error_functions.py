#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt
import numerical_methods as nm
import initial_conditions as ic
import numbers
import math


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
    u_A_grid_implicit, h_A_grid_implicit, x3 = nm.A_grid_implicit_method(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_C_grid_implicit, h_C_grid_implicit, x4 = nm.C_grid_implicit_method(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)

    # Note x1, x2, x3 and x4 are all the same variables as nx, nt, xmin and xmax are the same for all methods
    
    # construct exact solution on both colocated and staggered grid
    u_A_grid = np.zeros_like(x1)
    u_C_grid = np.zeros_like(x1)
    h = np.zeros_like(x1)
    
    for i in range(len(x1)):
        u_A_grid[i] = math.sin(x1[i])*math.sin(dt*nt)
        h[i] = math.cos(x1[i])*math.cos(dt*nt)
        u_C_grid[i] = math.sin(x1[i]+ dx/2)*math.sin(dt*nt)
    
        
    # find error between exact solution and solution found by numerical method
    error_A_grid_u = (u_A_grid - u_A_grid_explicit)**2
    error_C_grid_u = (u_C_grid - u_C_grid_explicit)**2
    error_implicit_u = (u_A_grid - u_A_grid_implicit)**2
    error_semi_implicit_u = (u_C_grid - u_C_grid_implicit)**2
    
    # plot error in u from 4 different methods
    fig1, ax1 = plt.subplots()
    ax1.plot(x1, error_A_grid_u, c = 'blue', label = "A-grid explicit")
    ax1.plot(x2 + dx/2, error_C_grid_u, c = 'green', label = "C-grid explicit")
    ax1.plot(x3, error_implicit_u, c = 'red', label = "A-grid implicit")
    ax1.plot(x4 + dx/2, error_semi_implicit_u, c ='orange', label = "C-grid semi-implicit")
    
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")
    
    error_A_grid_h = (h - h_A_grid_explicit)**2
    error_C_grid_h = (h - h_C_grid_explicit)**2
    error_implicit_h = (h - h_A_grid_implicit)**2
    error_semi_implicit_h = (h - h_C_grid_implicit)**2
    
    # plot error in h from 4 different methods
    fig2, ax2 = plt.subplots()
    ax2.plot(x1, error_A_grid_h, c = 'blue', label = "A-grid explicit")
    ax2.plot(x2, error_C_grid_h, c = 'green', label = "C-grid explicit")
    ax2.plot(x3, error_implicit_h, c = 'red', label = "A-grid implicit")
    ax2.plot(x4, error_semi_implicit_h, c = 'orange', label = "C-grid semi-implicit")
    
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel("x")
    
    error_norms_u = [np.linalg.norm(u_A_grid - u_A_grid_explicit), np.linalg.norm(u_C_grid - u_C_grid_explicit), np.linalg.norm(u_A_grid - u_A_grid_implicit), np.linalg.norm(u_C_grid - u_C_grid_implicit)]

    error_norms_h = [np.linalg.norm(h - h_A_grid_explicit), np.linalg.norm(h - h_C_grid_explicit), np.linalg.norm(h - h_A_grid_implicit), np.linalg.norm(h - h_C_grid_implicit)]
    
    return fig1, fig2, ax1, ax2, error_norms_u, error_norms_h


def error_fn_cossin(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function compares the solutions of the 4 numerical methods studied for the initial condition 
       defined in the function initialconditions_cossin and finds the Frobenius norm of the error between 
       these solutions and the exact solution
        Note this function can only be used with this initial condition as otherwise the exact solution is incorrect.
    nx_range:           range of total number of space steps (in order to vary mesh size)
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """
    
    # initialize lists
    norm_A_grid_listu = np.zeros_like(nx_range).astype('float')
    norm_C_grid_listu = np.zeros_like(nx_range).astype('float')
    norm_A_grid_implicit_listu = np.zeros_like(nx_range).astype('float')
    norm_C_grid_implicit_listu = np.zeros_like(nx_range).astype('float')

    norm_A_grid_listh = np.zeros_like(nx_range).astype('float')
    norm_C_grid_listh = np.zeros_like(nx_range).astype('float')
    norm_A_grid_implicit_listh = np.zeros_like(nx_range).astype('float')
    norm_C_grid_implicit_listh = np.zeros_like(nx_range).astype('float')

    dx_list = np.zeros_like(nx_range).astype('float')
    dt_list = np.zeros_like(nt_range).astype('float')


    # find u and h for each numerical method for a range of space mesh sizes 
    for j in range(len(nx_range)):
        nx = nx_range[j]
        nt = nt_range[j]
        
        # end function if nt is not an integer
        if isinstance(nt, numbers.Integral) == False:
            raise ValueError('nt is not an integer') 
        
        # derive the width of the spacestep
        dx_list[j] = (xmax - xmin)/nx
        dt_list[j] = total_time/nt
        u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_A_grid_implicit, h_A_grid_implicit, x3 = nm.A_grid_implicit_method(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_implicit, h_C_grid_implicit, x4 = nm.C_grid_implicit_method(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)

        # Note x1, x2, x3 and x4 are all the same variables as nx, nt, xmin and xmax are the same for all methods
        
        # construct exact solution on both colocated and staggered grid
        u_A_grid = np.zeros_like(x1).astype('float')
        u_C_grid = np.zeros_like(x1).astype('float')
        h = np.zeros_like(x1).astype('float')
    
        for i in range(len(x1)):
            u_A_grid[i] = (math.cos(x1[i]) - math.sin(x1[i]))*(math.cos(total_time) - math.sin(total_time))
            h[i] = (math.cos(x1[i]) + math.sin(x1[i]))*(math.cos(total_time) + math.sin(total_time))
            u_C_grid[i] = (math.cos(x1[i] + (xmax - xmin)/(2*nx)) - math.sin(x1[i] + (xmax - xmin)/(2*nx)))*(math.cos(total_time) - math.sin(total_time))
        
        # find the Frobenius norm of the error between exact solution and solution found by numerical method
        norm_A_grid_listu[j] = np.linalg.norm((u_A_grid - u_A_grid_explicit))
        norm_C_grid_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_explicit))
        norm_A_grid_implicit_listu[j] = np.linalg.norm(u_A_grid - u_A_grid_implicit)
        norm_C_grid_implicit_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_implicit))
    
        norm_A_grid_listh[j] = np.linalg.norm((h - h_A_grid_explicit))
        norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
        norm_A_grid_implicit_listh[j] = np.linalg.norm((h - h_A_grid_implicit))
        norm_C_grid_implicit_listh[j] = np.linalg.norm((h - h_C_grid_implicit))
        
    
    # plot results on a log scale
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    
    ax1.loglog(dx_list, norm_A_grid_listu, label = 'A-grid explicit')
    ax1.loglog(dx_list, norm_C_grid_listu, label = 'C-grid explicit')
    ax1.loglog(dx_list, norm_A_grid_implicit_listu, label = 'A-grid implicit')
    ax1.loglog(dx_list, norm_C_grid_implicit_listu, label = 'C-grid implicit')
    ax1.legend(loc = 'best')
    ax1.set_xlim([min(dx_list),max(dx_list)])
    ax1.set_xlabel(r"$\Delta x$")
    ax1.set_ylabel("Error in u")
    fig1.savefig("uerror_compared_dx_cossin.png")
    ax1.set_title('u_dx')
    fig1.show()
    
    ax2.loglog(dx_list, norm_A_grid_listh, label = 'A-grid explicit')
    ax2.loglog(dx_list, norm_C_grid_listh, label = 'C-grid explicit')
    ax2.loglog(dx_list, norm_A_grid_implicit_listh, label = 'A-grid implicit')
    ax2.loglog(dx_list, norm_C_grid_implicit_listh, label = 'C-grid semi-implicit')
    ax2.legend(loc = 'best')
    ax2.set_xlim([min(dx_list),max(dx_list)])
    ax2.set_xlabel(r"$\Delta x$")
    ax2.set_ylabel("Error in h")
    fig2.savefig("herror_compared_dx_cossin.png")
    ax2.set_title('h_dx')
    fig2.show()
    
    ax3.loglog(dt_list, norm_A_grid_listu, label = 'A-grid explicit')
    ax3.loglog(dt_list, norm_C_grid_listu, label = 'C-grid explicit')
    ax3.loglog(dt_list, norm_A_grid_implicit_listu, label = 'A-grid implicit')
    ax3.loglog(dt_list, norm_C_grid_implicit_listu, label = 'C-grid semi-implicit')  
    ax3.legend(loc = 'best')
    ax3.set_xlim([min(dt_list),max(dt_list)])
    ax3.set_xlabel(r"$\Delta t$")
    ax3.set_ylabel("Error in u")
    fig3.savefig("uerror_compared_dt_cossin.png")
    ax3.set_title('u_dt')
    fig3.show()
    
    ax4.loglog(dt_list, norm_A_grid_listh, label = 'A-grid explicit')
    ax4.loglog(dt_list, norm_C_grid_listh, label = 'C-grid explicit')
    ax4.loglog(dt_list, norm_A_grid_implicit_listh, label = 'A-grid implicit')
    ax4.loglog(dt_list, norm_C_grid_implicit_listh, label = 'C-grid semi-implicit')
    ax4.legend(loc = 'best')
    ax4.set_xlim([min(dt_list),max(dt_list)])
    ax4.set_xlabel(r"$\Delta t$")
    ax4.set_ylabel("Error in h")
    fig4.savefig("herror_compared_dt_cossin.png")
    ax4.set_title('h_dt')
    fig4.show()
    
    return dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_A_grid_implicit_listu, norm_A_grid_implicit_listh, norm_C_grid_implicit_listu, norm_C_grid_implicit_listh

