#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Mariana Clare

Functions which calculate the difference between an analytic solution for the shallow
water equations and the solution found by the numerical methods studied.

"""

import numpy as np
import matplotlib.pyplot as plt
import numerical_methods as nm
import initial_conditions as ic
import numbers
import math




def error_calc(nx, nt, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function finds the solutions of the 4 numerical methods studied 
       for the initial condition that u = 0 everywhere and h is cos(x) and calculates 
       the error between these solutions and the exact solution
       Note this function can only be used with this initial condition (cos) as otherwise 
       the exact solution is incorrect.
       
       Inputs:
           
         nx:                 number of space steps
         nt:                 number of time steps
         xmin:               minimum value of x on grid
         xmax:               maximum value of x on grid
         H:                  mean fluid depth set to 1 unless otherwise specified
         g:                  acceleration due to gravity scaled to 1 unless otherwise specified
         c:                  Courant number (c = root(gH)dt/dx)
           
       Outputs:
           
         dx:                      spacestep
         dt:                      timestep
         error_A_grid_u:          array of squared difference between solution of u 
                                  by explicit A-grid method and exact solution of u
         error_C_grid_u:          array of squared difference between solution of u 
                                  by explicit C-grid method and exact solution of u  
         error_A_grid_implicit_u: array of squared difference between solution of u 
                                  by implicit A-grid method and exact solution of u
         error_C_grid_implicit_u: array of squared difference between solution of u 
                                  by implicit C-grid method and exact solution of u
         error_A_grid_h:          array of squared difference between solution of h 
                                  by explicit A-grid method and exact solution of h
         error_C_grid_h:          array of squared difference between solution of h 
                                  by explicit C-grid method and exact solution of h
         error_A_grid_implicit_h: array of squared difference between solution of h 
                                  by implicit A-grid method and exact solution of h
         error_C_grid_implicit_h: array of squared difference between solution of h 
                                  by implicit C-grid method and exact solution of h
           

    """
    # derive the width of the spacestep and timestep
    dx = (xmax - xmin)/nx
    dt = (c*dx)/math.sqrt(g*H)
    
    # find u and h for each numerical method
    u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_A_grid_implicit, h_A_grid_implicit, x3 = nm.A_grid_implicit_method(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
    u_C_grid_implicit, h_C_grid_implicit, x4 = nm.C_grid_implicit_method(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)

    # Note x1, x2, x3 and x4 are all the same variables as nx, nt, xmin and xmax 
    # are the same for all methods
    
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
    error_A_grid_implicit_u = (u_A_grid - u_A_grid_implicit)**2
    error_C_grid_implicit_u = (u_C_grid - u_C_grid_implicit)**2
    
    error_A_grid_h = (h - h_A_grid_explicit)**2
    error_C_grid_h = (h - h_C_grid_explicit)**2
    error_A_grid_implicit_h = (h - h_A_grid_implicit)**2
    error_C_grid_implicit_h = (h - h_C_grid_implicit)**2
    
    
    return dx, dt, error_A_grid_u, error_C_grid_u, error_A_grid_implicit_u, error_C_grid_implicit_u,\
             error_A_grid_h, error_C_grid_h, error_A_grid_implicit_h, error_C_grid_implicit_h
             



def error_fn(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function finds the solutions of the 4 numerical methods studied 
       for the initial condition defined in the function initialconditions_cos
       and for a range of nx and nt. 
       It then calculates the L2 norm of the error between these solutions and 
       the exact solution for each nx, nt combination.
       These errors are then plotted against dx and dt and the gradient of this line is calculated.
       
       Note this function can only be used with this initial condition (cos) as otherwise 
       the exact solution is incorrect.
       
    Inputs:
        nx_range:           range of total number of space steps (in order to vary mesh size)
        nt_range:           range of total number of time steps corresponding to values in nx_range
        xmin:               minimum value of x on grid
        xmax:               maximum value of x on grid
        H:                  mean fluid depth set to 1 unless otherwise specified
        g:                  acceleration due to gravity scaled to 1 unless otherwise specified
        c:                  Courant number (c = root(gH)dt/dx)
        
    Outputs:
        gradient_u_dx:      array containing order of accuracy of u with respect to 
                            dx for each numerical method
        gradient_u_dt:      array containing order of accuracy of u with respect to 
                            dt for each numerical method
        gradient_h_dx:      array containing order of accuracy of h with respect to 
                            dx for each numerical method
        gradient_h_dt:      array containing order of accuracy of h with respect to 
                            dt for each numerical method
        
    """
    
    # initialize system
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
        
        # calculate the squared error between the analytic and the numeric solutions
        dx_list[j], dt_list[j], error_A_grid_u, error_C_grid_u, error_A_grid_implicit_u,\
             error_C_grid_implicit_u, error_A_grid_h, error_C_grid_h, error_A_grid_implicit_h,\
             error_C_grid_implicit_h = error_calc(nx, nt, xmin, xmax, H, g, c)

        # calculate the L2 norm of the error between the analytic solution and the 
        # numeric solution. Note this is the L2 norm as the values produced by the error_calc
        # function is the squared error.
        
        norm_A_grid_listu[j] = math.sqrt(sum(error_A_grid_u))
        norm_C_grid_listu[j] = math.sqrt(sum(error_C_grid_u))
        norm_A_grid_implicit_listu[j] = math.sqrt(sum(error_A_grid_implicit_u))
        norm_C_grid_implicit_listu[j] = math.sqrt(sum(error_C_grid_implicit_u))
    
        norm_A_grid_listh[j] = math.sqrt(sum(error_A_grid_h))
        norm_C_grid_listh[j] = math.sqrt(sum(error_C_grid_h))
        norm_A_grid_implicit_listh[j] = math.sqrt(sum(error_A_grid_implicit_h))
        norm_C_grid_implicit_listh[j] = math.sqrt(sum(error_C_grid_implicit_h))
    
    # log plot of dx vs error norm of u for each numerical method    
    plt.loglog(dx_list, norm_A_grid_listu, label = 'A-grid explicit')
    plt.loglog(dx_list, norm_C_grid_listu, label = 'C-grid explicit')
    plt.loglog(dx_list, norm_A_grid_implicit_listu, label = 'A-grid implicit')
    plt.loglog(dx_list, norm_C_grid_implicit_listu, label = 'C-grid implicit')
    plt.legend(loc = 'best')
    plt.xlim([min(dx_list), max(dx_list)])
    plt.xlabel(r"$\Delta x$")
    plt.ylabel("Error in u")
    plt.savefig("uerror_compared_dx_cossin.png")
    plt.title('u with respect to dx')
    plt.show()
    
    # log plot of dx vs error norm of h for each numerical method
    plt.loglog(dx_list, norm_A_grid_listh, c = 'firebrick', linewidth = 5, label = 'A-grid explicit')
    plt.loglog(dx_list, norm_C_grid_listh, c= 'orange', linestyle='--', linewidth = 4, label = 'C-grid explicit')
    plt.loglog(dx_list, norm_A_grid_implicit_listh, c= 'gold', linestyle = ':', linewidth = 7, label = 'A-grid implicit')
    plt.loglog(dx_list, norm_C_grid_implicit_listh, 'k', label = 'C-grid implicit')
    plt.legend(loc = 'best')
    plt.xlim([min(dx_list), max(dx_list)])
    plt.xlabel(r"$\Delta x$")
    plt.ylabel("Error in h")
    plt.savefig("herror_compared_dx_cossin.png")
    plt.title('h with respect to dx')
    plt.show()

    # log plot of dt vs error norm of u for each numerical method
    plt.loglog(dt_list, norm_A_grid_listu, label = 'A-grid explicit')
    plt.loglog(dt_list, norm_C_grid_listu, label = 'C-grid explicit')
    plt.loglog(dt_list, norm_A_grid_implicit_listu, label = 'A-grid implicit')
    plt.loglog(dt_list, norm_C_grid_implicit_listu, label = 'C-grid implicit') 
    plt.legend(loc = 'best')
    plt.xlim([min(dt_list), max(dt_list)])
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error in u")
    plt.savefig("uerror_compared_dt_cossin.png")
    plt.title('u with respect to dt')
    plt.show()
    
    # log plot of dx vs error norm of u for each numerical method
    plt.loglog(dt_list, norm_A_grid_listh, c = 'firebrick', linewidth = 5, label = 'A-grid explicit')
    plt.loglog(dt_list, norm_C_grid_listh, c= 'orange', linestyle='--', linewidth = 4, label = 'C-grid explicit')
    plt.loglog(dt_list, norm_A_grid_implicit_listh, c= 'gold', linestyle = ':', linewidth = 7, label = 'A-grid implicit')
    plt.loglog(dt_list, norm_C_grid_implicit_listh, 'k', label = 'C-grid implicit')
    plt.legend(loc = 'best')
    plt.xlim([min(dt_list), max(dt_list)])
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error in h")
    plt.savefig("herror_compared_dt_cossin.png")
    plt.title('h with respect to dt')
    plt.show()
    
    # attempt to fit a straight line on the relationship between log(dx) and the 
    # log of the error of u for each numerical method. The gradient of this line 
    # is the order of the scheme with respect to dx for u
    gradient_A_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listu),1)[0]
    gradient_C_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listu),1)[0]
    gradient_A_grid_implicit_u_dx = np.polyfit(np.log(dx_list),\
                                                      np.log(norm_A_grid_implicit_listu),1)[0]
    gradient_C_grid_implicit_u_dx = np.polyfit(np.log(dx_list),\
                                                      np.log(norm_C_grid_implicit_listu),1)[0]

    # attempt to fit a straight line on the relationship between log(dx) and the log 
    # of the error of h for each numerical method. The gradient of this line is the 
    # order of the scheme with respect to dx for h
    gradient_A_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listh),1)[0]
    gradient_C_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listh),1)[0]
    gradient_A_grid_implicit_h_dx = np.polyfit(np.log(dx_list), \
                                                      np.log(norm_A_grid_implicit_listh),1)[0]
    gradient_C_grid_implicit_h_dx = np.polyfit(np.log(dx_list), \
                                                      np.log(norm_C_grid_implicit_listh),1)[0]

    # attempt to fit a straight line on the relationship between log(dt) and the log 
    # of the error of u for each numerical method. The gradient of this line is the 
    # order of the scheme with respect to dt for u
    gradient_A_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listu),1)[0]
    gradient_C_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listu),1)[0]
    gradient_A_grid_implicit_u_dt = np.polyfit(np.log(dt_list), \
                                                      np.log(norm_A_grid_implicit_listu),1)[0]
    gradient_C_grid_implicit_u_dt = np.polyfit(np.log(dt_list), \
                                                      np.log(norm_C_grid_implicit_listu),1)[0]
    
    # attempt to fit a straight line on the relationship between log(dt) and the log 
    # of the error of h for each numerical method. The gradient of this line is the 
    # order of the scheme with respect to dt for h
    gradient_A_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listh),1)[0]
    gradient_C_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listh),1)[0]
    gradient_A_grid_implicit_h_dt = np.polyfit(np.log(dt_list),\
                                                      np.log(norm_A_grid_implicit_listh),1)[0]
    gradient_C_grid_implicit_h_dt = np.polyfit(np.log(dt_list),\
                                                      np.log(norm_C_grid_implicit_listh),1)[0]
    
    # compile the gradients to 4 arrays for ease of processing
    gradient_u_dx = [gradient_A_grid_u_dx, gradient_C_grid_u_dx, gradient_A_grid_implicit_u_dx, \
                     gradient_C_grid_implicit_u_dx]
    gradient_u_dt = [gradient_A_grid_u_dt, gradient_C_grid_u_dt, gradient_A_grid_implicit_u_dt, \
                     gradient_C_grid_implicit_u_dt]
    gradient_h_dx = [gradient_A_grid_h_dx, gradient_C_grid_h_dx, gradient_A_grid_implicit_h_dx, \
                     gradient_C_grid_implicit_h_dx]    
    gradient_h_dt = [gradient_A_grid_h_dt, gradient_C_grid_h_dt, gradient_A_grid_implicit_h_dt, \
                     gradient_C_grid_implicit_h_dt]      
    
    return gradient_u_dx, gradient_u_dt, gradient_h_dx, gradient_h_dt
