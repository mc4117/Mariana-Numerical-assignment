#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import error_functions as errfns
import math
import numpy as np
import matplotlib.pyplot as plt
import numerical_methods as nm
import initial_conditions as ic
import numbers

H = 1
g = 1
xmin = -math.pi
xmax = math.pi


def error_fn_cos(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function compares the solutions of the 4 numerical methods studied 
       for the initial condition defined in the function initialconditions_cossin 
       and finds the Frobenius norm of the error between these solutions and the exact solution.
       Note this function can only be used with this initial condition (cossin) as otherwise 
       the exact solution is incorrect.
    nx_range:           range of total number of space steps (in order to vary mesh size)
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1 unless otherwise specified
    c:                  courant number (c = root(gH)dt/dx)
    """
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
        u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
        u_A_grid_implicit, h_A_grid_implicit, x3 = nm.A_grid_implicit_method(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_implicit, h_C_grid_implicit, x4 = nm.C_grid_implicit_method(\
                ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)

        # Note x1, x2, x3 and x4 are all the same variables as nx, nt, xmin and 
        # xmax are the same for all methods
    

        
        # construct exact solution on both colocated and staggered grid
        u_A_grid = np.zeros_like(x1).astype('float')
        u_C_grid = np.zeros_like(x1).astype('float')
        h = np.zeros_like(x1).astype('float')
    

        for i in range(len(x1)):
            u_A_grid[i] = math.sin(x1[i])*math.sin(total_time)
            h[i] = math.cos(x1[i])*math.cos(total_time)
            u_C_grid[i] = math.sin(x1[i]+ dx_list[j]/2)*math.sin(total_time)
        
        # find the Frobenius norm of the error between exact solution and solution 
        # found by numerical method
        norm_A_grid_listu[j] = np.linalg.norm((u_A_grid - u_A_grid_explicit))
        norm_C_grid_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_explicit))
        norm_A_grid_implicit_listu[j] = np.linalg.norm(u_A_grid - u_A_grid_implicit)
        norm_C_grid_implicit_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_implicit))
    
        norm_A_grid_listh[j] = np.linalg.norm((h - h_A_grid_explicit))
        norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
        norm_A_grid_implicit_listh[j] = np.linalg.norm((h - h_A_grid_implicit))
        norm_C_grid_implicit_listh[j] = np.linalg.norm((h - h_C_grid_implicit))
        
        
    plt.loglog(dx_list, norm_A_grid_listu, label = 'A-grid explicit')
    plt.loglog(dx_list, norm_C_grid_listu, label = 'C-grid explicit')
    plt.loglog(dx_list, norm_A_grid_implicit_listu, label = 'A-grid implicit')
    plt.loglog(dx_list, norm_C_grid_implicit_listu, label = 'C-grid implicit')
    plt.legend(loc = 'best')
    plt.xlim([min(dx_list), max(dx_list)])
    plt.xlabel(r"$\Delta x$")
    plt.ylabel("Error in u")
    plt.savefig("uerror_compared_dx_cossin.png")
    plt.title('u_dx')
    plt.show()
    
    plt.loglog(dx_list, norm_A_grid_listh, label = 'A-grid explicit')
    plt.loglog(dx_list, norm_C_grid_listh, label = 'C-grid explicit')
    plt.loglog(dx_list, norm_A_grid_implicit_listh, label = 'A-grid implicit')
    plt.loglog(dx_list, norm_C_grid_implicit_listh, label = 'C-grid implicit')
    plt.legend(loc = 'best')
    plt.xlim([min(dx_list), max(dx_list)])
    plt.xlabel(r"$\Delta x$")
    plt.ylabel("Error in h")
    plt.savefig("herror_compared_dx_cossin.png")
    plt.title('h_dx')
    plt.show()
    """
    plt.loglog(dt_list, norm_A_grid_listu, label = 'A-grid explicit')
    plt.loglog(dt_list, norm_C_grid_listu, label = 'C-grid explicit')
    plt.loglog(dt_list, norm_A_grid_implicit_listu, label = 'A-grid implicit')
    plt.loglog(dt_list, norm_C_grid_implicit_listu, label = 'C-grid implicit')  
    plt.legend(loc = 'best')
    plt.xlim([min(dt_list), max(dt_list)])
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error in u")
    plt.savefig("uerror_compared_dt_cossin.png")
    plt.title('u_dt')
    plt.show()
    
    plt.loglog(dt_list, norm_A_grid_listh, label = 'A-grid explicit')
    plt.loglog(dt_list, norm_C_grid_listh, label = 'C-grid explicit')
    plt.loglog(dt_list, norm_A_grid_implicit_listh, label = 'A-grid implicit')
    plt.loglog(dt_list, norm_C_grid_implicit_listh, label = 'C-grid implicit')
    plt.legend(loc = 'best')
    plt.xlim([min(dt_list), max(dt_list)])
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error in h")
    plt.savefig("herror_compared_dt_cossin.png")
    plt.title('h_dt')
    plt.show()
    """
    
    
    return dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, \
           norm_C_grid_listh, norm_A_grid_implicit_listu, norm_A_grid_implicit_listh, \
           norm_C_grid_implicit_listu, norm_C_grid_implicit_listh


def main(total_timefactor, c, nx_range, H, g):

    total_time = total_timefactor*math.pi

    nt_range = np.zeros_like(nx_range).astype('int')


    for j in range(len(nx_range)):
        nx_r = nx_range[j]
        nt_range[j] = total_timefactor*nx_r/(2*c)

    print(nt_range)
    
    dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, \
        norm_C_grid_listh, norm_A_grid_implicit_listu, norm_A_grid_implicit_listh, \
        norm_C_grid_implicit_listu, norm_C_grid_implicit_listh = \
        error_fn_cos(nx_range, nt_range, total_time, xmin, \
                               xmax, H, g, c)
        
    cossin_gradient_A_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listu),1)[0]
    cossin_gradient_C_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listu),1)[0]
    cossin_gradient_A_grid_implicit_u_dx = np.polyfit(np.log(dx_list),\
                                                          np.log(norm_A_grid_implicit_listu),1)[0]
    cossin_gradient_C_grid_implicit_u_dx = np.polyfit(np.log(dx_list),\
                                                      np.log(norm_C_grid_implicit_listu),1)[0]

        # attempt to fit a straight line on the relationship between log(dx) and the log 
        # of the error of h for each numerical method. The gradient of this line is the 
        # order of the scheme with respect to dx for h
    cossin_gradient_A_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listh),1)[0]
    cossin_gradient_C_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listh),1)[0]
    cossin_gradient_A_grid_implicit_h_dx = np.polyfit(np.log(dx_list), \
                                                          np.log(norm_A_grid_implicit_listh),1)[0]
    cossin_gradient_C_grid_implicit_h_dx = np.polyfit(np.log(dx_list), \
                                                     np.log(norm_C_grid_implicit_listh),1)[0]
        
        # attempt to fit a straight line on the relationship between log(dt) and the log 
        # of the error of u for each numerical method. The gradient of this line is the 
        # order of the scheme with respect to dt for u
    cossin_gradient_A_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listu),1)[0]
    cossin_gradient_C_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listu),1)[0]
    cossin_gradient_A_grid_implicit_u_dt = np.polyfit(np.log(dt_list), \
                                                      np.log(norm_A_grid_implicit_listu),1)[0]
    cossin_gradient_C_grid_implicit_u_dt = np.polyfit(np.log(dt_list), \
                                                          np.log(norm_C_grid_implicit_listu),1)[0]
        
    # attempt to fit a straight line on the relationship between log(dt) and the log 
    # of the error of h for each numerical method. The gradient of this line is the 
    # order of the scheme with respect to dt for h
    cossin_gradient_A_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listh),1)[0]
    cossin_gradient_C_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listh),1)[0]
    cossin_gradient_A_grid_implicit_h_dt = np.polyfit(np.log(dt_list),\
                                                      np.log(norm_A_grid_implicit_listh),1)[0]
    cossin_gradient_C_grid_implicit_h_dt = np.polyfit(np.log(dt_list),\
                                                  np.log(norm_C_grid_implicit_listh),1)[0]
    plt.show()

    print ("Numerical method| u error vs dx| u error vs dt| h error vs dx| h error vs dt")
    print("A_grid explicit| %f | %f | %f | %f" % (cossin_gradient_A_grid_u_dx, \
                                                  cossin_gradient_A_grid_u_dt, cossin_gradient_A_grid_h_dx, cossin_gradient_A_grid_h_dt))
    print("C_grid explicit| %f | %f | %f | %f" % (cossin_gradient_C_grid_u_dx, \
        cossin_gradient_C_grid_u_dt, cossin_gradient_C_grid_h_dx, cossin_gradient_C_grid_h_dt))
    print("A_grid implicit|%f | %f | %f | %f" % (cossin_gradient_A_grid_implicit_u_dx, \
                                                 cossin_gradient_A_grid_implicit_u_dt, cossin_gradient_A_grid_implicit_h_dx, \
                                                 cossin_gradient_A_grid_implicit_h_dt))
    print("C_grid implicit|%f | %f | %f | %f" % (cossin_gradient_C_grid_implicit_u_dx, \
        cossin_gradient_C_grid_implicit_u_dt, cossin_gradient_C_grid_implicit_h_dx, \
        cossin_gradient_C_grid_implicit_h_dt))
    return None



total_time_range = [1/24, 1/12, 1/8, 1/6]
#total_time_range = [1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1, 9/8]
nx_range = [120, 240, 360, 480]
#nx_range = [120, 160, 200, 240]
c = 0.1

for i in total_time_range:
    print(i)
    main(i, 0.1, nx_range, H, g)

