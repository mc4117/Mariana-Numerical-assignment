#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:07:42 2017
@author: mc4117
"""

# -*- coding: utf-8 -*-
import numerical_methods as nm
import math
import numpy as np
import matplotlib.pyplot as plt
import initial_conditions as ic

total_time = math.pi/3

g = 1
H = 1
c = 0.1

xmin = -math.pi
xmax = math.pi
    

nx_range2 = [300, 600, 750, 900]
#nx_range2 = [160, 180, 200, 220, 240, 260]
nt_range2 = np.zeros_like(nx_range2).astype('int')

# as dx = (xmax - xmin)/nx = 2pi/nx and dt = c*dx/sqrt(gH) = 2pic/nxsqrt(gH)
# as nt = total_time/dt = pi/2 /(2pi c /nxsqrt(gH)) = 5 nx/2

for j in range(len(nx_range2)):
    nx = nx_range2[j]
    nt_range2[j] = 5 * nx/3
    


print(nt_range2)





def initialconditions_cossin(nx, xmin = 0, xmax = 1, plot = True):
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
        #ax1.set_title("Initital Condition where h is cos(x)")
        ax1.set_xlim = ([xmin, xmax])
        ax1.set_ylim([-1.1, 1.1])
        
        # add space between the title and the plot
        #plt.rcParams['axes.titlepad'] = 20 
        fig1.savefig("initial_condition_cos.png")
        fig1.show()
    return initialu, initialh, x




def error_fn_cossin(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function compares the solutions of the 4 numerical methods studied for the initial condition defined in the function initialconditions_cossin and finds the Frobenius norm of the error between these solutions and the exact solution
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
    dt_list = np.zeros_like(nt_range).astype('float')

    
    # find u and h for each numerical method for a range of space mesh sizes 
    for j in range(len(nx_range)):
        nx = nx_range[j]
        print(nx)
        nt = nt_range[j]
        # derive the width of the spacestep
        dx_list[j] = (xmax - xmin)/nx
        dt_list[j] = total_time/nt
        u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_implicit, h_implicit, x3 = nm.implicit_method(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_semi_implicit, h_semi_implicit, x4 = nm.semi_implicit_method(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)

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
        norm_implicit_listu[j] = np.linalg.norm(u_A_grid - u_implicit)
        norm_semi_implicit_listu[j] = np.linalg.norm((u_C_grid - u_semi_implicit))
    
        norm_A_grid_listh[j] = np.linalg.norm((h - h_A_grid_explicit))
        norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
        norm_implicit_listh[j] = np.linalg.norm((h - h_implicit))
        norm_semi_implicit_listh[j] = np.linalg.norm((h - h_semi_implicit))
        
    plt.loglog(dx_list, norm_A_grid_listu, label = 'A-grid')
    plt.loglog(dx_list, norm_C_grid_listu, label = 'C-grid')
    plt.loglog(dx_list, norm_implicit_listu, label = 'implicit')
    plt.loglog(dx_list, norm_semi_implicit_listu, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('u_dx')
    plt.show()
    
    plt.loglog(dx_list, norm_A_grid_listh, label = 'A-grid')
    plt.loglog(dx_list, norm_C_grid_listh, label = 'C-grid')
    plt.loglog(dx_list, norm_implicit_listh, label = 'implicit')
    plt.loglog(dx_list, norm_semi_implicit_listh, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('h_dx')
    plt.show()
    
    plt.loglog(dt_list, norm_A_grid_listu, label = 'A-grid')
    plt.loglog(dt_list, norm_C_grid_listu, label = 'C-grid')
    plt.loglog(dt_list, norm_implicit_listu, label = 'implicit')
    plt.loglog(dt_list, norm_semi_implicit_listu, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('u_dt')
    plt.show()
    
    plt.loglog(dt_list, norm_A_grid_listh, label = 'A-grid')
    plt.loglog(dt_list, norm_C_grid_listh, label = 'C-grid')
    plt.loglog(dt_list, norm_implicit_listh, label = 'implicit')
    plt.loglog(dt_list, norm_semi_implicit_listh, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('h_dt')
    plt.show()
    
    
    
    return dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_implicit_listu, norm_implicit_listh, norm_semi_implicit_listu, norm_semi_implicit_listh
    
def error_fn_cos(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):
    """This function compares the solutions of the 4 numerical methods studied for the initial condition defined in the function initialconditions_cossin and finds the Frobenius norm of the error between these solutions and the exact solution
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
    dt_list = np.zeros_like(nt_range).astype('float')
    

    
    # find u and h for each numerical method for a range of space mesh sizes 
    for j in range(len(nx_range)):
        nx = nx_range[j]
        print(nx)
        nt = nt_range[j]
        # derive the width of the spacestep
        dx_list[j] = (xmax - xmin)/nx
        dt_list[j] = total_time/nt
        
        u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
        u_implicit, h_implicit, x3 = nm.implicit_method(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)
        u_semi_implicit, h_semi_implicit, x4 = nm.semi_implicit_method(ic.initialconditions_cos, nx, nt, xmin, xmax, H, g, c)

        # Note x1, x2, x3 and x4 are all the same variables as nx, nt, xmin and xmax are the same for all methods
    

        
        # construct exact solution on both colocated and staggered grid
        u_A_grid = np.zeros_like(x1).astype('float')
        u_C_grid = np.zeros_like(x1).astype('float')
        h = np.zeros_like(x1).astype('float')
    
        for i in range(len(x1)):
            u_A_grid[i] = math.sin(x1[i])*math.sin(total_time)
            h[i] = math.cos(x1[i])*math.cos(total_time)
            u_C_grid[i] = math.sin(x1[i] + (xmax - xmin)/(2*nx))*math.sin(total_time)
        
        # find the Frobenius norm of the error between exact solution and solution found by numerical method
        norm_A_grid_listu[j] = np.linalg.norm((u_A_grid - u_A_grid_explicit))
        norm_C_grid_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_explicit))
        norm_implicit_listu[j] = np.linalg.norm((u_A_grid - u_implicit))
        norm_semi_implicit_listu[j] = np.linalg.norm((u_C_grid - u_semi_implicit))
    
        norm_A_grid_listh[j] = np.linalg.norm((h - h_A_grid_explicit))
        norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
        norm_implicit_listh[j] = np.linalg.norm((h - h_implicit))
        norm_semi_implicit_listh[j] = np.linalg.norm((h - h_semi_implicit))
        
    plt.loglog(dx_list, norm_A_grid_listu, label = 'A-grid')
    plt.loglog(dx_list, norm_C_grid_listu, label = 'C-grid')
    plt.loglog(dx_list, norm_implicit_listu, label = 'implicit')
    plt.loglog(dx_list, norm_semi_implicit_listu, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('u_dx')
    plt.show()
    
    plt.loglog(dx_list, norm_A_grid_listh, label = 'A-grid')
    plt.loglog(dx_list, norm_C_grid_listh, label = 'C-grid')
    plt.loglog(dx_list, norm_implicit_listh, label = 'implicit')
    plt.loglog(dx_list, norm_semi_implicit_listh, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('h_dx')
    plt.show()
    
    plt.loglog(dt_list, norm_A_grid_listu, label = 'A-grid')
    plt.loglog(dt_list, norm_C_grid_listu, label = 'C-grid')
    plt.loglog(dt_list, norm_implicit_listu, label = 'implicit')
    plt.loglog(dt_list, norm_semi_implicit_listu, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('u_dt')
    plt.show()
    
    plt.loglog(dt_list, norm_A_grid_listh, label = 'A-grid')
    plt.loglog(dt_list, norm_C_grid_listh, label = 'C-grid')
    plt.loglog(dt_list, norm_implicit_listh, label = 'implicit')
    plt.loglog(dt_list, norm_semi_implicit_listh, label = 'semi-implicit')
    plt.legend(loc = 'best')
    plt.title('h_dt')
    plt.show()
    """
    plt.plot(x1, u_A_grid_explicit, label = 'A explicit')
    plt.plot(x1 + (xmax-xmin)/(2*nx), u_C_grid_explicit, label = 'C explicit')
    plt.plot(x1, u_implicit, label = 'implicit')
    plt.plot(x1, u_semi_implicit, label = 'semi-implicit')
    plt.plot(x1, u_A_grid, label = 'exact')
    plt.legend()
    plt.show()
    
    plt.plot(x1, h_A_grid_explicit, label = 'a explicit')
    plt.plot(x1, h_C_grid_explicit, label = 'c explicit')
    plt.plot(x1, h_implicit, label = 'implicit')
    plt.plot(x1, h_semi_implicit, label = 'semi-implicit')
    plt.plot(x1, h, label = 'exact')
    plt.legend()
    plt.show()
    """
    return dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_implicit_listu, norm_implicit_listh, norm_semi_implicit_listu, norm_semi_implicit_listh

dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_implicit_listu, norm_implicit_listh, norm_semi_implicit_listu, norm_semi_implicit_listh = error_fn_cos(nx_range2, nt_range2, total_time, xmin, xmax, H, g, c)

cos_gradient_A_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listu),1)[0]
cos_gradient_C_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listu),1)[0]
cos_gradient_implicit_u_dx = np.polyfit(np.log(dx_list), np.log(norm_implicit_listu),1)[0]
cos_gradient_semi_implicit_u_dx = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listu),1)[0]
    
cos_gradient_A_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listh),1)[0]
cos_gradient_C_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listh),1)[0]
cos_gradient_implicit_h_dx = np.polyfit(np.log(dx_list), np.log(norm_implicit_listh),1)[0]
cos_gradient_semi_implicit_h_dx = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listh),1)[0]

cos_gradient_A_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listu),1)[0]
cos_gradient_C_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listu),1)[0]
cos_gradient_implicit_u_dt = np.polyfit(np.log(dt_list), np.log(norm_implicit_listu),1)[0]
cos_gradient_semi_implicit_u_dt = np.polyfit(np.log(dt_list), np.log(norm_semi_implicit_listu),1)[0]
    
cos_gradient_A_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listh),1)[0]
cos_gradient_C_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listh),1)[0]
cos_gradient_implicit_h_dt = np.polyfit(np.log(dt_list), np.log(norm_implicit_listh),1)[0]
cos_gradient_semi_implicit_h_dt = np.polyfit(np.log(dt_list), np.log(norm_semi_implicit_listh),1)[0]




dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_implicit_listu, norm_implicit_listh, norm_semi_implicit_listu, norm_semi_implicit_listh = gradient_A_grid_u, gradient_C_grid_u, gradient_implicit_u, gradient_semi_implicit_u, gradient_A_grid_h, gradient_C_grid_h, gradient_implicit_h, gradient_semi_implicit_h = error_fn_cossin(nx_range2, nt_range2, total_time, xmin, xmax, H, g, c)

cossin_gradient_A_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listu),1)[0]
cossin_gradient_C_grid_u_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listu),1)[0]
cossin_gradient_implicit_u_dx = np.polyfit(np.log(dx_list), np.log(norm_implicit_listu),1)[0]
cossin_gradient_semi_implicit_u_dx = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listu),1)[0]
    
cossin_gradient_A_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listh),1)[0]
cossin_gradient_C_grid_h_dx = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listh),1)[0]
cossin_gradient_implicit_h_dx = np.polyfit(np.log(dx_list), np.log(norm_implicit_listh),1)[0]
cossin_gradient_semi_implicit_h_dx = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listh),1)[0]

cossin_gradient_A_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listu),1)[0]
cossin_gradient_C_grid_u_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listu),1)[0]
cossin_gradient_implicit_u_dt = np.polyfit(np.log(dt_list), np.log(norm_implicit_listu),1)[0]
cossin_gradient_semi_implicit_u_dt = np.polyfit(np.log(dt_list), np.log(norm_semi_implicit_listu),1)[0]
    
cossin_gradient_A_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_A_grid_listh),1)[0]
cossin_gradient_C_grid_h_dt = np.polyfit(np.log(dt_list), np.log(norm_C_grid_listh),1)[0]
cossin_gradient_implicit_h_dt = np.polyfit(np.log(dt_list), np.log(norm_implicit_listh),1)[0]
cossin_gradient_semi_implicit_h_dt = np.polyfit(np.log(dt_list), np.log(norm_semi_implicit_listh),1)[0]

