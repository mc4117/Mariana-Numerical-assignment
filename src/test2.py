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

total_time = math.pi/3

g = 1
H = 1
c = 0.1

xmin = -math.pi
xmax = math.pi
    

nx_range2 = [30, 60, 75, 90]
#nx_range2 = [160, 180, 200, 220, 240, 260]
#nt_range2 = np.zeros_like(nx_range2).astype('int')
nt_range2 = np.zeros_like(nx_range2)
# as dx = (xmax - xmin)/nx = 2pi/nx and dt = c*dx/sqrt(gH) = 2pic/nxsqrt(gH)
# as nt = total_time/dt = pi/2 /(2pi c /nxsqrt(gH)) = 5 nx/2

for j in range(len(nx_range2)):
    nx = nx_range2[j]
    nt_range2[j] = 5 * nx/3
    

print(nt_range2)

def semi_implicit_method(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using the staggered scheme with a semi-implicit method
        Both equations are discretised using the theta method using theta = 1/2 (ie. crank nicholson) and centred in space.

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """    
    x = np.linspace(xmin, xmax, nx+1)
    xhalf = x + (xmax-xmin)/(2*nx)
    
    # set initial conditions
    initialuAgrid, initialhAgrid = initialconditions(x)
    uhalf, initialhCgrid = initialconditions(xhalf)
    #uhalf = np.zeros(len(initialuCgrid))
    
    # for a c-grid the velocity u is stagerred in the x-direction by half
    #for i in range(nx + 1):
    #    uhalf[i] = (initialu[i] + initialu[(i+1)%nx])/2
    # therefore uhalf[i] = u_{i + 1/2}
    
    # initialize the system
    u_semi_implicit = np.zeros_like(uhalf)
    h_semi_implicit = np.zeros_like(initialhAgrid)

    uOld = uhalf.copy()
    hOld = initialhAgrid.copy()
    
    # for plotting reasons we have chosen to include the end point in the scheme
    # hence the matrix has the following dimensions: 
    matrix = np.zeros((nx+ 1,nx + 1))
    
    # because of the way the matrix has been constructed with a modulus operator we still have periodic boundaries:

    for i in range(nx+1):    
        matrix[i,i] = 1 + c**2/2
        matrix[i, (i-1)%nx] = -(c**2)/4
        matrix[i, (i+1)%nx] = -(c**2)/4
    
    # loop over timesteps
    for it in range(nt):
        
        semi_implicit_uvector = np.zeros_like(uOld)
        for i in range(nx + 1):
            semi_implicit_uvector[i] = -math.sqrt(g/H)*c*(hOld[(i + 1)%nx] - hOld[i%nx]) + ((c**2)/4)*uOld[(i+1)%nx] + (1-(c**2)/2)*uOld[i%nx] + ((c**2)/4)*uOld[(i-1)%nx]
        # note here that semi_implicit_uvector[nx] = semi_implicit_uvector[0] because of modulus operator so periodic boundaries are still kept
        
        # solve matrix equation to find u
        u_semi_implicit = np.linalg.solve(matrix, semi_implicit_uvector)

        semi_implicit_hvector = np.zeros_like(hOld)
        for i in range(nx + 1):
            semi_implicit_hvector[i] = -math.sqrt(H/g)*c*(uOld[i%nx] - uOld[(i-1)%nx]) + ((c**2)/4)*hOld[(i+1)%nx] + (1-(c**2)/2)*hOld[(i)%nx] + ((c**2)/4)*hOld[(i-1)%nx]
        # note here that semi_implicit_hvector[nx] = semi_implicit_hvector[0] because of modulus operator so periodic boundaries are still kept
        
        # solve matrix equation to find h
        h_semi_implicit = np.linalg.solve(matrix, semi_implicit_hvector)

        # copy u and h for next iteration
        uOld = u_semi_implicit.copy()
        hOld = h_semi_implicit.copy()

    return u_semi_implicit, h_semi_implicit, x


"""

def error_fn_cossin(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 1):


    #norm_A_grid_listu = np.zeros_like(nx_range).astype('float')
    #norm_C_grid_listu = np.zeros_like(nx_range).astype('float')
    #norm_implicit_listu = np.zeros_like(nx_range).astype('float')
    norm_semi_implicit_listu = np.zeros_like(nx_range).astype('float')

    #norm_A_grid_listh = np.zeros_like(nx_range).astype('float')
    #norm_C_grid_listh = np.zeros_like(nx_range).astype('float')
    #norm_implicit_listh = np.zeros_like(nx_range).astype('float')
    norm_semi_implicit_listh = np.zeros_like(nx_range).astype('float')

    dx_list = np.zeros_like(nx_range).astype('float')
    

    
    # find u and h for each numerical method for a range of space mesh sizes 
    for j in range(len(nx_range)):
        nx = nx_range[j]
        nt = nt_range[j]
        # derive the width of the spacestep
        dx_list[j] = (xmax - xmin)/nx
        
        #u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        #u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        #u_implicit, h_implicit, x3 = nm.implicit_method(initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_semi_implicit, h_semi_implicit, x1 = semi_implicit_method(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)

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
        #norm_A_grid_listu[j] = np.linalg.norm((u_A_grid - u_A_grid_explicit))
        #norm_C_grid_listu[j] = np.linalg.norm((u_C_grid - u_C_grid_explicit))
        #norm_implicit_listu[j] = np.linalg.norm((u_A_grid - u_implicit))
        norm_semi_implicit_listu[j] = np.linalg.norm((u_C_grid - u_semi_implicit))
    
        #norm_A_grid_listh[j] = np.linalg.norm((h - h_A_grid_explicit))
        #norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
        #norm_implicit_listh[j] = np.linalg.norm((h - h_implicit))
        norm_semi_implicit_listh[j] = np.linalg.norm((h - h_semi_implicit))
        
    #plt.plot((dx_list), (norm_A_grid_listu), label = 'u')
    #plt.plot((dx_list), (norm_A_grid_listh), label = 'h')
    #plt.legend(loc = 'best')
    #plt.title('Agrid')
    #plt.show()
    
    #plt.plot((dx_list), (norm_C_grid_listu), label = 'u')
    #plt.plot((dx_list), (norm_C_grid_listh), label = 'h')
    #plt.legend(loc = 'best')
    #plt.title('Cgrid')
    #plt.show()
    
    #plt.plot((dx_list), (norm_implicit_listu), label = 'u')
    #plt.plot((dx_list), (norm_implicit_listh), label = 'h')
    #plt.legend(loc = 'best')
    #plt.title('implicit')
    #plt.show()
    
    plt.loglog((dx_list), (norm_semi_implicit_listu), label = 'u')
    plt.loglog((dx_list), (norm_semi_implicit_listh), label = 'h')
    plt.legend(loc = 'best')
    plt.title('semi-implicit')
    plt.show()

    #gradient_A_grid_u = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listu),1)[0]
    #gradient_C_grid_u = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listu),1)[0]
    #gradient_implicit_u = np.polyfit(np.log(dx_list), np.log(norm_implicit_listu),1)[0]
    #gradient_semi_implicit_u = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listu),1)[0]
    
    #gradient_A_grid_h = np.polyfit(np.log(dx_list), np.log(norm_A_grid_listh),1)[0]
    #gradient_C_grid_h = np.polyfit(np.log(dx_list), np.log(norm_C_grid_listh),1)[0]
    #gradient_implicit_h = np.polyfit(np.log(dx_list), np.log(norm_implicit_listh),1)[0]
    #gradient_semi_implicit_h = np.polyfit(np.log(dx_list), np.log(norm_semi_implicit_listh),1)[0]
    plt.show()
    return None
"""
def error_fn_cossin(nx_range, nt_range, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1):


    
    norm_A_grid_listu = np.zeros_like(nx_range).astype('float')
    norm_C_grid_listu = np.zeros_like(nx_range).astype('float')
    norm_A_grid_implicit_listu = np.zeros_like(nx_range).astype('float')
    norm_C_grid_implicit_listu = np.zeros_like(nx_range).astype('float')

    norm_A_grid_listh = np.zeros_like(nx_range).astype('float')
    norm_C_grid_listh = np.zeros_like(nx_range).astype('float')
    norm_A_grid_implicit_listh = np.zeros_like(nx_range).astype('float')
    norm_C_grid_implicit_listh = np.zeros_like(nx_range).astype('float')

    dx_list = np.zeros_like(nx_range).astype('float')
    dt_list = np.zeros_like(nx_range).astype('float')
    

    
    # find u and h for each numerical method for a range of space mesh sizes 
    for j in range(len(nx_range)):
        nx = nx_range[j]
        nt = nt_range[j]
        # derive the width of the spacestep
        dx_list[j] = (xmax - xmin)/nx
        dt_list[j] = total_time/nt
        
        u_A_grid_explicit, h_A_grid_explicit, x1 = nm.A_grid_explicit(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_explicit, h_C_grid_explicit, x2 = nm.C_grid_explicit(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_A_grid_implicit, h_A_grid_implicit, x3 = nm.A_grid_implicit_method(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)
        u_C_grid_implicit, h_C_grid_implicit, x4 = semi_implicit_method(ic.initialconditions_cossin, nx, nt, xmin, xmax, H, g, c)

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
        norm_A_grid_implicit_listh[j] = np.linalg.norm((h - h_A_grid_implicit))
        norm_C_grid_listh[j] = np.linalg.norm((h - h_C_grid_explicit))
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
    
    ax4.loglog(dt_list, norm_C_grid_listh, label = 'C-grid explicit')
    ax4.loglog(dt_list, norm_A_grid_listh, label = 'A-grid explicit')
    ax4.loglog(dt_list, norm_C_grid_implicit_listh, label = 'C-grid semi-implicit')
    ax4.loglog(dt_list, norm_A_grid_implicit_listh, label = 'A-grid implicit')
    ax4.legend(loc = 'best')
    ax4.set_xlim([min(dt_list),max(dt_list)])
    ax4.set_xlabel(r"$\Delta t$")
    ax4.set_ylabel("Error in h")
    fig4.savefig("herror_compared_dt_cossin.png")
    ax4.set_title('h_dt')
    fig4.show()
    
    return dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_A_grid_implicit_listu, norm_A_grid_implicit_listh, norm_C_grid_implicit_listu, norm_C_grid_implicit_listh


#gradient_A_grid_u, gradient_C_grid_u, gradient_implicit_u, gradient_semi_implicit_u, gradient_A_grid_h, gradient_C_grid_h, gradient_implicit_h, gradient_semi_implicit_h = error_fn_cossin(nx_range2, nt_range2, total_time, xmin, xmax, H, g, c)
#print(gradient_A_grid_u, gradient_C_grid_u, gradient_implicit_u, gradient_semi_implicit_u, gradient_A_grid_h, gradient_C_grid_h, gradient_implicit_h, gradient_semi_implicit_h)
dx_list, dt_list, norm_A_grid_listu, norm_A_grid_listh, norm_C_grid_listu, norm_C_grid_listh, norm_A_grid_implicit_listu, norm_A_grid_implicit_listh, norm_C_grid_implicit_listu, norm_C_grid_implicit_listh = error_fn_cossin(nx_range2, nt_range2, total_time, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1)

    

