#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare

This file contains the main code for solving the shallow water equations using the 
numerical schemes. It also tests various properties of the schemes.
It calls on functions defined in the following python files:
initial_conditions.py, numerical_methods.py, plotting_functions.py and error_functions.py
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import initial_conditions as ic
import numerical_methods as nm
import plotting_functions as pltfns
import error_functions as errfns


def main():

    print('Finding solution to Shallow Water Equations...')
    
    # define the meshgrid parameters
    
    xmin = 0 # minimum value of x on grid
    xmax = 1 # maximum value of x on grid
    
    nx = 100  # number of points from x = xmin to x = xmax
    nt = 100 # number of time steps
    
    # First attempt a simple initial condition with a co-located forward-backward scheme
    
    # set up a meshgrid on x
    initialx = np.linspace(xmin,xmax,nx+1) 
    # note want extra point at the boundary for plotting reasons but in reality 
    # h[0] and h[nx] are equal and u[0] and u[nx] are equal 
    
    # plot initial conditions where u is zero everywhere and h has a bump in the centre 
    # and is surrounded by zero either side (hereafter referred to as cosbell)
    initialu, initialh = ic.initialconditions_cosbell(initialx)

    figic, axic = plt.subplots()
    axic.plot(initialx, initialu, 'r--', label = 'Initial u')        
    axic.plot(initialx, initialh, 'g-', label = 'Initial h')

    axic.legend(loc = 'best')
    axic.set_xlabel("x")
    axic.set_xlim([min(initialx),max(initialx)])
    axic.set_ylim([-0.1, 1.1])
        
    figic.savefig("initial_condition_cosbell.png")
    
    axic.set_title("Initial Condition where h has a bump in the centre (cosbell curve)")
    
    plt.show()

    # to plot solution first set the colours to be used in the plot
    mymap = plt.get_cmap("YlOrRd")
    colorrange_multiiterations = mymap(np.r_[np.array([0.3, 0.55, 1]), np.array([0.3, 0.55, 1])])

    # then set the linestyles that are used for each line in the plot
    linestylerange = np.array(['-', '--', '-'])

    plotparameterrange = [colorrange_multiiterations, linestylerange]
    
    # now plot solution at various time iterations for an explicit method on a co-located 
    # grid for the initial condition where u is zero everywhere and h is a cosbell curve
    
    # initialize plots 
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    
    fig1, fig2, ax1, ax2 = pltfns.plot_multiple_iterations(fig1, ax1, fig2, ax2, \
            ic.initialconditions_cosbell, nx, 3*nt, 3, nm.A_grid_explicit, \
            plotparameterrange, xmin, xmax, H = 1, g = 1, c = 0.1)
    
    ax1.plot(initialx, initialu, 'g', label = 'initial u')
    ax2.plot(initialx, initialh, 'g', label = 'initial h')
    
   # display legend
    ax1.legend(loc = 'best', fontsize = 10)
    ax2.legend(loc = 'best', fontsize = 10)
    
    fig1.savefig("velocity_colocated_explicit_cosbell.png")
    fig2.savefig("height_colocated_explicit_cosbell.png")
    
    ax1.set_title("Velocity, u, calculated using the co-located explicit scheme \n\
and initial condition of a cosbell curve") 
    ax2.set_title("Height, h, calculated using the co-located explicit scheme \n\
and initial condition of a cosbell curve")
    
    plt.show()
    
    
    # This seems to work well but through von-neumann stability analysis we find 
    # that this is unstable for c>2
    
    print('Varying Courant numbers...')
    
    # initialize plots
    
    fig_u_Agrid_c, ax_u_Agrid_c = plt.subplots()
    fig_h_Agrid_c, ax_h_Agrid_c = plt.subplots()

    
    # set color and linestyle range for each line in the plot as before
    
    mymap1 = plt.get_cmap("YlOrRd")
    colorrange_multic1 = mymap1(np.r_[np.array([0.4, 0.7,1]), np.array([0.4, 0.7, 1])])
    linestylerange1 = np.array(['-', '-', ':'])
    
    plotparameterrange1 = [colorrange_multic1, linestylerange1]
    
    # define range of Courant numbers to test where explicit co-located method becomes unstable
    crangecolocated1 = [1.75, 1.95, 2.05]
    
    # plot explicit method on co-located grid for different Courant numbers
    fig_u_Agrid_c, ax_u_Agrid_c, fig_h_Agrid_c, ax_h_Agrid_c = pltfns.plot_multiple_c(\
            fig_u_Agrid_c, ax_u_Agrid_c, fig_h_Agrid_c, ax_h_Agrid_c, ic.initialconditions_cossin,\
            nm.A_grid_explicit, crangecolocated1, plotparameterrange1,\
            nx = 100, nt = 100, H = 1, g = 1)
    
    # as this is unstable try implicit method

    # set another color and linestyle range for each line in plot. These are different
    # so as to differentiate between the explicit and implicit methods
    
    mymap2 = plt.get_cmap("YlGnBu")
    colorrange_multic2 = mymap2(np.r_[np.array([0.4, 0.7,1]), np.array([0.4, 0.7, 1])])
    linestylerange2 = np.array(['-', '-', '-'])

    plotparameterrange2 = [colorrange_multic2, linestylerange2]
    
    # define different range of Courant numbers for implicit methods   
    crange2 = [1.5, 2, 4]

    # plot implicit method on co-located grid for different Courant numbers on same graph as plotted
    # explicit method on co-located grid
    fig_u_Agrid_c, ax_u_Agrid_c, fig_h_Agrid_c, ax_h_Agrid_c = pltfns.plot_multiple_c(\
            fig_u_Agrid_c, ax_u_Agrid_c, fig_h_Agrid_c, ax_h_Agrid_c, ic.initialconditions_cossin,\
            nm.A_grid_implicit_method, crange2, plotparameterrange2,\
            nx = 100, nt = 100, H = 1, g = 1)
    
    fig_u_Agrid_c.savefig('velocity_varying_courant_colocated.png')
    fig_h_Agrid_c.savefig('height_varying_courant_colocated.png')

    fig_u_Agrid_c.suptitle("Velocity, u, for varying Courant numbers using co-located \n\
explicit and implicit schemes", fontsize = 12)
    fig_h_Agrid_c.suptitle("Height, h, for varying Courant numbers using co-located\n\
explicit and implicit schemes", fontsize = 12)
    plt.show()

    # perform same analysis for staggered grid
    
    # initialize plots
    fig_u_Cgrid_c, ax_u_Cgrid_c = plt.subplots()
    fig_h_Cgrid_c, ax_h_Cgrid_c = plt.subplots()
    
    # define range of Courant numbers to test where explicit staggered method becomes unstable
    crangestaggered1 = [0.8, 0.9, 1.005]

    # plot explicit method on staggered grid for different Courant numbers
    fig_u_Cgrid_c, ax_u_Cgrid_c, fig_h_Cgrid_c, ax_h_Cgrid_c = pltfns.plot_multiple_c(\
            fig_u_Cgrid_c, ax_u_Cgrid_c, fig_h_Cgrid_c, ax_h_Cgrid_c, ic.initialconditions_cos,\
            nm.C_grid_explicit, crangestaggered1, plotparameterrange1,\
            nx = 100, nt = 100, H = 1, g = 1)
    
    # plot implicit method on staggered grid for same range of Courant numbers used for 
    # co-located implicit method on same graph as plotted explicit method on staggered grid
    
    fig_u_Cgrid_c, ax_u_Cgrid_c, fig_h_Cgrid_c, ax_h_Cgrid_c = pltfns.plot_multiple_c(\
            fig_u_Cgrid_c, ax_u_Cgrid_c, fig_h_Cgrid_c, ax_h_Cgrid_c, ic.initialconditions_cos,\
            nm.C_grid_implicit_method, crange2, plotparameterrange2,\
            nx = 100, nt = 100, H = 1, g = 1)
    
    fig_u_Cgrid_c.savefig('velocity_varying_courant_staggered.png')
    fig_h_Cgrid_c.savefig('height_varying_courant_staggered.png') 
    
    fig_u_Cgrid_c.suptitle("Velocity, u, for varying Courant number using staggered \n\
explicit and implicit schemes", fontsize = 12)
    fig_h_Cgrid_c.suptitle("Height, h, for varying Courant number using staggered\n\
explicit and implicit schemes", fontsize = 12)
    plt.show()



    
    # by taking a different initial condition it is clear to see that the co-located grid 
    # gives unphysical results for both implicit and explicit methods
    
    print('Initialising Shallow Water equations with different initial condition...')

    # plot initial conditions where u is zero everywhere and h is zero everywhere 
    # apart from one point at the centre where it is one
    
    initialuspike, initialhspike = ic.initialconditions_spike(initialx)
    
    figic1, axic1 = plt.subplots()
    axic1.plot(initialx, initialuspike, 'r--', label = 'Initial u')
    axic1.plot(initialx, initialhspike, 'g-', label = 'Initial h')

    axic1.legend(loc = 'best')
    axic1.set_xlabel("x")
    axic1.set_xlim([min(initialx),max(initialx)])
    axic1.set_ylim([-0.1, 1.1])
        
    figic1.savefig("initial_condition_spike.png")

    axic1.set_title("Initial Condition with spike in h")
    
    plt.show()
    
    # in order to see the phenomenon more clearly use a slightly coarser grid than before

    nx_adapted = 10 # number of points from x = 0 to x = 1
    nt_adapted = 15 # number of time steps in 1 second
    number_plotted = 2 # number of different iterations to be plotted on graph
    
    # initialize plots 
    
    fig_u_spike, ax_u_spike = plt.subplots()
    fig_h_spike, ax_h_spike = plt.subplots()
    
    # change linestyle range
    linestylerange1 = np.array(['-', '-'])
    plotparameterrange1[1] = linestylerange1
    
    # plot solution at various time iterations for an explicit method on a co-located grid 
    # for the initial condition where u is zero everywhere and h is zero everywhere 
    # apart from one point at the centre where it is one
    
    fig_u_spike, fig_h_spike, ax_u_spike, ax_h_spike = pltfns.plot_multiple_iterations(fig_u_spike,\
        ax_u_spike, fig_h_spike, ax_h_spike,ic.initialconditions_spike, nx_adapted, nt_adapted, \
        number_plotted, nm.A_grid_explicit, plotparameterrange1, xmin, xmax, H = 1, g = 1, \
        c = 0.1, plot_meshgrid = True)

    # This is unphysical therefore try with a staggered grid
    
    # change linestyle range
    linestylerange2 = np.array(['-.', '-.'])
    plotparameterrange2[1] = linestylerange2
    
    # plot on the same figure as used for staggered explicit method (and for same initial conditions),
    # solution at various time iterations for an explicit method on staggered grid 
    
    fig_u_spike, fig_h_spike, ax_u_spike, ax_h_spike = pltfns.plot_multiple_iterations(fig_u_spike,\
        ax_u_spike, fig_h_spike, ax_h_spike, ic.initialconditions_spike, nx_adapted, nt_adapted, \
        number_plotted, nm.C_grid_explicit, plotparameterrange2, xmin, xmax, H = 1, g = 1, c = 0.1, \
        staggered = True, plot_meshgrid = True) 

    
    fig_u_spike.savefig("velocity_pike.png")
    fig_h_spike.savefig("height_spike.png")
    
    ax_u_spike.set_title("Velocity, u, calculated using the co-located and staggered explicit schemes \n\
and initial condition of a spike")    
    ax_h_spike.set_title("Height, h, calculated using the co-located and staggered explicit schemes \n\
and initial condition of a spike")
    
    plt.show()
    
    # However a von-neumann stability analysis shows this is unstable for c > 1 therefore try 
    # the following implicit method and check whether solution is still physical

    
    # plot solution at various time iterations for an explicit method on a staggered grid 
    # for the initial condition where u is zero everywhere and h is zero everywhere apart 
    # from one point at the centre where it is one
    
    # initialize plots 
    
    fig1_C_grid_implicit, ax1_C_grid_implicit = plt.subplots()
    fig2_C_grid_implicit, ax2_C_grid_implicit = plt.subplots()
    
    fig1_C_grid_implicit, fig2_C_grid_implicit, ax1_C_grid_implicit, ax2_C_grid_implicit = \
    pltfns.plot_multiple_iterations(fig1_C_grid_implicit, ax1_C_grid_implicit, fig2_C_grid_implicit,\
        ax2_C_grid_implicit, ic.initialconditions_spike, nx_adapted, nt_adapted, \
        number_plotted, nm.C_grid_implicit_method, plotparameterrange1, xmin, xmax, H = 1,\
         g = 1, c = 0.1, staggered = True, plot_meshgrid = True)   

    
    fig1_C_grid_implicit.savefig("velocity_staggered_implicit_spike.png")
    fig2_C_grid_implicit.savefig("height_staggered_implicit_spike.png")
    
    ax1_C_grid_implicit.set_title("Velocity, u, calculated using the staggered implicit scheme \n\
and initial condition of a spike")   
    ax2_C_grid_implicit.set_title("Height, h, calculated using the staggered implicit scheme \n\
and initial condition of a spike")
    
    plt.show()
    
    
    # Finally we examine the error in the numerical method.
    # This also serves as a check that our code for the numerical methods is working 
    # as expected.
    
    # For the initial solutions used so far it is difficult to find the analytic solution.
    # Therefore we use the following initial condition
    
    xmin_1 = -math.pi
    xmax_1 = math.pi
    
    # define meshgrid for x
    initialxcos = np.linspace(xmin_1, xmax_1, nx)
    
    # plot initial condition where u is zero everywhere and h is cos(x)
    initialucos, initialhcos = ic.initialconditions_cos(initialxcos)
    
    figic2, axic2 = plt.subplots()
    axic2.plot(initialxcos, initialucos, 'r--', label = 'Initial u')
    axic2.plot(initialxcos, initialhcos, 'g-', label = 'Initial h')

    axic2.legend(loc = 'best')
    axic2.set_xlabel("x")
    axic2.set_xlim([xmin_1,xmax_1])
    axic2.set_ylim([-1.1, 1.1])
        
    figic2.savefig("initial_condition_cos.png")
    axic2.set_title("Initital Condition where h is cos(x)")

    plt.show()
 
    print('Calculating errors...')

    # set parameters 
    
    H = 1    # mean fluid depth
    g = 1    # acceleration due to gravity
    c = 0.1  # Courant number 
    
    # plot u and h calculated by each different numerical scheme
    
    fig1_analytic, fig2_analytic, ax1_analytic, ax2_analytic, x1 = pltfns.compare_results(\
            ic.initialconditions_cos, nx, nt, xmin_1, xmax_1, H, g, c)

    # calculate width of spacestep and timestep
    dx = (xmax_1 - xmin_1)/nx
    dt = (c*dx)/math.sqrt(g*H)

    # construct analytic solution of shallow water equations for initial condition
    # described by initialconditions_cos

    u = np.zeros_like(x1)
    h = np.zeros_like(x1)

    for i in range(len(x1)):
        u[i] = math.sin(x1[i])*math.sin(dt*nt)
    
    for i in range(len(x1)):
        h[i] = math.cos(x1[i])*math.cos(dt*nt)

    # plot analytic solution on same figure which has results from numerical schemes
    ax1_analytic.plot(x1, u, c = 'black', linestyle = ':', label = "analytic solution")
    ax1_analytic.legend(loc = 'best')

    ax2_analytic.plot(x1, h, c = 'black', linestyle = ':', label = "analytic solution")
    ax2_analytic.legend(loc = 'best', fontsize = 'small')
    
    fig1_analytic.savefig("comparison_with_analytic_u.png")
    fig2_analytic.savefig("comparison_with_analytic_h.png")
    
    ax1_analytic.set_title("Velocity, u, for the initial condition where u is 0 everywhere\n\
and h is cos(x)")
    ax2_analytic.set_title("Height, h, for the initial condition where u is 0 everywhere\n\
and h is cos(x)")

    plt.show()
    
    # This does not provide much clarity as all the solutions are very close together
    
    # therefore instead look at the error between the analytic solution and the numerical method
    
    # first calculate the square of the difference between the analytic solution 
    # and the numerical method at each point (note these are arrays)
    
    dx, dt, error_A_grid_u, error_C_grid_u, error_A_grid_implicit_u, error_C_grid_implicit_u,\
             error_A_grid_h, error_C_grid_h, error_A_grid_implicit_h, error_C_grid_implicit_h,\
             normuAgrid, normuCgrid, normh = errfns.error_calc(ic.initialconditions_cos, nx, \
                                nt, xmin_1, xmax_1, H = 1, g = 1, c = 0.1)
    
    # calculate max error value of u to use to set y-axis limits on graph
    umaxerror =  max(max(error_A_grid_u), max(error_C_grid_u), max(error_A_grid_implicit_u),\
                     max(error_C_grid_implicit_u))
    
    # calculate max error value of h to use to set y-axis limits on graph
    
    hmaxerror =  max(max(error_A_grid_h), max(error_C_grid_h), max(error_A_grid_implicit_h),\
                     max(error_C_grid_implicit_h))
    
    # define x meshgrid
    xerr = np.linspace(xmin_1, xmax_1, nx + 1)
    
    # plot error in u from 4 different methods
    fig1_error, ax1_error = plt.subplots()
    ax1_error.plot(xerr, error_A_grid_u, c = 'firebrick', linewidth = 5, label = "A-grid explicit")
    ax1_error.plot(xerr + dx/2, error_C_grid_u, c= 'orange', linestyle='--', linewidth = 4,\
                   label = "C-grid explicit")
    ax1_error.plot(xerr, error_A_grid_implicit_u, c = 'black', label = "A-grid implicit")
    ax1_error.plot(xerr + dx/2, error_C_grid_implicit_u, c ='gold', linestyle = ':', \
                   linewidth = 7, label = "C-grid implicit")
    
    ax1_error.set_xlim([xmin_1,xmax_1])
    ax1_error.set_ylim([-umaxerror/10, umaxerror + umaxerror/10])
    ax1_error.set_xlabel("x")
    ax1_error.legend(loc=9, fontsize = 'small')
    
    # plot error in h from 4 different methods
    fig2_error, ax2_error = plt.subplots()
    ax2_error.plot(xerr, error_A_grid_h, c = 'firebrick', linewidth = 5, label = "A-grid explicit")
    ax2_error.plot(xerr + dx/2, error_C_grid_h, c= 'orange', linestyle='--', linewidth = 4,\
                   label = "C-grid explicit")
    ax2_error.plot(xerr, error_A_grid_implicit_h, c = 'black', label = "A-grid implicit")
    ax2_error.plot(xerr + dx/2, error_C_grid_implicit_h, c ='gold', linestyle = ':', \
                   linewidth = 7, label = "C-grid implicit")

    ax2_error.set_xlim([xmin_1,xmax_1])
    ax2_error.set_ylim([-hmaxerror/10, hmaxerror + hmaxerror/10])
    ax2_error.set_xlabel("x")
    ax2_error.legend(loc = 1, fontsize = 'small')
    
    fig1_error.savefig("error_in_u.png")
    fig2_error.savefig("error_in_h.png")
    
    ax1_error.set_title("Squared error in velocity, u, for the initial condition\n\
where u is 0 everywhere and h is cos(x)")    
    ax2_error.set_title("Squared error in height, h, for the initial condition\n\
where u is 0 everywhere and h is cos(x)")
    
    plt.show()
    
    
    # calculate the l2 error norm ie. the l2 norm of the difference between 
    # the analytic solution and the numeric solution divided by the l2 norm
    # of the analytic solution.
    # Note the numerator is an l2 norm as the values produced by the error_calc
    # function is the square of the difference
    print("l2 error norm of u for A-grid explicit: %.8f" % \
          (math.sqrt(sum(error_A_grid_u))/normuAgrid))
    print("l2 error norm of u for C-grid explicit: %.8f" % \
          (math.sqrt(sum(error_C_grid_u))/normuCgrid))
    print("l2 error norm of u for A-grid implicit: %.8f" % \
          (math.sqrt(sum(error_A_grid_implicit_u))/normuAgrid))
    print("l2 error norm of u for C-grid implicit: %.8f" % \
          (math.sqrt(sum(error_C_grid_implicit_u))/normuCgrid))

    print("l2 error norm of h for A-grid explicit: %.8f" % \
          (math.sqrt(sum(error_A_grid_h))/normh))
    print("l2 error norm of h for C-grid explicit: %.8f" % \
          (math.sqrt(sum(error_C_grid_h))/normh))
    print("l2 error norm of h for A-grid implicit: %.8f" % \
          (math.sqrt(sum(error_A_grid_implicit_h))/normh))
    print("l2 error norm of h for C-grid implicit: %.8f" % \
          (math.sqrt(sum(error_C_grid_implicit_h))/normh))
    
    
    
    # to further test the numerical methods and that the code is working correctly
    # use a different inital condition which also has an analytical solution

    # plot initial conditions where h is cos(x) + sin(x) and u is cos(x) - sin(x)
    initialucossin, initialhcossin = ic.initialconditions_cossin(initialxcos)

    figic3, axic3 = plt.subplots()
    axic3.plot(initialxcos, initialucossin, 'r--', label = 'Initial u')        
    axic3.plot(initialxcos, initialhcossin, 'g-', label = 'Initial h')

    axic3.legend(loc = 'best')
    axic3.set_xlabel("x")
    axic3.set_xlim([min(initialxcos),max(initialxcos)])
    axic3.set_ylim([-1.5, 1.5])
        
    figic3.savefig("initial_condition_cossin.png")

    axic3.set_title("Initital Condition where h is cos(x) + sin(x) and u is cos(x) - sin(x)")    

    plt.show()
    
    # would like to compare the errors with respect to dx and dt 
    # to do this we make a selection of nx and total time such that nt is an integer
    
    # the total time must be kept constant so that we are comparing the schemes 
    # at the same point in time
    
    # first we use a large Courant number to find the order of the error with respect to dt
    c_1 = 0.5
    total_time_1 = math.pi/12
    
    nx_range_1 = [120, 240, 360, 480]
    nt_range_1 = np.zeros_like(nx_range_1).astype('int')

    # as dx = (xmax - xmin)/nx = 2pi/nx and dt = c*dx/sqrt(gH) = 2pic/nxsqrt(gH),
    # nt = total_time/dt = pi/12 /(2pi c /nxsqrt(gH)) = nx/12

    for j in range(len(nx_range_1)):
        nx_r_1 = nx_range_1[j]
        nt_range_1[j] = nx_r_1/12

    
    # plot the log of dx (or dt) against log of the error norm of u (or h)
    # then calculate the gradient of the curve plotted to find the order of accuracy
    # of u or h with respect to dx or dt
    
    gradient_u_dx_1, gradient_u_dt_1, gradient_h_dx_1, gradient_h_dt_1 = \
        errfns.error_fn(nx_range_1, nt_range_1, total_time_1, xmin_1, \
                               xmax_1, H = 1, g = 1, c = c_1)
    
    plt.show()
    
    # note the error plotted against dx and the error plotted against dt is the same 
    # because as keeping c, nt and nx constant impossible to vary dx without varying dt.
    # Hence the gradients are also the same. Therefore the results should be interpreted 
    # as the error in u or h with respect to dt in both cases as the Courant number is large


    # second we use a small Courant number to find the order of the error with respect to dx
    c_2 = 0.005
    
    # again the total time must be kept constant so that we are comparing the schemes 
    # at the same point in time
    total_time_2 = math.pi
    
    nx_range_2 = [12, 24, 36, 48]
    nt_range_2 = np.zeros_like(nx_range_2).astype('int')

    # as dx = (xmax - xmin)/nx = 2pi/nx and dt = c*dx/sqrt(gH) = 2pic/nxsqrt(gH),
    # nt = total_time/dt = pi /(2pi c /nxsqrt(gH))

    for j in range(len(nx_range_2)):
        nx_r_2 = nx_range_2[j]
        nt_range_2[j] = 100 * nx_r_2

    
    # plot the log of dx against log of the error norm of u (or h)
    # then calculate the gradient of the curve plotted to find the order of accuracy
    # of u or h with respect to dx or dt
    
    gradient_u_dx_2, gradient_u_dt_2, gradient_h_dx_2, gradient_h_dt_2 = \
        errfns.error_fn(nx_range_2, nt_range_2, total_time_2, xmin_1, \
                               xmax_1, H = 1, g = 1, c = c_2)
    
    plt.show()
    
    # note the error plotted against dx and the error plotted against dt is the same 
    # because as keeping c, nt and nx constant impossible to vary dx without varying dt.
    # Hence the gradients are also the same. Therefore the results should be interpreted 
    # as the error in u or h with respect to dx in both cases as the Courant number is small

    # the following orders agree with the taylor expansion apart from the error with
    # respect to dt for C_grid_explicit
    print ("Numerical method| u error vs dx| h error vs dx| u error vs dt| h error vs dt")
    print("A_grid explicit| %f | %f | %f | %f" % (gradient_u_dx_2[0], gradient_h_dx_2[0], \
                                                  gradient_u_dt_1[0], gradient_h_dt_1[0]))
    print("C_grid explicit| %f | %f | %f | %f" % (gradient_u_dx_2[1], gradient_h_dx_2[1], \
                                                  gradient_u_dt_1[1], gradient_h_dt_1[1]))
    print("A_grid implicit| %f | %f | %f | %f" % (gradient_u_dx_2[2], gradient_h_dx_2[2], \
                                                  gradient_u_dt_1[2], gradient_h_dt_1[2]))
    print("C_grid implicit| %f | %f | %f | %f" % (gradient_u_dx_2[3], gradient_h_dx_2[3], \
                                                  gradient_u_dt_1[3], gradient_h_dt_1[3]))

    
    print('Timing code...')
    
    # set number of timesteps and space steps on grid
    
    nx_1 = 1000
    nt_1 = 1000
    
    # Finally we would like to compare the computational cost of each scheme by 
    # comparing how long each scheme takes to run
    
    # this test is carried out for each initial condition
    t0cos, t1cos, t2cos, t3cos, t4cos = pltfns.compare_results(\
        ic.initialconditions_cos, nx_1, nt_1, xmin_1, xmax_1, H, g, c, timing = True)

    t0cossin, t1cossin, t2cossin, t3cossin, t4cossin = pltfns.compare_results(\
        ic.initialconditions_cossin, nx_1, nt_1, xmin_1, xmax_1, H, g, c, timing = True)
    
    t0spike, t1spike, t2spike, t3spike, t4spike = pltfns.compare_results(\
        ic.initialconditions_spike, nx_1, nt_1, xmin_1, xmax_1, H, g, c, timing = True)

    print("cos initial condition")
    
    print("A-grid explicit: %f seconds" % (t1cos - t0cos))
    print("C-grid explicit: %f seconds" % (t2cos - t1cos))
    print("A-grid implicit: %f seconds" % (t3cos - t2cos))
    print("C-grid implicit: %f seconds" % (t4cos - t3cos))
    
    print("cossin initial condition")
    
    print("A-grid explicit: %f seconds" % (t1cossin - t0cossin))
    print("C-grid explicit: %f seconds" % (t2cossin - t1cossin))
    print("A-grid implicit: %f seconds" % (t3cossin - t2cossin))
    print("C-grid implicit: %f seconds" % (t4cossin - t3cossin))
    
    print("spike initial condition")
    
    print("A-grid explicit: %f seconds" % (t1spike - t0spike))
    print("C-grid explicit: %f seconds" % (t2spike - t1spike))
    print("A-grid implicit: %f seconds" % (t3spike - t2spike))
    print("C-grid implicit: %f seconds" % (t4spike - t3spike))
    

    
main()

