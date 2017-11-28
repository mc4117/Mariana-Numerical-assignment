#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare

This file contains the main code for solving the shallow water equations using the 
numerical schemes. It also tests various properties of the scheme.
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
    
    # define the meshgrid working on
    xmin = 0 # minimum value of x on grid
    xmax = 1 # maximum value of x on grid
    
    nx = 100  # number of points from x = xmin to x = xmax
    nt = 100 # number of time steps
    
    # First attempt a simple initial condition with a colocated forward-backward scheme
    
    # set up a meshgrid on x
    initialx = np.linspace(xmin,xmax,nx+1) 
    # note want extra point at the boundary for plotting reasons but in reality 
    # h[0] and h[nx] are equal and u[0] and u[nx] are equal 
    
    # plot initial conditions where u is zero everywhere and h has a bump in the centre 
    # and is surrounded by zero either side (hereafter referred to as cosbell)
    initialu, initialh = ic.initialconditions_cosbell(initialx)

    figic, axic = plt.subplots()
        
    axic.plot(initialx, initialh, 'g-', label = 'Initial h conditions')
    axic.plot(initialx, initialu, 'r--', label = 'Initial u conditions')
    axic.legend(loc = 'best')
    axic.set_xlabel("x")
    axic.set_title("Initial Condition where h has a bump in the centre (cosbell curve)")
    axic.set_xlim([min(initialx),max(initialx)])
    axic.set_ylim([-0.1, 1.1])
        
    figic.savefig("initial_condition_cosbell.png")
    plt.show()

    # to plot solution first set the colours to be used in the plot
    mymap = plt.get_cmap("YlOrRd")
    colorrange_multiiterations = mymap(np.r_[np.array([0.3, 0.55, 1]), np.array([0.3, 0.55, 1])])

    # then set the linestyles that are used for each line in the plot
    linestylerange = np.array(['-', '--', '-'])

    plotparameterrange = [colorrange_multiiterations, linestylerange]
    
    # now plot solution at various time iterations for an explicit method on a colocated 
    # grid for the initial condition where u is zero everywhere 
    # and h is cosbell curve
    
    fig1, fig2, ax1, ax2 = pltfns.plot_multiple_iterations(ic.initialconditions_cosbell, nx, 3*nt,\
            3, nm.A_grid_explicit, plotparameterrange, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1)
    
    # display legend and title
    ax1.set_title("Velocity, u, calculated using the colocated explicit scheme \n\
and initial condition of a cosbell curve")
    
    ax2.set_title("Height, h, calculated using the colocated explicit scheme \n\
and initial condition of a cosbell curve")
    
    fig1.savefig("velocity_colocated_explicit_cosbell.png")
    fig2.savefig("height_colocated_explicit_cosbell.png")
    
    plt.show()
    
    
    # This seems to work well but through von-neumann stability analysis we find 
    # that this is unstable for c>2
    
    print('Varying Courant numbers...')
    
    # set colorrange as before
    colorrange_multic = plotparameterrange[0]
    
    # plotting explicit method on colocated grid for different courant numbers
    
    crange = np.linspace(0,3,4)
    fig1_A_multiplec, fig2_A_multiplec, ax1_A_multiplec, ax2_A_multiplec = pltfns.plot_multiple_c(\
            ic.initialconditions_cosbell, nm.A_grid_explicit, crange, colorrange_multic, H = 1, g = 1)
    fig1_A_multiplec.suptitle("Velocity, u, for varying Courant number using colocated \
explicit scheme\n and initial condition of cosbell curve", fontsize = 13)
    fig2_A_multiplec.suptitle("Height, h, for varying Courant number using colocated\n\
explicit scheme and initial condition of cosbell curve", fontsize = 13)

    fig1_A_multiplec.savefig("velocity_varying_courant_explicit.png")
    fig2_A_multiplec.savefig("height_varying_courant_explicit.png")
    
    plt.show()
    
    # As this is very unstable try an implict method on a colocated grid which is stable everywhere
    
    fig1_implicit_multiplec, fig2_implicit_multiplec, ax1_implicit_multiplec, \
            ax2_implicit_multiplec = pltfns.plot_multiple_c(ic.initialconditions_cosbell, \
            nm.A_grid_implicit_method, crange, colorrange_multic)
    fig1_implicit_multiplec.suptitle("Velocity, u, for varying courant numbers using colocated \
implicit scheme\n and initial condition of cosbell curve", fontsize = 13)
    fig2_implicit_multiplec.suptitle("Height, h, for varying courant numbers using colocated \
implicit scheme\n and initial condition of cosbell curve", fontsize = 13)
    
    fig1_implicit_multiplec.savefig("velocity_varying_courant_implicit.png")
    fig2_implicit_multiplec.savefig("height_varying_courant_implicit.png")
    
    plt.show()
    
    # by taking a different initial condition it is clear to see that the colocated grid 
    # gives unphysical results for both implicit and explicit methods
    
    print('Initialising Shallow Water equations with different initial condition...')

    # plot initial conditions where u is zero everywhere and h is zero everywhere 
    # apart from one point at the centre where it is one
    
    initialuspike, initialhspike = ic.initialconditions_spike(initialx)
    
    figic1, axic1 = plt.subplots()
    axic1.plot(initialx, initialhspike, 'g-', label = 'Initial h conditions')
    axic1.plot(initialx, initialuspike, 'r--', label = 'Initial u conditions')
    axic1.legend(loc = 'best')
    axic1.set_xlabel("x")
    axic1.set_title("Initial Condition with spike in h")
    axic1.set_xlim([min(initialx),max(initialx)])
    axic1.set_ylim([-0.1, 1.1])
        
    figic1.savefig("initial_condition_spike.png")
    plt.show()
    
    # in order to see the phenomenon more clearly use a slightly coarser grid than before

    nx_adapted = 20 # number of points from x = 0 to x = 1
    nt_adapted = 9 # number of time steps in 1 second
    number_plotted = 3 # number of different iterations to be plotted on graph
    
    # plot solution at various time iterations for an explicit method on a colocated grid 
    # for the initial condition where u is zero everywhere and h is zero everywhere 
    # apart from one point at the centre where it is one
    
    fig1_A_grid, fig2_A_grid, ax1_A_grid, ax2_A_grid = pltfns.plot_multiple_iterations(\
        ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.A_grid_explicit, \
        plotparameterrange, H = 1, g = 1, c = 0.1, plot_meshgrid = True)
    
    ax1_A_grid.set_title("Velocity, u, calculated using the colocated explicit scheme")
    ax2_A_grid.set_title("Height, h, calculated using the colocated explicit scheme")
    
    fig1_A_grid.savefig("velocity_colocated_explicit_spike.png")
    fig2_A_grid.savefig("height_colocated_explicit_spike.png")

    
    # plot solution at various time iterations for an implicit method on a colocated grid 
    # for the initial condition where u is zero everywhere and h is zero everywhere 
    # apart from one point at the centre where it is one
    fig1_implicit, fig2_implicit, ax1_implicit, ax2_implicit = pltfns.plot_multiple_iterations(\
            ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, \
            nm.A_grid_implicit_method, plotparameterrange, H = 1, g = 1, c = 0.1, plot_meshgrid = True)
    ax1_implicit.set_title("Velocity, u, calculated using the colocated implicit scheme") 
    ax2_implicit.set_title("Height, h, calculated using the colocated implicit scheme")
    
    fig1_implicit.savefig("velocity_colocated_implicit_spike.png")
    fig2_implicit.savefig("height_colocated_implicit_spike.png")
    
    plt.show()
    
    # Therefore instead we use a staggered grid
    
    # plot solution at various time iterations for an explicit method on a staggered grid 
    # for the initial condition where u is zero everywhere and h is zero everywhere apart 
    # from one point at the centre where it is one
    
    fig1_C_grid, fig2_C_grid, ax1_C_grid, ax2_C_grid = pltfns.plot_multiple_iterations(\
        ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.C_grid_explicit, \
        plotparameterrange, H = 1, g = 1, c = 0.1, staggered = True, plot_meshgrid = True)   
    ax1_C_grid.set_title("Velocity, u, calculated using the staggered explicit scheme")    
    ax2_C_grid.set_title("Height, h, calculated using the staggered explicit scheme")

    fig1_C_grid.savefig("velocity_staggered_explicit_spike.png")
    fig2_C_grid.savefig("height_staggered_explicit_spike.png")
    
    # However a von-neumann stability analysis shows this is unstable for c > 1 therefore try 
    # the following implicit method and check whether solution is still physical
    
    # plot solution at various time iterations for a implicit method on a staggered grid for 
    # the initial condition where u is zero everywhere and h is zero everywhere apart 
    # from one point at the centre where it is one
    
    fig1_C_grid_implicit, fig2_C_grid_implicit, ax1_C_grid_implicit, ax2_C_grid_implicit = \
    pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, \
        number_plotted, nm.C_grid_implicit_method, plotparameterrange, H = 1, g = 1, c = 0.1, \
        staggered = True, plot_meshgrid = True)   
    ax1_C_grid_implicit.set_title("Velocity, u, calculated using the staggered implicit scheme")   
    ax2_C_grid_implicit.set_title("Height, h, calculated using the staggered implicit scheme")
    
    fig1_C_grid_implicit.savefig("velocity_staggered_implicit_spike.png")
    fig2_C_grid_implicit.savefig("height_staggered_implicit_spike.png")
    
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
    axic2.plot(initialxcos, initialhcos, 'g-', label = 'Initial h conditions')
    axic2.plot(initialxcos, initialucos, 'r--', label = 'Initial u conditions')
    axic2.legend(loc = 'best')
    axic2.set_xlabel("x")
    axic2.set_title("Initital Condition where h is cos(x)")
    axic2.set_xlim([xmin_1,xmax_1])
    axic2.set_ylim([-1.1, 1.1])
        
    figic2.savefig("initial_condition_cos.png")
    plt.show()
 
    print('Calculating errors...')

    # set parameters and number of timesteps and space steps on grid

    nx_1 = 1000
    nt_1 = 1000
    
    H = 1    # mean fluid depth
    g = 1    # acceleration due to gravity
    c = 0.1  # Courant number 
    
    # plot u and h calculated by each different numerical scheme
    
    fig1_analytic, fig2_analytic, ax1_analytic, ax2_analytic, x1 = pltfns.compare_results(\
            ic.initialconditions_cos, nx_1, nt_1, xmin_1, xmax_1, H, g, c)

    # calculate width of spacestep and timestep
    dx = (xmax_1 - xmin_1)/nx_1
    dt = (c*dx)/math.sqrt(g*H)

    # construct analytic solution of shallow water equations for initial condition
    # described by initialconditions_cos

    u = np.zeros_like(x1)
    h = np.zeros_like(x1)

    for i in range(len(x1)):
        u[i] = math.sin(x1[i])*math.sin(dt*nt_1)
    
    for i in range(len(x1)):
        h[i] = math.cos(x1[i])*math.cos(dt*nt_1)

    # plot analytic solution on same figure which has results from numerical schemes
    ax1_analytic.plot(x1, u, c = 'black', linestyle = ':', label = "analytic solution")
    ax1_analytic.set_title("Velocity, u, for the initial condition where u is 0 everywhere\n\
and h is cos(x)")
    ax1_analytic.legend(loc = 'best')

    ax2_analytic.plot(x1, h, c = 'black', linestyle = ':', label = "analytic solution")
    ax2_analytic.set_title("Height, h, for the initial condition where u is 0 everywhere\n\
and h is cos(x)")
    ax2_analytic.legend(loc = 'best', fontsize = 'small')
    
    fig1_analytic.savefig("comparison_with_analytic_u.png")
    fig2_analytic.savefig("comparison_with_analytic_h.png")
    
    # This does not provide much clarity as all the solutions are very close together
    
    # therefore instead look at the error between the analytic solution and the numerical method
    
    # first calculate square of error between the analytic solution and the numerical method
    # at each point (note these are arrays)
    
    dx, dt, error_A_grid_u, error_C_grid_u, error_A_grid_implicit_u, error_C_grid_implicit_u,\
             error_A_grid_h, error_C_grid_h, error_A_grid_implicit_h, error_C_grid_implicit_h,\
             normuAgrid, normuCgrid, normh = errfns.error_calc(ic.initialconditions_cos, nx_1, \
                                nt_1, xmin = -math.pi, xmax = math.pi, H = 1, g = 1, c = 0.1)
    
    # calculate max error value of u to use to set y-axis limits on graph
    umaxerror =  max(max(error_A_grid_u), max(error_C_grid_u), max(error_A_grid_implicit_u),\
                     max(error_C_grid_implicit_u))
    
    # calculate max error value of h to use to set y-axis limits on graph
    
    hmaxerror =  max(max(error_A_grid_h), max(error_C_grid_h), max(error_A_grid_implicit_h),\
                     max(error_C_grid_implicit_h))
    
    # define x meshgrid
    xerr = np.linspace(xmin_1, xmax_1, nx_1 + 1)
    
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
    ax1_error.set_title("Squared error in velocity, u, for the initial condition\n\
where u is 0 everywhere and h is cos(x)" )
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
    ax2_error.set_title("Squared error in height, h, for the initial condition\n\
where u is 0 everywhere and h is cos(x)" )
    ax2_error.legend(loc = 1, fontsize = 'small')
    
    fig1_error.savefig("error_in_u.png")
    fig2_error.savefig("error_in_h.png")
    
    plt.show()
    
    
    # calculate the l2 error norm ie. the l2 norm of the difference between 
    # the analytic solution and the numeric solution normalised by the l2 norm
    # of the analytic solution.
    # Note the numerator is an l2 norm as the values produced by the error_calc
    # function is the squared error.
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
        
    axic3.plot(initialxcos, initialhcossin, 'g-', label = 'Initial h conditions')
    axic3.plot(initialxcos, initialucossin, 'r--', label = 'Initial u conditions')
    axic3.legend(loc = 'best')
    axic3.set_xlabel("x")
    axic3.set_title("Initital Condition where h is cos(x) + sin(x) and u is cos(x) - sin(x)")
    axic3.set_xlim([min(initialxcos),max(initialxcos)])
    axic3.set_ylim([-1.5, 1.5])
        
    figic3.savefig("initial_condition_cossin.png")
    plt.show()
    
    # would like to compare the errors with respect to dx and dt 
    # to do this we make a selection of nx and total time such that nt is an integer
    
    # the total time must be kept constant so that we are comparing the schemes 
    # at the same point in time
    
    # first we use a large Courant number to find the order of the error with respect to dt
    c_1 = 0.1
    total_time_1 = math.pi/12
    
    nx_range_1 = [120, 240, 360, 480]
    nt_range_1 = np.zeros_like(nx_range_1).astype('int')

    # as dx = (xmax - xmin)/nx = 2pi/nx and dt = c*dx/sqrt(gH) = 2pic/nxsqrt(gH),
    # nt = total_time/dt = pi/12 /(2pi c /nxsqrt(gH)) = 5 nx/12

    for j in range(len(nx_range_1)):
        nx_r_1 = nx_range_1[j]
        nt_range_1[j] = 5 * nx_r_1/12

    
    # plot the log of dx (or dt) against log of the error norm of u (or h)
    # then calculate the gradient of the curve plotted to find the order of accuracy
    # of u or h with respect to dx or dt
    
    gradient_u_dx_1, gradient_u_dt_1, gradient_h_dx_1, gradient_h_dt_1 = \
        errfns.error_fn(nx_range_1, nt_range_1, total_time_1, xmin = -math.pi, \
                               xmax = math.pi, H = 1, g = 1, c = c_1)
    
    plt.show()
    
    # note the error plotted against dx and the error plotted against dt is the same
    # because as keeping the c, nt and nx constant impossible to vary dx without varying dt.
    # Hence the gradients are also the same


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
        errfns.error_fn(nx_range_2, nt_range_2, total_time_2, xmin = -math.pi, \
                               xmax = math.pi, H = 1, g = 1, c = c_2)
    
    plt.show()
    # note the error plotted against dx and the error plotted against dt is the same
    # because as keeping the c, nt and nx constant impossible dx without varying dt.
    # Hence the gradients are also the same

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
    
    # Finally we would like to compare the computational cost of each scheme by 
    # comparing how long each scheme takes to run
    t0, t1, t2, t3, t4 = pltfns.compare_results(ic.initialconditions_cossin, nx_1, \
        nt_1, xmin_1, xmax_1, H, g, c, timing = True)

    

    print("A-grid explicit: %f seconds" % (t1 - t0))
    print("C-grid explicit: %f seconds" % (t2 - t1))
    print("A-grid implicit: %f seconds" % (t3 - t2))
    print("C-grid implicit: %f seconds" % (t4 - t3))
    
main()

