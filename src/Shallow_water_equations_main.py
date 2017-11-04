#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
import initial_conditions as ic
import numerical_methods as nm
import plotting_functions as pltfns

plt.rcParams.update({'figure.max_open_warning': 0})

def main():
    
    # defining the grid and mesh we are working on
    nx = 60  # number of points from x = xmin to x = xmax
    xmin = 0 # minimum value of x on grid
    xmax = 1 # maximum value of x on grid
    nt = 100 # number of time steps in 1 second
    
    # First we attempt a simple initial condition with a colocated forward-backward scheme
    
    # plot initial conditions where u is zero everywhere and h has a bump in the centre and is surrounded by zero either side
    ic.initialconditions_cosbell(nx,nt, xmin, xmax, plot = True)

    # plot solution at various time iterations for an explicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h has a bump in the centre and is surrounded by zero either side
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    timerange = np.linspace(0, 2, 4)
    for i in timerange:
        u, h, x = nm.A_grid_explicit(ic.initialconditions_cosbell, nx, i*nt)
        ax1.plot(x, u, label = 'u after ' + str(int(i*nt)) + ' timesteps')
        ax2.plot(x, h, label = 'h after ' + str(int(i*nt)) + ' timesteps')
        
    
    # add space between the title and the plot
    #plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
    
    ax1.legend(loc = 'best')
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")
    ax1.set_title("Velocity, u, calculated using the colocated explicit scheme \n and initial condition of a cos bell curve")

    ax2.legend(loc = 'best')
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel("x")
    ax2.set_title("Height, h, calculated using the colocated explicit scheme \n and initial condition of a cos bell curve")
    
    
    # This seems to work well but through von-neumann stability analysis we find that this is unstable for c>2
    
    mymap = plt.get_cmap("YlOrRd")
    colorrange_multic = mymap(np.r_[np.array([0.3, 0.55, 1]), np.array([0.3, 0.55, 1])])
    
    # plotting for different courant numbers we see this solution is very unstable
    crange = np.linspace(0,3,4)
    ax1_A_multiplec, ax2_A_multiplec = pltfns.plot_multiple_c(ic.initialconditions_cosbell, nm.A_grid_explicit, crange, colorrange_multic)
    ax1_A_multiplec.set_title("Velocity, u, for varying courant numbers calculated using the colocated explicit scheme \n and initial condition of a cos bell curve")
    ax2_A_multiplec.set_title("Height, h, for varying courant numbers calculated using the colocated explicit scheme \n and initial condition of a cos bell curve")

    # Therefore we try an implict method on a colocated grid which is stable everywhere
    ax1_implicit_multiplec, ax2_implicit_multiplec = pltfns.plot_multiple_c(ic.initialconditions_cosbell, nm.implicit_method, crange, colorrange_multic)
    ax1_implicit_multiplec.set_title("Velocity, u, for varying courant numbers \n calculated using the colocated implicit scheme \n and initial condition of a cos bell curve")
    ax2_implicit_multiplec.set_title("Height, h, for varying courant numbers \n calculated using the colocated implicit scheme \n and initial condition of a cos bell curve")
    
    # by taking a differnt initial condition it is clear to see that the colocated grid gives unphysical 
    # results for both implicit and explicit methods

    # plot initial conditions where u is zero everywhere and h is zero everywhere apart from one point at the centre where it is one
    
    ic.initialconditions_spike(nx, nt)
    
    # In order to see the phenomenon more clearly we use a slightly coarser grid than before
    
    nx_adapted = 20 # number of points from x = 0 to x = 1
    nt_adapted = 10 # number of time steps in 1 second
    number_plotted = 3 # number of different iterations to be plotted on graph
    
    # set the colours to be used in the plot
    mymap = plt.get_cmap("YlOrRd")
    colorrange_multiiterations = mymap(np.r_[np.array([0.3, 0.55, 1]), np.array([0.3, 0.55, 1])])

    # set the linestyles that are used for each line in the plot
    linestylerange = np.array(['-', '--', '-'])

    plotparameterrange = [colorrange_multiiterations, linestylerange]

    # plot solution at various time iterations for an explicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_A_grid, ax2_A_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.A_grid_explicit, plotparameterrange)
    ax1_A_grid.set_title("Velocity, u, calculated using the colocated explicit scheme")
    ax2_A_grid.set_title("Height, h, calculated using the colocated explicit scheme")
    
    # plot solution at various time iterations for an implicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_implicit, ax2_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.implicit_method, plotparameterrange)
    ax1_implicit.set_title("Velocity, u, calculated using the colocated implicit scheme")
    ax2_implicit.set_title("Height, h, calculated using the colocated implicit scheme")
    
    # Therefore instead we use a staggered grid
    
    # plot solution at various time iterations for an explicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_C_grid, ax2_C_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.C_grid_explicit, plotparameterrange, staggered = True)
    ax1_C_grid.set_title("Velocity, u, calculated using the staggered explicit scheme")
    ax2_C_grid.set_title("Height, h, calculated using the staggered explicit scheme")
    
    # However a von-neumann stability analysis shows this is unstable for c > 1 therefore try the following semi-implicit method
    
    # plot solution at various time iterations for a semi-implicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_semi_implicit, ax2_semi_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.semi_implicit_method, plotparameterrange, staggered = True)
    ax1_semi_implicit.set_title("Velocity, u, calculated using the staggered semi-implicit scheme")
    ax2_semi_implicit.set_title("Height, h, calculated using the staggered semi-implicit scheme")
    
    # Finally we examine the error in the numerical method
    # For the initial solutions used so far it is difficult to find the exact solution.
    # Therefore we use the following initial condition
    
    xmin = -math.pi
    xmax = math.pi
    
    # plot initial condition where u is zero everywhere and h is cos(x)
    ic.initialconditions_cos(nx, nt, xmin, xmax)
 
    # set parameters and number of timesteps and space steps on grid

    nx_1 = 100
    nt_1 = 400
    
    H = 1
    g = 1
    c = 0.1
    
    # results of all 4 methods
    
    ax1_exact, ax2_exact, x1 = pltfns.compare_results(ic.initialconditions_cos, nx_1, nt_1, xmin, xmax, H, g, c)

    # calculate width of spacestep and timestep
    dx = (xmax - xmin)/nx_1
    dt = (c*dx)/math.sqrt(g*H)

    # constructing exact solution

    u = np.zeros_like(x1)
    h = np.zeros_like(x1)

    # u = sin(x)sin(t)
    for i in range(len(x1)):
        u[i] = math.sin(x1[i])*math.sin(dt*nt_1)
    
    # h = cos(x)cos(t)
    for i in range(len(x1)):
        h[i] = math.cos(x1[i])*math.cos(dt*nt_1)

    # plot exact solution on plot as well
    ax1_exact.plot(x1, u, c = 'black', linestyle = ':', label = "exact solution")
    ax1_exact.set_title("Velocity, u, for the initial condition where u is 0 everywhere and h is " r"$\cos(x)$" )
    ax1_exact.legend(loc = 'best')

    ax2_exact.plot(x1, h, c = 'black', linestyle = ':', label = "exact solution")
    ax2_exact.set_title("Height, h, for the initial condition where u is 0 everywhere and h is " r"$\cos(x)$" )
    ax2_exact.legend(loc = 'best')
    
    # This does not provide much clarity as all the solutions are very close together
    
    # therefore instead look at the error between the exact solution and the numerical method
    
    ax1_error, ax2_error = pltfns.error_fn(nx_1, nt_1, xmin, xmax, H, g, c)

    ax1_error.set_title("Squared error in velocity, u, for the initial condition where u is 0 everywhere and h is " r"$\cos(x)$" )
    ax1_error.legend(loc = 'best')

    ax2_error.set_title("Squared error in height, h, for the initial condition where u is 0 everywhere and h is " r"$\cos(x)$" )
    ax2_error.legend(loc = 'best')
    
main()

