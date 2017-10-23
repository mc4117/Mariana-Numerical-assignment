#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import initial_conditions as ic
import numerical_methods as nm
import plotting_functions as pltfns



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
    plt.rcParams['axes.titlepad'] = 20 
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
    crange_A = np.linspace(0,0.75,4)
    ax1_A_multiplec, ax2_A_multiplec = pltfns.plot_multiple_c(ic.initialconditions_cosbell, nm.A_grid_explicit, crange_A, colorrange_multic)
    ax1_A_multiplec.set_title("Velocity, u, for varying courant numbers calculated using the colocated explicit scheme \n and initial condition of a cos bell curve")
    ax2_A_multiplec.set_title("Height, h, for varying courant numbers calculated using the colocated explicit scheme \n and initial condition of a cos bell curve")

    # Therefore we try an implict method on a colocated grid which is stable everywhere
    crange_implicit = np.linspace(0,3,4)
    ax1_implicit_multiplec, ax2_implicit_multiplec = pltfns.plot_multiple_c(ic.initialconditions_cosbell, nm.implicit_method, crange_implicit, colorrange_multic)
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
    ax1_C_grid, ax2_C_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.C_grid_explicit, plotparameterrange)
    ax1_C_grid.set_title("Velocity, u, calculated using the staggered explicit scheme")
    ax2_C_grid.set_title("Height, h, calculated using the staggered explicit scheme")
    
    # However a von-neumann stability analysis shows this is unstable for c > 1 therefore try the following semi-implicit method
    
    # plot solution at various time iterations for a semi-implicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_semi_implicit, ax2_semi_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx_adapted, nt_adapted, number_plotted, nm.semi_implicit_method, plotparameterrange)
    ax1_semi_implicit.set_title("Velocity, u, calculated using the staggered semi-implicit scheme")
    ax2_semi_implicit.set_title("Height, h, calculated using the staggered semi-implicit scheme")
    
    # Finally we present the following results - all 4 schemes for the two initial conditions discussed and a further third initial condition
    
    ##### CODE MISSING OF FINAL 4 GRAPHS!!!
    
main()

