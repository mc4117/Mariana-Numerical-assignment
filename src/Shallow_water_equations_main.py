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
        u, h, x1 = nm.A_grid_explicit(ic.initialconditions_cosbell, nx, i*nt)
        ax1.plot(x1, h, label = 'h after ' + str(int(i*nt)) + ' timesteps')
        ax2.plot(x1, u, label = 'u after ' + str(int(i*nt)) + ' timesteps')
    
    # add space between the title and the plot
    plt.rcParams['axes.titlepad'] = 20 
    # increase the font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label]):
        item.set_fontsize(15)
    
    ax1.legend(loc = 'best')
    ax1.set_xlim([xmin,xmax])
    ax1.set_xlabel("x")
    ax1.set_title("Velocity, u, calculated using the colocated explicit scheme \n and initial condition of a cos curve")

    ax2.legend(loc = 'best')
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel("x")
    ax2.set_title("Height, h, calculated using the colocated explicit scheme \n and initial condition of a cos curve")
    
    
    # This seems to work well but through von-neumann stability analysis we find that this is unstable for c>2
    
    ##### CODE WITH DIFFERENT C!!!!
    
    # Therefore we try an implict method on a colocated grid which is stable everywhere
    
    #### CODE WITH DIFFERENT C!!!!
    
    # by taking a differnt initial condition it is clear to see that the colocated grid gives unphysical 
    # results for both implicit and explicit methods

    # plot initial conditions where u is zero everywhere and h is zero everywhere apart from one point at the centre where it is one
    
    ic.initialconditions_spike(nx, nt)
    
    # In order to see the phenomenon more clearly we use a slightly coarser grid than before
    
    nx = 20 # number of points from x = 0 to x = 1
    nt = 10 # number of time steps in 1 second

    # plot solution at various time iterations for an explicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_A_grid, ax2_A_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.A_grid_explicit)
    ax1_A_grid.set_title("Velocity, u, calculated using the colocated explicit scheme")
    ax2_A_grid.set_title("Height, h, calculated using the colocated explicit scheme")
    
    # plot solution at various time iterations for an implicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_implicit, ax2_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.implicit_method)
    ax1_implicit.set_title("Velocity, u, calculated using the colocated implicit scheme")
    ax2_implicit.set_title("Height, h, calculated using the colocated implicit scheme")
    
    # Therefore instead we use a staggered grid
    
    # plot solution at various time iterations for an explicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_C_grid, ax2_C_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.C_grid_explicit)
    ax1_C_grid.set_title("Velocity, u, calculated using the staggered explicit scheme")
    ax2_C_grid.set_title("Height, h, calculated using the staggered explicit scheme")
    
    # However a von-neumann stability analysis shows this is unstable for c > 1 therefore try the following semi-implicit method
    
    # plot solution at various time iterations for a semi-implicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_semi_implicit, ax2_semi_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.semi_implicit_method)
    ax1_semi_implicit.set_title("Velocity, u, calculated using the staggered semi-implicit scheme")
    ax2_semi_implicit.set_title("Height, h, calculated using the staggered semi-implicit scheme")
    
    # Finally we present the following results - all 4 schemes for the two initial conditions discussed and a further third initial condition
    
    ##### CODE MISSING OF FINAL 4 GRAPHS!!!
    
main()

