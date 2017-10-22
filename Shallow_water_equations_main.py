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
    nx = 20 # number of points from x = 0 to x = 1
    nt = 10 # number of time steps

    # plot initial conditions where u is zero everywhere and h is zero everywhere apart from one point at the centre where it is one
    ic.initialconditions_spike(nx, nt)
    
    # plot solution at various time iterations for an explicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_A_grid, ax2_A_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.A_grid_explicit)
    ax1_A_grid.set_title("Velocity, u, calculated using the colocated explicit scheme")
    ax2_A_grid.set_title("Height, h, calculated using the colocated explicit scheme")
    
    # plot solution at various time iterations for an explicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_C_grid, ax2_C_grid = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.C_grid_explicit)
    ax1_C_grid.set_title("Velocity, u, calculated using the staggered explicit scheme")
    ax2_C_grid.set_title("Height, h, calculated using the staggered explicit scheme")
    
    # plot solution at various time iterations for an implicit method on a colocated grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_implicit, ax2_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.implicit_method)
    ax1_implicit.set_title("Velocity, u, calculated using the colocated implicit scheme")
    ax2_implicit.set_title("Height, h, calculated using the colocated implicit scheme")
    
    # plot solution at various time iterations for a semi-implicit method on a staggered grid for the initial condition where u is zero everywhere 
    # and h is zero everywhere apart from one point at the centre where it is one
    ax1_semi_implicit, ax2_semi_implicit = pltfns.plot_multiple_iterations(ic.initialconditions_spike, nx, nt, nm.semi_implicit_method)
    ax1_semi_implicit.set_title("Velocity, u, calculated using the staggered semi-implicit scheme")
    ax2_semi_implicit.set_title("Height, h, calculated using the staggered semi-implicit scheme")
    
    # plot initial conditions where u is zero everywhere and h has a bump in the centre and is surrounded by zero either side
    #ic.initialconditions_cosbell(nx,nt)
    
main()

