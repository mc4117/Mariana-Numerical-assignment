#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""

import numpy as np
from math import sqrt

def A_grid_explicit(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using an explicit method on the colocated scheme

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """
    
    
    # set initial conditions
    initialu, initialh, x = initialconditions(nx, xmin, xmax, plot = False)
    
    # initialize the system
    uOld = initialu.copy()
    hOld = initialh.copy()
    u = np.zeros_like(uOld)
    h = np.zeros_like(hOld)


    # loop over time steps
    for it in range(nt): 
        for y in range(nx):
            # forward in time and centred in space
            u[y] = uOld[y] - 0.5*c*sqrt(g/H)*(hOld[(y+1)%nx] - hOld[(y-1)%nx])
        for y in range(nx):    
            # backward in time and centred in space
            h[y] = hOld[y] - 0.5*c*sqrt(H/g)*(u[(y+1)%nx] - u[(y-1)%nx]) # backward in time and centred in space 
        
        # as would like to plot from 0 to 1 set the value of u and h at end point using periodic boundary conditions
        u[nx] = u[0]
        h[nx] = h[0]
        
        # copy u and h for next iteration
        hOld = h.copy()
        uOld = u.copy()
    return u, h, x

def C_grid_explicit(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using an explicit method on the staggered scheme

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """
    # set initial conditions
    initialu, initialh, x = initialconditions(nx, xmin, xmax, plot = False)
    
    
    uhalf = np.zeros(len(initialu))
    
    # for a c-grid the velocity u is stagerred in the x-direction by half
    for i in range(nx + 1):
        uhalf[i] = (initialu[i] + initialu[(i+1)%nx])/2
        # therefore uhalf[i] = u_{i + 1/2}
    
    # initialize the system
    h_cgrid = np.zeros_like(initialh)
    
    uOld = uhalf.copy()
    hOld = initialh.copy()

    
    # loop over time steps
    for it in range(nt): 
        for y in range(nx):
            # forward in time and centred in space
            uhalf[y] = uOld[y] - (c*sqrt(g/H))*(hOld[(y + 1)%nx] - hOld[y%nx]) 
        for y in range(nx):
        #backward in time and centred in space
            h_cgrid[y] = hOld[y] - (c*sqrt(H/g))*(uhalf[y%nx] - uhalf[(y-1)%nx]) 

        # as would like to plot from 0 to 1 set the value of u and h at end point using periodic boundary conditions
        # however as the grid is staggered uhalf[nx] = u_{nx + 1/2}. Therefore when plot uhalf need to remove last point
        uhalf[nx] = uhalf[0]
        h_cgrid[nx] = h_cgrid[0]
        
        # copy u and h for next iteration
        hOld = h_cgrid.copy()
        uOld = uhalf.copy()
    return uhalf, h_cgrid, x

def implicit_method(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using the colocated scheme with an implicit method
        Both equations are discretised as backward in time and centred in space.

    initial conditions: function which specifies the initial conditions for the system 
    nx:                 number of space steps
    nt:                 number of time steps
    xmin:               minimum value of x on grid
    xmax:               maximum value of x on grid
    H:                  mean fluid depth set to 1 unless otherwise specified
    g:                  acceleration due to gravity scaled to 1
    c:                  courant number (c = root(gH)dt/dx)
    """
    
    # set initial conditions
    initialu, initialh, x = initialconditions(nx, xmin, xmax, plot = False)
    uOld = initialu.copy()
    hOld = initialh.copy()

    # construct matrix to solve implicit method matrix equation
    # as matrix constructed is not dependent on time, only needs to be constructed once
    
    # for plotting reasons we have chosen to include the end point in the scheme
    # hence the matrix has the following dimensions: 
    matrix = np.zeros((nx+ 1,nx + 1))
    
    # because of the way the matrix has been constructed with a modulus operator we still have periodic boundaries:
    for i in range(nx+1):    
        matrix[i,i] = 1 + c**2/2
        matrix[i, (i-2)%nx] = -(c**2)/4
        matrix[i, (i+2)%nx] = -(c**2)/4
    
    # find inverse of this matrix
    inverse = np.linalg.inv(matrix)
    
    # loop over timesteps
    for it in range(nt):      
        
        uvector = np.zeros_like(uOld)
        for i in range(nx+1):
            uvector[i] = sqrt(H/g)*(c/2)*(uOld[(i+1)%nx] - uOld[(i-1)%nx])
        # note here that uvector[nx] = uvector[0] because of modulus operator so periodic boundaries are still kept   
            
        # solve matrix equation to find h
        h = np.dot(inverse, (hOld - uvector))
        h[nx] = h[0]
    
        hvector = np.zeros_like(hOld)
        for i in range(nx):
            hvector[i] = sqrt(g/H)*(c/2)*(hOld[(i+1)%nx] - hOld[(i-1)%nx])
        # note here that hvector[nx] = hvector[0] because of modulus operator so periodic boundaries are still kept   

        # solve matrix equation to find u
        u = np.dot(inverse, (uOld - hvector))
        u[nx] = u[0]
        
        # copy u and h for next iteration
        hOld = h.copy()
        uOld = u.copy()
    return u, h, x

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
    
    # set initial conditions
    initialu, initialh, x = initialconditions(nx, xmin, xmax, plot = False)
    
    uhalf = np.zeros(len(initialu))
    
    # for a c-grid the velocity u is stagerred in the x-direction by half
    for i in range(nx + 1):
        uhalf[i] = (initialu[i] + initialu[(i+1)%nx])/2
    # therefore uhalf[i] = u_{i + 1/2}
    
    # initialize the system
    u_semi_implicit = np.zeros_like(uhalf)
    h_semi_implicit = np.zeros_like(initialh)

    uOld = uhalf.copy()
    hOld = initialh.copy()
    
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
            semi_implicit_uvector[i] = -sqrt(g/H)*c*(hOld[(i + 1)%nx] - hOld[i%nx]) + ((c**2)/4)*uOld[(i+1)%nx] + (1-(c**2)/2)*uOld[i%nx] + ((c**2)/4)*uOld[(i-1)%nx]
        # note here that semi_implicit_uvector[nx] = semi_implicit_uvector[0] because of modulus operator so periodic boundaries are still kept
        
        # solve matrix equation to find u
        u_semi_implicit = np.linalg.solve(matrix, semi_implicit_uvector)

        semi_implicit_hvector = np.zeros_like(hOld)
        for i in range(nx + 1):
            semi_implicit_hvector[i] = -sqrt(H/g)*c*(uOld[i%nx] - uOld[(i-1)%nx]) + ((c**2)/4)*hOld[(i+1)%nx] + (1-(c**2)/2)*hOld[(i)%nx] + ((c**2)/4)*hOld[(i-1)%nx]
        # note here that semi_implicit_hvector[nx] = semi_implicit_hvector[0] because of modulus operator so periodic boundaries are still kept
        
        # solve matrix equation to find h
        h_semi_implicit = np.linalg.solve(matrix, semi_implicit_hvector)

        # copy u and h for next iteration
        uOld = u_semi_implicit.copy()
        hOld = h_semi_implicit.copy()

    return u_semi_implicit, h_semi_implicit, x