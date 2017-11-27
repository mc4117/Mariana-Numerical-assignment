#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare

Four functions which simulate the shallow water equations 
using four different numerical schemes.

"""

import numpy as np
from math import sqrt
import numbers

def A_grid_explicit(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using an explicit method 
       on the colocated scheme

    Inputs:
        initial conditions: function which specifies the initial conditions for the system 
        nx:                 number of space steps
        nt:                 number of time steps
        xmin:               minimum value of x on grid
        xmax:               maximum value of x on grid
        H:                  mean fluid depth set to 1 unless otherwise specified
        g:                  acceleration due to gravity scaled to 1 unless otherwise specified
        c:                  Courant number (c = root(gH)dt/dx)
        
    Outputs:
        u:                  array containing velocity solution of shallow water equations
                            found by A-grid explicit method
        h:                  array containing height solution of shallow water equations
                            found by A-grid explicit method
        x:                  array of x-meshgrid
        
    """
    
    # end function if nt is not an integer
    if isinstance(nt, numbers.Integral) == False:
        raise ValueError('nt is not an integer') 
    
    # want extra point at the boundary for plot but in practice
    # u[0] and u[nx], and h[0] and h[nx] are equal
    x = np.linspace(xmin,xmax,nx+1) 
    
    # set initial conditions
    initialu, initialh = initialconditions(x)
    
    # initialize the system
    uOld = initialu.copy()
    hOld = initialh.copy()
    u = np.zeros_like(uOld)
    h = np.zeros_like(hOld)

    
    # loop over time steps
    for it in range(nt): 
        for i in range(nx):
            # forward in time and centred in space
            u[i] = uOld[i] - 0.5*c*sqrt(g/H)*(hOld[(i+1)%nx] - hOld[(i-1)%nx])
        for j in range(nx):    
            # backward in time and centred in space
            h[j] = hOld[j] - 0.5*c*sqrt(H/g)*(u[(j+1)%nx] - u[(j-1)%nx])
        
        # as would like to plot from 0 to 1 set the value of u and h at end point 
        # using periodic boundary conditions
        
        u[nx] = u[0]
        h[nx] = h[0]
        
        # copy u and h for next iteration
        hOld = h.copy()
        uOld = u.copy()
    
    return u, h, x


def C_grid_explicit(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using an explicit method on 
        the staggered grid

    Inputs:
        initial conditions: function which specifies the initial conditions for the system 
        nx:                 number of space steps
        nt:                 number of time steps
        xmin:               minimum value of x on grid
        xmax:               maximum value of x on grid
        H:                  mean fluid depth set to 1 unless otherwise specified
        g:                  acceleration due to gravity scaled to 1 unless otherwise specified
        c:                  Courant number (c = root(gH)dt/dx)
    
    Outputs:
        uhalf:              array containing velocity solution of shallow water equations
                            found by C-grid explicit method
        h:                  array containing height solution of shallow water equations
                            found by C-grid explicit method
        x:                  array of x-meshgrid
        
    """
    
    # end function if nt is not an integer
    if isinstance(nt, numbers.Integral) == False:
        raise ValueError('nt is not an integer') 
    
    # want extra point at the boundary for plot but in practice
    # u[0] and u[nx], and h[0] and h[nx] are equal
    x = np.linspace(xmin,xmax,nx+1) 
    
    # define x on staggered meshgrid where the grid is shifted by half dx
    xhalf = x + (xmax-xmin)/(2*nx)


    # set initial conditions on the staggered grid
    initialuAgrid, initialhAgrid = initialconditions(x)
    initialuCgrid, initialhCgrid = initialconditions(xhalf)

    
    # initialize the system
    h = np.zeros_like(initialhAgrid)
    uhalf = np.zeros_like(initialuCgrid)
    
    uOld = initialuCgrid.copy()
    hOld = initialhAgrid.copy()

    
    # loop over time steps
    for it in range(nt): 
        for i in range(nx):
            # forward in time and centred in space
            uhalf[i] = uOld[i] - c*sqrt(g/H)*(hOld[(i + 1)%nx] - hOld[i]) 
            # backward in time and centred in space
            h[i] = hOld[i] - c*sqrt(H/g)*(uhalf[i] - uhalf[(i-1)%nx]) 

        # as would like to plot from 0 to 1 set the value of u and h at end point 
        # using periodic boundary conditions
        uhalf[nx] = uhalf[0]
        h[nx] = h[0]
        
        # copy u and h for next iteration
        hOld = h.copy()
        uOld = uhalf.copy()
        
    return uhalf, h, x


def A_grid_implicit_method(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using the implicit method on 
        a colocated grid.
        Both equations are discretised as backward in time and centred in space.

    Inputs:   
        initial conditions: function which specifies the initial conditions for the system 
        nx:                 number of space steps
        nt:                 number of time steps
        xmin:               minimum value of x on grid
        xmax:               maximum value of x on grid
        H:                  mean fluid depth set to 1 unless otherwise specified
        g:                  acceleration due to gravity scaled to 1 unless otherwise specified
        c:                  Courant number (c = root(gH)dt/dx)
        
    Outputs:
        u:                  array containing velocity solution of shallow water equations
                            found by A-grid implicit method
        h:                  array containing height solution of shallow water equations
                            found by A-grid implicit method
        x:                  array of x-meshgrid
        
    """
    
    # end function if nt is not an integer
    if isinstance(nt, numbers.Integral) == False:
        raise ValueError('nt is not an integer') 
    
    # want extra point at the boundary for plot but in practice
    # u[0] and u[nx], and h[0] and h[nx] are equal
    x = np.linspace(xmin,xmax,nx+1) 
    
    # set initial conditions
    initialu, initialh = initialconditions(x)
    
    uOld = initialu.copy()
    hOld = initialh.copy()

    # construct matrix to solve implicit method matrix equation
    # as matrix constructed is not dependent on time, only needs to be constructed once
    
    # for plotting reasons we have chosen to include the end point in the scheme
    # hence the matrix has the following dimensions: 
    matrix = np.zeros((nx+ 1,nx + 1))

    
    for i in range(nx+1):    
        matrix[i,i] = 1 + c**2/2
        matrix[i, (i-2)%nx] = -c**2/4
        matrix[i, (i+2)%nx] = -c**2/4
       
    # note because of the way the matrix has been constructed with a modulus operator 
    # still have periodic boundaries
    
    # find inverse of this matrix
    inverse = np.linalg.inv(matrix)
    
    # loop over timesteps
    for it in range(nt):      
        # construct vector uvector such that matrix*u = uvector 
        uvector = np.zeros_like(uOld)
        
        for i in range(nx + 1):
            uvector[i] = uOld[i] - 0.5*c*sqrt(g/H)*(hOld[(i+1)%nx] - hOld[(i-1)%nx])
        
        # note here that uvector[nx] = uvector[0] because of modulus operator so 
        # periodic boundaries are still kept   

        # solve matrix equation to find u
        u = np.dot(inverse, uvector)
        
        # construct vector hvector such that matrix*h = hvector         
        hvector = np.zeros_like(hOld)
        
        for i in range(nx+1):
            hvector[i] = hOld[i] - 0.5*c*sqrt(H/g)*(uOld[(i+1)%nx] - uOld[(i-1)%nx])
            
        # note here that hvector[nx] = hvector[0] because of modulus operator so 
        # periodic boundaries are still kept   
            
        # solve matrix equation to find h
        h = np.dot(inverse, hvector)
    

        
        
        # copy u and h for next iteration
        hOld = h.copy()
        uOld = u.copy()
        
    return u, h, x

def C_grid_implicit_method(initialconditions, nx, nt, xmin = 0, xmax = 1, H = 1, g = 1, c = 0.1):
    """This function simulates the shallow water equations using the implicit method 
       on a staggered grid.
       Both equations are discretised using the theta method using theta = 1/2 
       (ie. crank nicholson) and centred in space.

    Inputs:
        initial conditions: function which specifies the initial conditions for the system 
        nx:                 number of space steps
        nt:                 number of time steps
        xmin:               minimum value of x on grid
        xmax:               maximum value of x on grid
        H:                  mean fluid depth set to 1 unless otherwise specified
        g:                  acceleration due to gravity scaled to 1 unless otherwise specified
        c:                  Courant number (c = root(gH)dt/dx)

    Outputs:
        u:                  array containing velocity solution of shallow water equations
                            found by C-grid implicit method
        h:                  array containing height solution of shallow water equations
                            found by C-grid implicit method
        x:                  array of x-meshgrid
        
    """    
    
    # end function if nt is not an integer
    if isinstance(nt, numbers.Integral) == False:
        raise ValueError('nt is not an integer') 
    
    # want extra point at the boundary for plot but in practice
    # u[0] and u[nx], and h[0] and h[nx] are equal
    x = np.linspace(xmin,xmax,nx+1) 
    
    # define x on staggered meshgrid where the grid is shifted by half dx
    xhalf = x + (xmax - xmin)/(2*nx)


    # set initial conditions
    initialuAgrid, initialhAgrid = initialconditions(x)
    initialuCgrid, initialhCgrid = initialconditions(xhalf)

    
    # initialize the system
    
    uOld = initialuCgrid.copy()
    hOld = initialhAgrid.copy()
    
    # for plotting reasons we have chosen to include the end point in the scheme
    # hence the matrix has the following dimensions: 
    matrix = np.zeros((nx+ 1,nx + 1))
    
    
    for i in range(nx+1):    
        matrix[i,i] = 1 + c**2/2
        matrix[i, (i-1)%nx] = -c**2/4
        matrix[i, (i+1)%nx] = -c**2/4
        
    # note because of the way the matrix has been constructed with a modulus operator 
    # still have periodic boundaries
    
    # loop over timesteps
    for it in range(nt):
        # construct vector uvector such that matrix*h = uvector        
        uvector = np.zeros_like(uOld)
        for i in range(nx + 1):
            uvector[i] = -sqrt(g/H)*c*(hOld[(i + 1)%nx] - hOld[i%nx]) + \
               (c**2/4)*uOld[(i+1)%nx] + (1-c**2/2)*uOld[i%nx] + (c**2/4)*uOld[(i-1)%nx]
        
        # note here that uvector[nx] = uvector[0] because of modulus operator so 
        # periodic boundaries are still kept
        
        # solve matrix equation to find u
        u = np.linalg.solve(matrix, uvector)

        # construct vector hvector such that matrix*u = hvector
        hvector = np.zeros_like(hOld)
        
        for i in range(nx + 1):
            hvector[i] = -sqrt(H/g)*c*(uOld[i%nx] - uOld[(i-1)%nx]) + \
               (c**2/4)*hOld[(i+1)%nx] + (1-c**2/2)*hOld[(i)%nx] + (c**2/4)*hOld[(i-1)%nx]
        
        # note here that hvector[nx] = hvector[0] because of modulus operator so 
        # periodic boundaries are still kept
        
        # solve matrix equation to find h
        h = np.linalg.solve(matrix, hvector)

        # copy u and h for next iteration
        uOld = u.copy()
        hOld = h.copy()

    return u, h, x