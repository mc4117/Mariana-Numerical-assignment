#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Mariana Clare

Function which plots the dispersion relations for the 4 numerical schemes considered
and the analytic solution.

"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

def main():
    # set courant number to be equal to 0.4
    c = 0.4
    x = np.linspace(0, pi, 1000)

    # dispersion relation for analytic solution 
    omega = x
    
    # dispersion relation for explicit co-located scheme
    omega1 = (2/c) * np.arcsin(c/2*np.sin(x))

    # dispersion relation for implicit co-located scheme
    omega2 = (1/c) * np.arctan(c*np.sin(x))

    # dispersion relation for explicit staggered scheme
    omega3 = (2/c) * np.arcsin(c*np.sin(x/2))

    # dispersion relation for implicit staggered scheme
    omega4 = (1/c) * np.arctan(2*c*np.sin(x/2)/ (1 - c**2*np.sin(x/2)**2))

    # plot dispersion relation on graph
    plt.plot(x, omega1, label = "Explicit co-located")
    plt.plot(x, omega2, label = "Implicit co-located")
    plt.plot(x, omega3, label = "Explicit staggered")
    plt.plot(x, omega4, label = "Implicit staggered")
    plt.plot(x, omega, 'k--', label = "Analytic")
    plt.legend(loc = 'best')
    plt.xlabel(r"$k \Delta x$")
    plt.ylabel(r"$\omega \Delta x$")
    plt.xlim([0, pi])
    plt.ylim([0, pi])
#    plt.title("Postive branches of dispersion relations")
    plt.savefig("dispersion_relations.png")
    plt.show()
    
main()
