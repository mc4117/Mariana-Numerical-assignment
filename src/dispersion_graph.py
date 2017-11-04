#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare
"""
import numpy as np
import matplotlib.pyplot as plt
import math


# set courant number to be equal to 0.4
c = 0.4
x = np.linspace(0, math.pi, 1000)

# dispersion relation for explicit co-located scheme
omega1 = (2/c) * np.arcsin((c/2)*(np.sin(x)))

# dispersion relation for implicit co-located scheme
omega2 = (1/c) * np.arctan(c*np.sin(x))

# dispersion relation for explicit staggered scheme
omega3 = (2/c) * np.arcsin(c*np.sin(x/2))

# dispersion relation for implicit staggered scheme
omega4 = (1/c) * np.arctan((2*c*np.sin(x/2))/ (1 - (c**2*np.sin(x/2)**2)))

# plot dispersion relation on graph
plt.plot(x, omega1, label = "Explicit co-located")
plt.plot(x, omega2, label = "Implicit co-located")
plt.plot(x, omega3, label = "Explicit staggered")
plt.plot(x, omega4, label = "Semi-implicit staggered")
plt.plot(x, x, 'k--', label = "Exact analytic")
plt.legend(loc = 'best')
plt.xlabel(r"$k \Delta x$")
plt.ylabel(r"$\omega \Delta x$")
plt.xlim([0, math.pi])
plt.ylim([0, math.pi])
#plt.title("Postive branch of dispersion relation for \n analytic solution and numerical schemes \n")
plt.show()