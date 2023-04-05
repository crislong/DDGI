import matplotlib.pyplot as plt
import scipy
import numpy as np
import math

#
# functions to solve the dispersion relation
#

def solution(a, b, c):
    
    ''' Solve a 2nd order algebric equation    
    '''
    return (-b + np.sqrt(b**2 - 4* a * c))/(2*a)


# terms for the Bertin & Romeo dispersion relation (no drag)

def A(xi):

    return xi/16

def B(lamb, epsilon, xi):

    return lamb *((xi+1)*lamb - (epsilon+xi))/4

def C(lamb, epsilon):

    return lamb**4 -lamb**3 *(epsilon+1)

# terms for the Longarini 2022 dispersion relation without backreaction

def B_st(lamb, epsilon, xi, st):

    return lamb *((xi + 1 + 1/(st**2))*lamb - (epsilon + xi))/4

def C_st(lamb, epsilon,st):

    return lamb**4 * (1 + st**(-2)) - lamb**3 * (1 + epsilon + (1+epsilon)/(st**2))

# terms for the Longarini 2022 dispersion relation with backreaction

def B_st_BR(lamb, epsilon, xi, st):

    return lamb *((xi + 1 + 1/(st**2) + epsilon*(1+xi) /(st**2) + 
                   epsilon**2 * xi /(st**2))*lamb - (epsilon+xi))/4

def C_st_BR(lamb, epsilon,st):

    return lamb**4 * (1 + st**(-2) + 2*epsilon/(st**2) + epsilon**2 /(st**2)
                     )-lamb**3 *(epsilon+1 + (1+epsilon)/(st**2) + (2*epsilon*(1+epsilon))/(st**2) 
                                +(1+epsilon)*epsilon**2 /(st**2))


def marg_stab(epsilon, xi, st, Q):

    ''' Marginal stability curve as a function of the dimensionless perturbation wavelength
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    epsilon = dtg ratio
    xi = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    '''
    
    lamb = np.linspace(0,1+epsilon, 500)
    if (Q == 0):
        return [lamb,solution(A(xi), B(lamb, epsilon, xi), C(lamb, epsilon))]
    
    if (Q == 1):
        return [lamb,solution(A(xi), B_st(lamb, epsilon, xi, st), C_st(lamb, epsilon, st))]
    
    if (Q == 2):
        return [lamb,solution(A(xi), B_st_BR(lamb, epsilon, xi, st), C_st_BR(lamb, epsilon, st))]
    
    else:
        print('Error with Q')
        return 0
    

def const_jeanslength(hr, mstar, radius, sigmatot):

    return 2 * hr**2 * mstar/(radius * sigmatot)


def most_unst_wavelength_dless(epsilon, xi, st, Q):

    ''' Most unstable wavelength in dimensionless unit 
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    epsilon = dtg ratio
    xi = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    '''
    
    return marg_stab(epsilon, xi, st, Q)[0][np.argmax(marg_stab(epsilon, xi, st, Q)[1])]


def most_unst_wavelength_dimensional(epsilon, xi, st, Q, hr, mstar, radius, sigmatot):

    ''' Most unstable wavelength in physics unit 
    epsilon = dtg ratio
    xi = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    hr = aspect ratio at the radius of interest
    radius = radius of interest
    sigmatot = total surface density at the radius of interest
    '''

    return most_unst_wavelength_dless(epsilon, xi, st, Q
                ) * 2 * const_jeanslength(hr, mstar, radius, sigmatot)


def jeans_mass_dless(epsilon, xi, st, Q):

    ''' Jeans mass in dimensionless units 
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    epsilon = dtg ratio
    xi = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    '''

    return most_unst_wavelength_dless(epsilon, xi, st, Q) ** 2


def jeans_mass_dimensional(epsilon, xi, st, Q, hr, mstar, radius, sigmatot):

    ''' Jeans mass in physical units
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    epsilon = dtg ratio
    xi = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    hr = aspect ratio at the radius of interest
    radius = radius of interest
    sigmatot = total surface density at the radius of interest
    '''

    return sigmatot * most_unst_wavelength_dimensional(
        epsilon, xi, st, Q, hr, mstar, radius, sigmatot)**2